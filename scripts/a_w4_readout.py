"""W4 A: 1 秒 / 2 秒窗口对照用的 readout 变体(基于 C 的第三周 LFT 入口)。

C 的 experiment3/week3/scripts/run_week3_c_readout.py 固定 n_pool=30、特征形状
[n,20,30,128]、协议 feature_shape=[20,30,128]。本模块只把这些位置参数化:
2 秒窗口对应 15 个时间位置,其余(LFT 结构、tokens、训练配方)完全不变。

用法:由 scripts/37_a_w4_lft_compare.py 导入;不要单独运行。
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiment3/week3/scripts"))
sys.path.insert(0, str(REPO / "experiment3/week4/pool_compare/_compat"))  # fcntl 垫片(仅 Windows)
import run_week3_c_readout as c  # noqa: E402  C 的 W3 readout 入口(冻结 EEGDM LFT)

W4_FEATURES = REPO / "experiment3/week4/pool_compare/features"


def w4_sources(window_sec, signal_rate=200):
    """第四周 1s/2s/3s 特征目录(EEG/EOG 各一,含 feature_index.csv 与逐记录 npy)。

    signal_rate=100 时是 downsampling 变体目录:信号以原生 100 Hz 直接进入主干
    (跳过 2× 重采样),window_points = 100 * window_sec,其余口径不变。
    """
    suffix = "" if signal_rate == 200 else f"_ds{signal_rate}"
    return {
        "EEG": W4_FEATURES / f"eeg_std_t1_w{window_sec}s{suffix}_cond14_label25",
        "EOG": W4_FEATURES / f"eog_std_t1_w{window_sec}s{suffix}_cond14_label25",
    }


def n_positions(window_sec):
    return 30 // window_sec


class W4CachedLFT(c.CachedLFT):
    """n_pool 可参数的 LFT(CachedLFT 固定 30;2 秒窗口为 15)。"""

    def __init__(self, n_layers, n_tokens, n_pool):
        super().__init__(n_layers, n_tokens)
        self.decoder = c.LatentActivityDecoder(
            d_model=128, d_kv_embed_factor=1, n_layer=n_layers, n_query=1,
            n_pool=n_pool, pool_merge="share", multi_query_merge="seq",
            ch_order=list(c.MODALITIES), have_ch_pos_embed=True,
            cat_ch_pos_embed=False, ch_pos_emb_sym=None,
            clst_dim="TP", clst_pos_embed_dim="", n_clst=n_tokens,
            stack_struct="scf", num_heads=8, ff=4, have_crossnorm=False,
            across_pool_stack_struct=None,
        )
        self.classifier = c.TransformerClassifier(
            n_tower=1, n_pool=n_pool, n_clst=n_tokens, d_embed=128,
            have_pos_embed=True, pos_embed_dim="TPN", stack_struct="sfsfsfsf",
            num_heads=8, ff=4, final_act="pool", n_class=5,
        )


class W4LFTNoClassifier(torch.nn.Module):
    """W4CachedLFT 去掉 classifier module 的变体(提速方法②,只改这一项)。

    与 CachedLFT 完全相同的 decoder(LFT tokenizer,20 层);原 TransformerClassifier
    (sfsfsfsf 塔 + final_act="pool")替换为对 decoder 输出的 (T,P,N) 平均池化
    + 单层 Linear(128,5)。官方 classifier 的 pool 口径即 mean over (T,P,N),
    见 vendor/EEGDM/model/classifier.py final_act="pool"。
    """

    def __init__(self, n_layers, n_tokens, n_pool):
        super().__init__()
        self.decoder = c.LatentActivityDecoder(
            d_model=128, d_kv_embed_factor=1, n_layer=n_layers, n_query=1,
            n_pool=n_pool, pool_merge="share", multi_query_merge="seq",
            ch_order=list(c.MODALITIES), have_ch_pos_embed=True,
            cat_ch_pos_embed=False, ch_pos_emb_sym=None,
            clst_dim="TP", clst_pos_embed_dim="", n_clst=n_tokens,
            stack_struct="scf", num_heads=8, ff=4, have_crossnorm=False,
            across_pool_stack_struct=None,
        )
        self.head = torch.nn.Linear(128, 5)

    @staticmethod
    def to_official_axes(x):
        return W4CachedLFT.to_official_axes(x)

    def forward(self, x):
        y, _ = self.decoder(self.to_official_axes(x))  # B T P N H,T=1
        return self.head(y.mean(dim=(1, 2, 3)))        # B 128 -> B 5


class W4CachedEpochs(c.CachedEpochs):
    """CachedEpochs 的窗口参数化版本;groups() 只把硬编码 30 换成 n_positions。"""

    def __init__(self, rows, positions, layers, sources, n_positions):
        super().__init__(rows, positions, layers, sources)
        self.n_positions = n_positions

    def groups(self, block_size, blocks_per_group, rng=None):
        blocks = [(record, base + offset, offset, min(count - offset, block_size))
                  for record, base, count in self.blocks for offset in range(0, count, block_size)]
        if rng is not None:
            rng.shuffle(blocks)
        for start in range(0, len(blocks), blocks_per_group):
            selected = blocks[start:start + blocks_per_group]
            x = np.empty((sum(b[3] for b in selected), 2, self.n_layers, self.n_positions, 128),
                         dtype=np.float16)
            positions = np.empty(len(x), dtype=np.int64)
            end = 0
            for record, base, offset, count in selected:
                for ch, values in enumerate(self.maps[record]):
                    x[end:end + count, ch] = values[offset:offset + count, self.layer_slice]
                positions[end:end + count] = np.arange(base, base + count)
                end += count
            if rng is not None:
                order = rng.permutation(len(x))
                x, positions = x[order], positions[order]
            yield x, positions


def w4_build_heads(config, seed, device, n_pool, no_classifier=False):
    torch.manual_seed(seed)
    n_layers = len(config["layers"])
    linear = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2 * n_layers * 128, 5))
    lft = (W4LFTNoClassifier(n_layers, config["lft_tokens_per_second"], n_pool)
           if no_classifier else W4CachedLFT(n_layers, config["lft_tokens_per_second"], n_pool))
    return {name: model.to(device) for name, model in (("C0", linear), ("C1", lft))}


def w4_smoke(config, rows, train, device, sources, n_pool, no_classifier=False):
    subset = W4CachedEpochs(rows, train, config["layers"], sources, n_pool)
    rng = np.random.default_rng(0)
    size = config["batch_size"]
    raw, positions = next(subset.groups(size, 2, rng))
    x = torch.as_tensor(raw[:size], device=device, dtype=torch.float32)
    y = torch.as_tensor(subset.labels[positions[:size]], device=device)
    c.require(torch.isfinite(x).all().item(), "Nonfinite cached features in smoke batch")
    tokens = W4CachedLFT.to_official_axes(x)
    for ch in range(2):
        c.require(torch.equal(tokens[:, :, 0, :, 0, ch], x[:, ch]), "Cache axis mapping differs")
    out = dict(input_shape=list(x.shape), official_shape=list(tokens.shape),
               sample_ids=[subset.rows[i]["sample_id"] for i in positions[:size]], heads={})
    for name, model in w4_build_heads(config, 0, device, n_pool, no_classifier).items():
        optimizer = c.optimizer_for(model, config)
        layer = next(model.parameters())
        initial = layer.detach().clone()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = c.time.perf_counter()
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = c.logits_for(name, model, x)
                loss = c.F.cross_entropy(logits, y)
            c.require(logits.shape == (len(x), 5) and torch.isfinite(logits).all().item(),
                      f"{name} output invalid")
            loss.backward()
            c.require(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                          for p in model.parameters()), f"{name} missing/nonfinite gradient")
            optimizer.step()
        c.require(not torch.equal(layer, initial), f"{name} weights did not update")
        out["heads"][name] = dict(
            logits_shape=list(logits.shape), loss=float(loss.detach()), updates=4,
            seconds=c.time.perf_counter() - start, parameters=sum(p.numel() for p in model.parameters()),
            peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2**20
                                if device.type == "cuda" else None),
        )
    return out


def w4_inspect_inputs(dev_reference, sources, window_sec, signal_rate=200):
    """C.inspect_inputs 的窗口参数化版本;其余核验(索引对齐、开发集参考、形状)一致。"""
    npool = n_positions(window_sec)
    points_per_sec = 200 if signal_rate == 200 else 100
    configs = {name: c.json.loads((path / "feature_config.json").read_text(encoding="utf-8"))
               for name, path in sources.items()}
    protocol = dict(
        upstream_commit="003b5a2a016b2f02788ea5693d8b7cd00b6bd969",
        checkpoint_sha256="0fe2d385d074834f3c22d9458798da49b2e5a58bd817284f130aae8814f5c1ef",
        condition=14, diffusion_t=1, random_noise=False, query="gate",
        layers=list(range(20)), reduction="std, torch correction=1",
        axes=["sample", "layer", "time_window", "hidden"], feature_shape=[20, npool, 128],
        storage_dtype="float16",
        internal_segments=6 if signal_rate == 200 else 3,
        continuous_30s_context=False,
        target_fs=signal_rate,
        window_points=points_per_sec * window_sec, stride_points=points_per_sec * window_sec,
        window_seconds=window_sec,
    )
    for modality, cfg in configs.items():
        c.require(cfg["modality"] == modality.lower(), f"Wrong modality in {modality} cache")
        c.require(all(cfg[k] == v for k, v in protocol.items()), f"{modality} protocol differs")
    # 模态一致性只比较提取设置;脚本文本哈希在 input_sha256 审计里分别记录
    # (EEG 与 EOG 提取之间脚本只加过对 2 秒窗口的保护性分支,1 秒窗口的数值路径完全相同)。
    settings = lambda cfg: {k: v for k, v in cfg.items()
                            if k not in ("modality", "w4_script_sha256", "c_script_sha256")}
    c.require(settings(configs["EEG"]) == settings(configs["EOG"]),
              "EEG/EOG extraction settings differ")
    indices = {name: c.read_csv(path / "feature_index.csv") for name, path in sources.items()}
    split = c.read_csv(c.SPLIT)
    grouped = c.audit_index(indices["EEG"], split, strict=False)
    c.audit_index(indices["EOG"], split, strict=False)
    # W4 筛查口径:只覆盖 label25(14 人)+ 8 名验证者(与 B 第三周提取一致)。
    # 14 位有标签训练者共 27 条记录(其中一位只有一晚),验证者 16 条,合计 43 条。
    c.require(len(grouped) == 43, "Expected 43 records (label25 + val)")
    c.require(len({r["subject_id"] for r in indices["EEG"] if r["split"] == "train"}) == 14
              and len({r["subject_id"] for r in indices["EEG"] if r["split"] == "val"}) == 8,
              "Expected 14 train + 8 val subjects")
    c.require(sum(r["split"] == "train" for r in indices["EEG"]) == 32310
              and sum(r["split"] == "val" for r in indices["EEG"]) == 21186,
              "Expected 32310 train + 21186 val samples")
    c.align_indices(indices["EEG"], indices["EOG"])
    rows = indices["EEG"]
    subjects, train, dev = c.selected_positions(rows, c.read_csv(c.SUBSETS), "label25")
    keys = ("sample_id", "subject_id", "record_id", "epoch_index", "y_true")
    expected_dev = sorted(tuple(r[k] for k in keys) for r in c.read_csv(dev_reference))
    actual_dev = sorted(tuple(rows[i][k] for k in keys) for i in dev)
    c.require(actual_dev == expected_dev, "Development index differs from the shared reference")
    for record, items in grouped.items():
        for path in sources.values():
            values = np.load(path / f"{record}.npy", mmap_mode="r", allow_pickle=False)
            c.require(values.shape == (len(items), 20, npool, 128) and values.dtype == np.float16,
                      f"Invalid feature header: {path.name}/{record}")
    audit = dict(
        samples=len(rows), records=len(grouped),
        split_counts=c.Counter(r["split"] for r in rows),
        split_subject_counts=c.Counter(r["split"] for r in split),
        labeled_subjects=sorted(subjects), labeled_epochs=len(train), dev_epochs=len(dev),
        feature_protocol=protocol, modality_order=c.MODALITIES, test_used=False,
        eog_provenance="local regeneration under prior D reference protocol",
        window_seconds=window_sec, signal_rate=signal_rate,
        input_sha256={str(p.relative_to(REPO)).replace("\\", "/"): c.sha(p) for p in [
            c.SPLIT, c.SUBSETS, REPO / "src/evaluation/evaluate.py",
            c.UPSTREAM / "model/classifier.py", Path(__file__),
            *(folder / file for folder in sources.values()
              for file in ("feature_index.csv", "feature_config.json")),
        ]},
        development_reference_sha256=c.sha(dev_reference),
    )
    return rows, train, dev, audit
