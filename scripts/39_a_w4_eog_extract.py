"""Step W4-A4: 从跨模态 30k checkpoint 提取 EOG-only 下游特征(空 EEG 条件)。

第四周跨模态三组的下游统一只输入 EOG、EEG 条件为空。本脚本从 A 的同窗 EEG 30k
checkpoint 提取 EOG std t=1 池化特征(2560 维),供 B 的 34 号探针评分。

与 B 26 号提取的唯一区别是模型来源:W4 的 m64x12 同窗模型(EMA 权重)。空条件前向与
官方前向逐位一致——w4d_model 的 xc_flag 为零初始化 Embedding(空条件加 0)、xc_in 只在
给定 condition_eeg 时生效,官方 LatentActivityExtractor 直接逐层调用 calc_cond/in_layer/
layers,不经 w4_forward,两者在空条件下等价。提取器因此完全复用官方代码,不改任何 vendor 源文件。

数据权限:只读 label25(14 人,sseed2024)+ 8 名验证者(53,496 段);不读测试集。
验收(协议要求「有无 EEG 文件两种环境核对一致」):--zero-eeg 把 NPZ 的 EEG 通道置零后
重跑,EOG 特征应与正常跑逐值一致(输出 SHA 相同)。

用法(仓库根目录):
  正常:
    python scripts/39_a_w4_eog_extract.py --run-dir experiment4/week4/runs/W4_A_xmodal_same_30k_seed0_v01
  验收(EEG 通道置零):
    python scripts/39_a_w4_eog_extract.py --run-dir experiment4/week4/runs/W4_A_xmodal_same_30k_seed0_v01 --zero-eeg
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
W4D = REPO / "experiment4/week4/src"
sys.path.insert(0, str(W4D))
sys.path.insert(0, str(REPO / "experiment3/week2/vendor/EEGDM"))
from model.classifier import LatentActivityExtractor, LatentActivityReducer  # noqa: E402

sys.path.insert(0, str(REPO / "experiment3/week2/scripts"))
import extract_features as c  # noqa: E402  C 的提取管线(transform/record_splits/write_csv)

import w4d_common as C  # noqa: E402
import w4d_model as M  # noqa: E402

VERSION = "W4_A_eog_xmodal_v01"


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_ema_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(set(state) >= {"cfg_name", "model_kwargs", "seed", "step", "model", "ema_model"},
            f"Unexpected checkpoint structure: {sorted(state)}")
    require(state["cfg_name"] == "m64x12" and state["step"] == 30000,
            f"Unexpected config/step: {state['cfg_name']} step={state['step']}")
    require(state["model_kwargs"] == C.model_kwargs("m64x12"), "model_kwargs differs from m64x12")
    m = M.build_model("m64x12", seed=state["seed"], cross_modal="same_window")
    missing, unexpected = m.load_state_dict(state["ema_model"], strict=False)
    require(not missing and not unexpected, f"EMA load failed: missing={missing} unexpected={unexpected}")
    return m.eval().to(device)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True, help="W4_A run 目录(runs/ 下的小文件目录)")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO / "experiment4/week4/big/runs/W4_A_xmodal_same_30k_seed0_v01/ckpt_step30000.pt")
    p.add_argument("--budget", choices=["label10", "label25"], default="label25")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--zero-eeg", action="store_true", help="验收:EEG 通道置零后重跑")
    p.add_argument("--record-limit", type=int)
    p.add_argument("--output", type=Path,
                   default=REPO / "experiment3/week4/pool_compare/features/eog_std_t1_w4_xmodal_label25")
    args = p.parse_args()

    tag = "zeroeeg" if args.zero_eeg else "normal"
    out = args.output.with_name(args.output.name + f"_{tag}") if args.zero_eeg else args.output
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.set_num_threads(4)
    model = load_ema_model(args.checkpoint, device)
    # m64x12 的 n_class=2:条件 = 模态 ID(0=EEG,1=EOG),与第三周官方主干的 14 通道不同
    extractor = LatentActivityExtractor(model, diffusion_t=1, query=["gate"],
                                        use_cond=[1]).eval().cuda()
    reducer = LatentActivityReducer(query=["gate"], reduce=["std"], L=6000,
                                    window_size=200, window_step=200).eval().cuda()

    subjects = c.read_csv(REPO / "manifests/label_subsets_v2.csv")
    train = {r["subject_id"] for r in subjects
             if r["sampling_seed"] == "2024" and r[args.budget] == "1"}
    split = {r["subject_id"]: r["split"]
             for r in c.read_csv(REPO / "manifests/split_subjects.csv")}
    allowed = train | {s for s, part in split.items() if part == "val"}
    records = [(r, s) for r, s in sorted(c.record_splits().items())
               if s in ("train", "val") and r[:5] in allowed]
    if args.record_limit:
        records = records[:args.record_limit]

    n_layers = len(model.layers)          # m64x12:12 层
    d_model = model.d_model                # m64x12:64
    pooled_dim = n_layers * d_model
    cfg = dict(version=VERSION, checkpoint=str(args.checkpoint),
               checkpoint_sha256=sha256(args.checkpoint), model="m64x12 xmodal same_window EMA step30000",
               modality="eog", condition="modality id 1 (EOG); empty EEG condition (official forward equivalence)",
               diffusion_t=1, random_noise=False, query="gate", layers=list(range(n_layers)),
               reduction="std, torch correction=1", window_points=200, stride_points=200,
               length_points=6000, axes=["sample", "layer", "time_window", "hidden"],
               feature_shape=[n_layers, 30, d_model], pooled_dim=pooled_dim, storage_dtype="float16",
               internal_segments=6, continuous_30s_context=False, target_fs=200,
               subjects=f"{args.budget}(sseed2024) + val", zero_eeg=args.zero_eeg,
               run_config=str(args.run_dir / "config.json"),
               script_sha256=sha256(__file__))
    cfgpath = out / "feature_config.json"
    if cfgpath.exists() and json.loads(cfgpath.read_text(encoding="utf-8")) != cfg:
        raise ValueError("输出目录配置不一致;请换新目录")
    cfgpath.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    index, profiles, ids, pooled_parts = [], [], [], []
    for rec, partition in records:
        target, done = out / f"{rec}.pooled.npy", out / f"{rec}.done.json"
        with np.load(c.DATA / f"{rec}.npz", allow_pickle=False) as z:
            labels, epochs = z["labels"], z["epoch_index"]
            n = len(labels)
            if done.exists():
                profile = json.loads(done.read_text(encoding="utf-8"))
            else:
                signal_all = z["signal"]
                if args.zero_eeg:
                    signal_all = signal_all.copy()
                    signal_all[:, 0, :] = 0.0
                signals = signal_all[:, 1, :]
                pooled = np.empty((n, pooled_dim), dtype=np.float32)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for b in range(0, n, args.batch_size):
                    x = c.transform(signals[b:b + args.batch_size])
                    t = torch.from_numpy(x).cuda()
                    y = reducer(extractor((t, None)))[:, :, 0, :, 0, 0, :].cpu().numpy()
                    if y.shape != (x.shape[0], n_layers, 30, d_model) or not np.isfinite(y).all():
                        raise ValueError("Invalid feature output")
                    stored = y.astype(np.float16)
                    if not np.isfinite(stored).all():
                        raise ValueError("float16 overflow")
                    pooled[b:b + len(stored)] = stored.astype(np.float32).mean(axis=2).reshape(len(stored), pooled_dim)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                np.save(target, pooled)
                profile = dict(record_id=rec, split=partition, n_samples=n, seconds=elapsed,
                               samples_per_second=n / elapsed,
                               peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20,
                               batch_size=args.batch_size, zero_eeg=args.zero_eeg)
                done.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            pooled = np.load(target, allow_pickle=False)
            if pooled.shape != (n, pooled_dim) or pooled.dtype != np.float32 or not np.isfinite(pooled).all():
                raise ValueError(f"池化缓存异常 {rec}")
            pooled_parts.append(pooled)
            profiles.append(profile)
            for epoch, label in zip(epochs, labels):
                sid = f"{rec}_{int(epoch):04d}"
                ids.append(sid)
                index.append(dict(sample_id=sid, subject_id=rec[:5], record_id=rec, epoch_index=int(epoch),
                                  split=partition, y_true=int(label)))
            print(json.dumps(profile), flush=True)

    if len(set(ids)) != len(ids):
        raise ValueError("sample_id 重复")
    x = np.concatenate(pooled_parts)
    npz = out / f"EOG_std_t1_w4_xmodal_pooled_{len(ids)}x{pooled_dim}_unstandardized.npz"
    np.savez(npz, sample_id=np.array(ids), x=x)
    c.write_csv(out / "sample_index.csv", index)
    c.write_csv(out / "resource_profile.csv", profiles)
    summary = dict(status="complete", records=len(records), samples=len(ids),
                   subjects=sorted({r["subject_id"] for r in index}), test_used=False,
                   split_counts={s: sum(r["split"] == s for r in index) for s in ("train", "val")},
                   total_seconds=sum(pr["seconds"] for pr in profiles), npz=npz.name,
                   npz_sha256=sha256(npz), zero_eeg=args.zero_eeg)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [npz, out / "sample_index.csv", cfgpath]
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(f)}  {f.name}\n" for f in files), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
