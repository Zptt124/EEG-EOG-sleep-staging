"""Step W4-A1: 冻结 EEGDM 特征提取 · 1 秒 / 2 秒 std pooling 窗口对照。

基于 B 的 scripts/26_b_extract_eegdm_w3.py,只改窗口相关部分:
- 新增 --window-sec {1,2}:窗口秒数。window_size = window_step = 200 * w(200 Hz 采样)。
  L=6000 native 模式下 n_pool = 30 / w(1 秒 30 个位置,2 秒 15 个位置)。
- 其余完全一致:官方 TUEV checkpoint(SHA 固定)、condition 14、query gate、20 层、
  std(reduction="std, torch correction=1")、diffusion_t=1、无随机输入加噪。
- 数据权限:只读 label25(14 人,sseed2024,含 label10 的 5 人)训练者与 8 名验证者;不读测试集。
- 池化口径与 26 号一致:窗口特征 float16,--save-full 保存 [n,20,30/w,128],
  转 float32 对时间位置取均值拉平为 20*128=2560(两种窗口的池化维度相同)。

用法(仓库根目录):
  1 秒(与 W3 口径一致):
    python scripts/36_a_w4_window_extract.py --modality eeg --window-sec 1 --reduce std --diffusion-t 1 --save-full --cache-kernels
  2 秒:
    python scripts/36_a_w4_window_extract.py --modality eeg --window-sec 2 --reduce std --diffusion-t 1 --save-full --cache-kernels
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
C_SCRIPT = REPO / "experiment3/week2/scripts/extract_features.py"
SUBSETS = REPO / "manifests/label_subsets_v2.csv"
SAMPLING_SEED = "2024"
VERSION = "W4_A_features_v01"
B26_SHA = "inlined-adaptation"

spec = importlib.util.spec_from_file_location("c_extract", C_SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def allowed_subjects(budget):
    """预算内训练者 + 全部验证者;测试者永不进入。"""
    train = {r["subject_id"] for r in c.read_csv(SUBSETS)
             if r["sampling_seed"] == SAMPLING_SEED and r[budget] == "1"}
    split = {r["subject_id"]: r["split"] for r in c.read_csv(c.SOURCE / "manifests/split_subjects.csv")}
    if not train or any(split[s] != "train" for s in train):
        raise ValueError(f"{budget} 名单为空或含非训练者")
    return train | {s for s, p in split.items() if p == "val"}


def checkpoint_timesteps():
    """从 checkpoint 自带超参里找 num_train_timesteps;官方 conf/pretrain/base.yaml 为 50。"""
    hp = torch.load(c.ROOT / "checkpoints/backbone.ckpt", map_location="cpu", weights_only=False)["hyper_parameters"]
    found = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "num_train_timesteps":
                    found.append((f"{path}.{k}", int(v)))
                walk(v, f"{path}.{k}")
    walk(dict(hp), "hyper_parameters")
    return found


class Extractor(c.Extractor):
    def __init__(self, condition, reduce, diffusion_t, window_sec):
        super().__init__(condition)
        w = 200 * window_sec
        self.window_sec = window_sec
        self.extractor = c.LatentActivityExtractor(self.model, diffusion_t=diffusion_t, query=["gate"],
                                                   use_cond=[condition]).eval().cuda()
        # 官方 reducer 断言 (L - window_size) % window_step == 0:
        # 1 秒窗在 1000/6000 上都成立;2 秒窗(400/400)只在 6000(native 模式,15 个位置)上成立。
        # 提取只用 native 模式,跳过不合法的 1000 档。
        self.reducers = {
            n: c.LatentActivityReducer(query=["gate"], reduce=[reduce], L=n,
                                       window_size=w, window_step=w).eval().cuda()
            for n in (1000, 6000) if (n - w) % w == 0
        }

    @torch.inference_mode()
    def transformed(self, x, mode="native"):
        positions = 30 // self.window_sec
        t = torch.from_numpy(x).cuda()
        if mode == "native":
            y = self.reducers[6000](self.extractor((t, None)))[:, :, 0, :, 0, 0, :]
        elif mode == "explicit" and 1000 in self.reducers:
            y = torch.cat([self.reducers[1000](self.extractor((t[:, :, i:i + 1000], None)))
                           [:, :, 0, :, 0, 0, :] for i in range(0, 6000, 1000)], dim=2)
        else:
            raise ValueError(f"mode={mode} 不支持窗口 {self.window_sec}s(2 秒窗口只用 native/6000)")
        y = y.cpu().numpy()
        if y.shape != (len(x), 20, positions, 128) or not np.isfinite(y).all():
            raise ValueError('Invalid feature output')
        return y


def config(args, timesteps):
    w = 200 * args.window_sec
    positions = 30 // args.window_sec
    cfg = c.config(args.modality, args.condition)
    cfg.update(version=VERSION, diffusion_t=args.diffusion_t, random_noise=False,
               reduction=f"{args.reduce}, torch" + (" correction=1" if args.reduce == "std" else ""),
               checkpoint_num_train_timesteps=timesteps, subjects=f"{args.budget}(sseed{SAMPLING_SEED}) + val",
               window_seconds=args.window_sec, window_points=w, stride_points=w,
               feature_shape=[20, positions, 128],
               pooled=f"float16 {args.window_sec}s-window -> float32 mean over {positions} time positions -> 2560",
               full_cache_saved=args.save_full, w4_script_sha256=sha256(__file__),
               c_script_sha256=sha256(C_SCRIPT))
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["eeg", "eog"], required=True)
    p.add_argument("--window-sec", type=int, choices=[1, 2], required=True)
    p.add_argument("--reduce", choices=["std", "mean"], required=True)
    p.add_argument("--diffusion-t", type=int, required=True)
    p.add_argument("--condition", type=int, choices=[10, 14, 18], default=14)
    p.add_argument("--budget", choices=["label10", "label25", "label50", "label100"], default="label25")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--save-full", action="store_true")
    p.add_argument("--cache-kernels", action="store_true")
    p.add_argument("--record-limit", type=int)
    p.add_argument("--reference-npz", type=Path, help="同配置的既有池化 NPZ,逐 sample_id 核对数值")
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    timesteps = checkpoint_timesteps()
    print(json.dumps({"checkpoint_num_train_timesteps": timesteps}), flush=True)
    for _, n in timesteps:
        if not 0 <= args.diffusion_t < n:
            raise ValueError(f"diffusion_t={args.diffusion_t} 超出 checkpoint 步数 {n}")

    tag = f"{args.reduce}_t{args.diffusion_t}_w{args.window_sec}s_cond{args.condition}_{args.budget}"
    out = args.output or REPO / "experiment3/week4/pool_compare/features" / f"{args.modality}_{tag}"
    if args.record_limit:
        out = out.with_name(out.name + f"_smoke{args.record_limit}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = config(args, timesteps)
    cfgpath = out / "feature_config.json"
    if cfgpath.exists() and json.loads(cfgpath.read_text(encoding="utf-8")) != cfg:
        raise ValueError("输出目录配置不一致;请换新目录")
    cfgpath.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    torch.manual_seed(0)
    extractor = Extractor(args.condition, args.reduce, args.diffusion_t, args.window_sec)
    if args.cache_kernels:
        extractor.cache_frozen_kernels()
    subjects = allowed_subjects(args.budget)
    records = [(r, s) for r, s in sorted(c.record_splits().items())
               if s in ("train", "val") and r[:5] in subjects]
    if args.record_limit:
        records = records[:args.record_limit]

    positions = 30 // args.window_sec
    index, profiles, ids, pooled_parts = [], [], [], []
    for rec, partition in records:
        target, done = out / f"{rec}.pooled.npy", out / f"{rec}.done.json"
        with np.load(c.DATA / f"{rec}.npz", allow_pickle=False) as z:
            labels, epochs = z["labels"], z["epoch_index"]
            n = len(labels)
            if done.exists():
                profile = json.loads(done.read_text(encoding="utf-8"))
            else:
                signals = z["signal"][:, 0 if args.modality == "eeg" else 1, :]
                pooled = np.empty((n, 2560), dtype=np.float32)
                full = (np.lib.format.open_memmap(out / f"{rec}.partial.npy", mode="w+", dtype=np.float16,
                                                  shape=(n, 20, positions, 128)) if args.save_full else None)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for b in range(0, n, args.batch_size):
                    stored = extractor(signals[b:b + args.batch_size]).astype(np.float16)
                    if not np.isfinite(stored).all():
                        raise ValueError("float16 overflow")
                    if full is not None:
                        full[b:b + len(stored)] = stored
                    pooled[b:b + len(stored)] = stored.astype(np.float32).mean(axis=2).reshape(len(stored), 2560)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if full is not None:
                    full.flush()
                    del full
                    (out / f"{rec}.partial.npy").replace(out / f"{rec}.npy")
                np.save(target, pooled)
                profile = dict(record_id=rec, split=partition, n_samples=n, seconds=elapsed,
                               samples_per_second=n / elapsed, peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20,
                               batch_size=args.batch_size, cache_frozen_kernels=args.cache_kernels)
                done.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            pooled = np.load(target, allow_pickle=False)
            if pooled.shape != (n, 2560) or pooled.dtype != np.float32 or not np.isfinite(pooled).all():
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
    npz = out / f"{args.modality.upper()}_{tag}_pooled_{len(ids)}x2560_unstandardized.npz"
    np.savez(npz, sample_id=np.array(ids), x=x)
    c.write_csv(out / "sample_index.csv", index)
    c.write_csv(out / "resource_profile.csv", profiles)

    check = None
    if args.reference_npz:
        with np.load(args.reference_npz, allow_pickle=False) as ref:
            pos = {s: i for i, s in enumerate(ref["sample_id"])}
            missing = [s for s in ids if s not in pos]
            if missing:
                raise ValueError(f"参考 NPZ 缺 {len(missing)} 个 sample_id")
            diff = np.abs(ref["x"][[pos[s] for s in ids]] - x)
            check = dict(reference=str(args.reference_npz), reference_sha256=sha256(args.reference_npz),
                         rows=len(ids), max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()))
            print(json.dumps(check), flush=True)

    summary = dict(status="complete" if not args.record_limit else "smoke", records=len(records), samples=len(ids),
                   subjects=sorted({r["subject_id"] for r in index}), test_used=False,
                   split_counts={s: sum(r["split"] == s for r in index) for s in ("train", "val")},
                   total_seconds=sum(p["seconds"] for p in profiles), npz=npz.name, reference_check=check)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [npz, out / "sample_index.csv", cfgpath]
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(f)}  {f.name}\n" for f in files), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
