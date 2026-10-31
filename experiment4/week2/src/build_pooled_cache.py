#build_pooled_cache.py —— 把 EEG/EOG 原始冻结特征池化成 2560 维缓存（R1 表征，只需跑一次）
#全部 154,124 段（训练 54 人 + 验证 8 人）都存，包括无标签者；标签隐藏在训练入口 label_guard 里做
#缓存行顺序 = EEG feature_index.csv 的行顺序；EOG 按 sample_id 对齐到同一顺序
#这里只做池化，不做标准化（标准化要按预算只用有标签者拟合，在 run_ssl.py 里做）
#用法（Ubuntu）：python -u build_pooled_cache.py --eeg EEG特征文件夹 --eog EOG特征文件夹 --output 缓存文件夹

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ssl_common import load_pooled, read_csv, sha256

INDEX_COLS = ["sample_id", "subject_id", "record_id", "epoch_index", "split", "y_true"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg", type=Path, required=True)
    ap.add_argument("--eog", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if (args.output / "cache_manifest.json").exists():
        print(f"ERROR: {args.output} already has a cache, choose a new folder")
        sys.exit(1)
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows_eeg = read_csv(args.eeg / "feature_index.csv")
    rows_eog = read_csv(args.eog / "feature_index.csv")
    #按 sample_id 把 EOG 对齐到 EEG 的顺序，并核对关键字段
    eog_by_id = {r["sample_id"]: r for r in rows_eog}
    if len(eog_by_id) != len(rows_eog) or set(eog_by_id) != {r["sample_id"] for r in rows_eeg}:
        print("ERROR: EEG/EOG sample_id sets differ or EOG has duplicates")
        sys.exit(1)
    rows_eog = [eog_by_id[r["sample_id"]] for r in rows_eeg]
    for a, b in zip(rows_eeg, rows_eog):
        for key in ("subject_id", "record_id", "epoch_index", "split", "y_true"):
            if a[key] != b[key]:
                print(f"ERROR: {key} mismatch at {a['sample_id']}")
                sys.exit(1)
    print(f"OK: {len(rows_eeg)} samples aligned by sample_id", flush=True)

    #索引表（行顺序与两份缓存一致）
    with (args.output / "sample_index.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(INDEX_COLS) + "\n")
        for r in rows_eeg:
            f.write(",".join(r[k] for k in INDEX_COLS) + "\n")

    shapes = {}
    for name, folder, rows in (("eeg", args.eeg, rows_eeg), ("eog", args.eog, rows_eog)):
        print(f"Pooling {name.upper()} features from {folder} ...", flush=True)
        x = load_pooled(folder, rows)
        np.save(args.output / f"{name}_pooled.npy", x)
        shapes[name] = list(x.shape)
        print(f"Saved {name}_pooled.npy {x.shape} {x.dtype}, mean {x.mean():.4f}, std {x.std():.4f}", flush=True)
        del x  #先释放 EEG 再读 EOG，峰值内存约 1.6 GB

    print("Computing SHA256 of cache files ...", flush=True)
    manifest = {
        "representation": "R1_EEGDM_raw_frozen_cond14_timepooled",
        "pooling": "mean over 30 windows of [20,30,128] -> 2560 per modality",
        "row_order": "EEG feature_index.csv order; EOG aligned by sample_id",
        "n_samples": len(rows_eeg),
        "shapes": shapes,
        "dtype": "float32",
        "source_eeg": str(args.eeg), "source_eog": str(args.eog),
        "source_eeg_index_sha256": sha256(args.eeg / "feature_index.csv"),
        "source_eog_index_sha256": sha256(args.eog / "feature_index.csv"),
        "files_sha256": {n: sha256(args.output / n) for n in ("sample_index.csv", "eeg_pooled.npy", "eog_pooled.npy")},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with (args.output / "cache_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"Done in {manifest['minutes']} min")


if __name__ == "__main__":
    main()
