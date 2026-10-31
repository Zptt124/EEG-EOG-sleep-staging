"""Step W2-B1: 由 8 人开发验证集的预处理 NPZ 生成"应有样本清单"(只读 labels / epoch_index)。

用途:evaluate.py --reference 用它核对任何一张 predictions_val.csv 是否覆盖完整样本、
y_true 是否与真值一致(第二周协议:重复 ID、缺样本、标签不一致必须报错)。
只打开 split=val 的记录;训练 / 测试 NPZ 不读;信号数组不读。

用法:
  python scripts/09_make_val_reference.py --processed data/processed \
      --out manifests/val_reference_v1.csv
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from common import BASE, PROCESSED, SPLIT

FIELDS = ["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]


def val_records(split_csv):
    with open(split_csv, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    return [(r["subject_id"], rec) for r in rows for rec in r["record_ids"].split(";")]


def build_reference(processed, split_csv):
    out = []
    for subject, rec in val_records(split_csv):
        with np.load(Path(processed) / f"{rec}.npz", allow_pickle=False) as d:
            labels, idx = d["labels"], d["epoch_index"]  # 只取这两个成员
        assert labels.shape == idx.shape and ((labels >= 0) & (labels <= 4)).all()
        out += [(f"{rec}_{int(i):04d}", subject, rec, int(i), int(y)) for i, y in zip(idx, labels)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--processed", default=str(PROCESSED))
    ap.add_argument("--split", default=str(SPLIT / "split_subjects.csv"))
    ap.add_argument("--out", default=str(SPLIT / "val_reference_v1.csv"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"拒绝覆盖 {out}")
    rows = build_reference(args.processed, args.split)
    with out.open("x", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FIELDS)
        w.writerows(rows)
    print(f"{len(rows)} 段, {len({r[1] for r in rows})} 人, "
          f"sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
