"""Step W2-B3: 三段上下文窗口索引(前一段 / 中心 / 后一段),只读 NPZ 的 epoch_index。

规则(协议 v2「上下文」):
  - 按 subject_id、record_id、原始 epoch_index 组窗;不跨人、不跨夜。
  - 邻居必须是同一记录里 epoch_index 恰好 ±1 的真实片段;预处理剔除 Movement/Unknown 或
    首尾裁剪造成的缺口不跨越。邻居不存在时用中心段复制填充,previous_is_real / next_is_real = 0。
  - 所有方法共用同一中心样本集合(每条记录的全部片段都是中心);长度 1 与长度 3 的
    center_sample_id 逐行相同,填充不改变可评价样本。
  - 不读取任何标签,也不会把邻居真标签交给模型;训练窗只含 label25 选中的 14 人,
    验证窗只含 8 名开发验证者(只用其信号作上下文)。其余 40 名训练者不进入。

boundary_rule 取值:real(两侧真实)| pad_previous | pad_next | pad_both(单片段记录)

用法:
  python scripts/11_make_window_index.py --processed data/processed --out reports/week2/window_index.csv
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from common import PROCESSED, SPLIT

FIELDS = ["center_sample_id", "subject_id", "record_id", "epoch_index", "split", "nominal_budget",
          "previous_sample_id", "next_sample_id", "previous_is_real", "next_is_real", "boundary_rule"]


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sid(record, idx):
    return f"{record}_{int(idx):04d}"


def windows_for_record(subject, record, split, budget, epoch_index):
    """epoch_index:该记录保留下来的原始片段序号(升序、唯一)。返回窗口行列表。"""
    idx = np.asarray(epoch_index, dtype=np.int64)
    assert idx.ndim == 1 and (np.diff(idx) > 0).all(), f"{record}: epoch_index 必须严格递增"
    present = set(idx.tolist())
    rows = []
    for i in idx.tolist():
        prev_real, next_real = (i - 1) in present, (i + 1) in present
        rule = ("real" if prev_real and next_real else
                "pad_both" if not prev_real and not next_real else
                "pad_previous" if not prev_real else "pad_next")
        rows.append({"center_sample_id": sid(record, i), "subject_id": subject, "record_id": record,
                     "epoch_index": i, "split": split, "nominal_budget": budget,
                     "previous_sample_id": sid(record, i - 1) if prev_real else sid(record, i),
                     "next_sample_id": sid(record, i + 1) if next_real else sid(record, i),
                     "previous_is_real": int(prev_real), "next_is_real": int(next_real),
                     "boundary_rule": rule})
    return rows


def build(processed, split_csv, subsets_csv, budget):
    split = {r["subject_id"]: r for r in read(split_csv)}
    chosen = [r["subject_id"] for r in read(subsets_csv) if r[budget] == "1"]
    targets = [(s, "train", budget) for s in chosen]
    targets += [(s, "val", "NA") for s, r in split.items() if r["split"] == "val"]
    rows = []
    for subject, part, nominal in targets:
        assert split[subject]["split"] == part, f"{subject} 不在 {part}"
        for rec in split[subject]["record_ids"].split(";"):
            with np.load(Path(processed) / f"{rec}.npz", allow_pickle=False) as d:
                idx = d["epoch_index"]  # 只取 epoch_index,不读标签与信号
            rows += windows_for_record(subject, rec, part, nominal, idx)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--processed", default=str(PROCESSED))
    ap.add_argument("--split", default=str(SPLIT / "split_subjects.csv"))
    ap.add_argument("--subsets", default=str(SPLIT / "label_subsets_v2.csv"))
    ap.add_argument("--budget", default="label25", choices=["label10", "label25", "label50", "label100"])
    ap.add_argument("--out", default="reports/week2/window_index.csv")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"拒绝覆盖 {out}")
    rows = build(args.processed, args.split, args.subsets, args.budget)
    with out.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    pad = lambda rs: sum(r["boundary_rule"] != "real" for r in rs)
    print(f"{args.budget}: 训练窗 {len(tr)}(填充 {pad(tr)}),验证窗 {len(va)}(填充 {pad(va)});"
          f"sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
