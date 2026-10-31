"""Step B2: 按受试者抽样少标注名单(25% / 100%),并统计类别数与实际人数。

规则(写入 README 共同协议):
  - 单位是训练集受试者(A 的 split_subjects.csv 中 split==train 的 54 人),同一人的所有夜晚一起有/无标签。
  - 先用 sampling_seed 对排序后的训练受试者做一次随机排列;前 k 人 = 该预算的名单。
    因此 25% 名单天然嵌套于更大预算(50%、100%),换预算不会换人。
  - 25% 的人数 k = round(0.25 * 54) = 14(54*0.25=13.5,Python 银行家舍入 -> 14),按人数而非片段数。
  - 抽样种子与训练种子分开记录。验证集/测试集不参与抽样。
  - 未入选的训练受试者记为"未标注训练对象":CNN 基线不使用其标签,也不用其信号(无半监督)。

输出(manifests/,版本化,不要原地修改;改动需升版本):
  label_subsets_<tag>.csv   每个训练受试者一行,含是否入选各预算
  reports/label_budget_<tag>.md  类别数与实际人数统计

用法:
  python scripts/06_make_label_subset.py --seed 2024 --tag v1
"""
import argparse
import sys

import numpy as np
import pandas as pd

from common import BASE, DOCS, PROCESSED, SPLIT, STAGE_NAMES, N_CLASSES

FRACTIONS = [0.25, 0.5, 1.0]


def load_train_subjects():
    df = pd.read_csv(SPLIT / "split_subjects.csv", dtype=str)
    df = df[df["split"] == "train"].sort_values("subject_id").reset_index(drop=True)
    return df


def count_labels(record_ids):
    """逐记录读取标签数组(只读 labels,不加载信号)。"""
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for rec in record_ids:
        lab = np.load(PROCESSED / f"{rec}.npz")["labels"]
        counts += np.bincount(lab, minlength=N_CLASSES)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="抽样种子(与训练种子分开)")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--processed", default=None, help="预处理 npz 目录(默认 data/processed)")
    args = ap.parse_args()

    global PROCESSED
    if args.processed:
        from pathlib import Path
        PROCESSED = Path(args.processed)

    train = load_train_subjects()
    subs = train["subject_id"].tolist()
    n = len(subs)
    order = np.random.default_rng(args.seed).permutation(n)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)             # sampling_rank: 1 = 最先被选入
    train["sampling_rank"] = rank
    train["n_records"] = train["record_ids"].str.split(";").map(len)
    train["n_epochs"] = [int(count_labels(r.split(";")).sum()) for r in train["record_ids"]]

    k_by_frac = {f: int(round(f * n)) for f in FRACTIONS}
    for f, k in k_by_frac.items():
        train[f"label{int(f * 100)}"] = (train["sampling_rank"] <= k).astype(int)
    assert (train["label25"] <= train["label50"]).all() and (train["label50"] <= train["label100"]).all()

    out = train.sort_values("sampling_rank")[
        ["subject_id", "record_ids", "n_records", "n_epochs", "sampling_rank",
         "label25", "label50", "label100"]].copy()
    out.insert(0, "sampling_seed", args.seed)
    out_path = SPLIT / f"label_subsets_{args.tag}.csv"
    if out_path.exists():
        sys.exit(f"{out_path} 已存在:名单已发布后不得原地覆盖,请换 --tag")
    out.to_csv(out_path, index=False)

    # ---- 统计 ----
    lines = [f"# 少标注名单统计 · {args.tag}", "",
             f"- 来源划分:`manifests/split_subjects.csv`(A 的 v1,训练集 {n} 人)",
             f"- 抽样种子:**{args.seed}**(训练种子另行记录在各 run 的 config.yaml)",
             f"- 名单文件:`manifests/label_subsets_{args.tag}.csv`;前 k 人嵌套:25% ⊂ 50% ⊂ 100%",
             "- 抽样单位为受试者(含其全部夜晚);验证/测试受试者不参与", "",
             "## 各预算的实际规模", "",
             "| 预算 | 有标签人数 | 有标签夜数 | 有标签片段数 | 占训练片段 | " +
             " | ".join(STAGE_NAMES) + " |",
             "|---|---|---|---|---|" + "---|" * N_CLASSES]
    total_epochs = int(train["n_epochs"].sum())
    stats = {}
    for f in FRACTIONS:
        col = f"label{int(f * 100)}"
        sel = train[train[col] == 1]
        recs = [r for rs in sel["record_ids"] for r in rs.split(";")]
        c = count_labels(recs)
        stats[f] = c
        lines.append(f"| {int(f * 100)}% | {len(sel)} | {len(recs)} | {int(c.sum()):,} | "
                     f"{c.sum() / total_epochs:.1%} | " + " | ".join(f"{int(v):,}" for v in c) + " |")
    lines += ["", "## 25% 档类别占比 vs 100% 档", "",
              "| 类别 | 25% 片段 | 25% 占比 | 100% 占比 |", "|---|---|---|---|"]
    for i, nm in enumerate(STAGE_NAMES):
        lines.append(f"| {nm} | {int(stats[0.25][i]):,} | {stats[0.25][i] / stats[0.25].sum():.1%} | "
                     f"{stats[1.0][i] / stats[1.0].sum():.1%} |")
    unl = train[train["label25"] == 0].sort_values("sampling_rank")
    lines += ["", f"## 25% 档下的未标注训练对象({len(unl)} 人)", "",
              "这些人的标签与信号在 25% 档 CNN 基线中**均不使用**(无半监督/伪标签)。", "",
              ", ".join(unl["subject_id"]), "",
              "## 25% 档有标签受试者", "",
              ", ".join(train[train["label25"] == 1].sort_values("sampling_rank")["subject_id"]), ""]
    rep = DOCS / f"label_budget_{args.tag}.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out_path}\nwrote {rep}")


if __name__ == "__main__":
    sys.exit(main())
