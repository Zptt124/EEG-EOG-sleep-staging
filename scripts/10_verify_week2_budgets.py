"""Step W2-B2: 独立核验 label_subsets_v2 / label_counts_v2(不依赖 build_week2_budgets.py 的输出)。

只读:split_subjects.csv、label_subsets_v1/v2.csv、label_counts_v2.csv、
54 名训练者的 NPZ 中的 labels 成员(验证/测试 NPZ 不打开,信号数组不读)。
任一项不符即退出码 1;全部通过时可选写出 Markdown 核验记录。

用法:
  python scripts/10_verify_week2_budgets.py --processed data/processed \
      [--report reports/week2/b_v2_verification.md]
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

from common import PROCESSED, SPLIT, STAGE_NAMES

BUDGETS = {10: 5, 25: 14, 50: 27, 100: 54}
CANDIDATE10 = ["SC456", "SC421", "SC474", "SC412", "SC403"]   # 文档第 6 页预设排序后的统计
CANDIDATE10_EPOCHS, CANDIDATE10_N1 = 11936, 1719


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--processed", default=str(PROCESSED))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    split = read(SPLIT / "split_subjects.csv")
    v1 = read(SPLIT / "label_subsets_v1.csv")
    v2 = read(SPLIT / "label_subsets_v2.csv")
    counts = {r["budget"]: r for r in read(SPLIT / "label_counts_v2.csv")}
    by_split = {k: {r["subject_id"] for r in split if r["split"] == k} for k in ("train", "val", "test")}
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    check("划分 54/8/16", [len(by_split[k]) for k in ("train", "val", "test")] == [54, 8, 16])
    check("v2 恰为 54 名训练者、无验证/测试者",
          {r["subject_id"] for r in v2} == by_split["train"] and len(v2) == 54)
    # v1 逐字保留:旧列、行序、旧标记完全相同
    legacy = [c for c in v1[0].keys()]
    check("v2 保留 v1 全部旧列、行序与取值",
          [{c: r[c] for c in legacy} for r in v2] == v1)
    check("v2 仅新增 label10 一列", [c for c in v2[0] if c not in legacy] == ["label10"])
    ranks = sorted(int(r["sampling_rank"]) for r in v2)
    check("sampling_rank 为 1..54 唯一", ranks == list(range(1, 55)))
    check("抽样种子均为 2024", {r["sampling_seed"] for r in v2} == {"2024"})

    flags = {b: {r["subject_id"] for r in v2 if r[f"label{b}"] == "1"} for b in BUDGETS}
    check("人数 5/14/27/54", [len(flags[b]) for b in BUDGETS] == list(BUDGETS.values()),
          str({b: len(flags[b]) for b in BUDGETS}))
    check("嵌套 10⊂25⊂50⊂100", flags[10] <= flags[25] <= flags[50] <= flags[100])
    check("与验证/测试互斥", all(not (flags[b] & (by_split["val"] | by_split["test"])) for b in BUDGETS))
    check("标记等于 rank 截断(未重洗牌/未按成绩挑人)",
          all((r[f"label{b}"] == "1") == (int(r["sampling_rank"]) <= n)
              for r in v2 for b, n in BUDGETS.items()))
    top5 = [r["subject_id"] for r in sorted(v2, key=lambda r: int(r["sampling_rank"]))[:5]]
    check("候选 10% 为 SC456/SC421/SC474/SC412/SC403", top5 == CANDIDATE10, str(top5))

    # 从 NPZ 重新计数(只读 labels)
    rec_counts = {}
    for r in v2:
        for rec in r["record_ids"].split(";"):
            with np.load(Path(args.processed) / f"{rec}.npz", allow_pickle=False) as d:
                labels = d["labels"]
            rec_counts[rec] = np.bincount(labels.astype(np.int64), minlength=5)
    rows = []
    for b in BUDGETS:
        recs = [rec for r in v2 if r[f"label{b}"] == "1" for rec in r["record_ids"].split(";")]
        c = sum(rec_counts[x] for x in recs)
        n = int(c.sum())
        total = sum(int(v.sum()) for v in rec_counts.values())
        rows.append((b, BUDGETS[b], len(recs), n, 100 * n / total, c))
        t = counts[f"label{b}"]
        check(f"label{b} 夜晚/片段/各类数与 label_counts_v2.csv 一致",
              int(t["labeled_records"]) == len(recs) and int(t["labeled_epochs"]) == n
              and [int(t[s]) for s in STAGE_NAMES] == [int(x) for x in c], f"{len(recs)} 晚, {n} 段")
        check(f"label{b} 无缺失类别", bool((c > 0).all()))
    c10 = rows[0][5]
    check(f"候选 10% = {CANDIDATE10_EPOCHS} 段、N1 = {CANDIDATE10_N1}",
          rows[0][3] == CANDIDATE10_EPOCHS and int(c10[1]) == CANDIDATE10_N1, f"{rows[0][3]}, N1={int(c10[1])}")
    check("label25 与第一周 v1 一致(14 人 / 27 晚 / 32310 段)",
          (rows[1][1], rows[1][2], rows[1][3]) == (14, 27, 32310), f"{rows[1][1:4]}")

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} 项通过")

    if args.report and not failed:
        lines = ["# label_subsets_v2 核验记录(B)", "",
                 "由 `scripts/10_verify_week2_budgets.py` 生成;只读 54 名训练者 NPZ 的 labels 成员,",
                 "未打开验证/测试 NPZ,未读信号,未参考任何验证成绩。", "",
                 "| 预算 | 有标签人数 | 夜晚数 | 片段数 | 占训练池片段 | " + " | ".join(STAGE_NAMES) + " |",
                 "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in STAGE_NAMES) + " |"]
        for b, n, nrec, nep, pct, c in rows:
            lines.append(f"| label{b} | {n} | {nrec} | {nep} | {pct:.2f}% | " + " | ".join(str(int(x)) for x in c) + " |")
        lines += ["", "候选 10% 人员(rank 1-5):" + "、".join(top5) + "。人数 5/54 = 9.26%,**不是严格 10% 的片段**,"
                  "表中占比以实际片段为准。", "", "## 检查项", ""]
        lines += [f"- [x] {name}" + (f"({detail})" if detail else "") for name, _, detail in checks]
        lines += ["", "## 文件 SHA256", ""]
        for p in ("split_subjects.csv", "label_subsets_v1.csv", "label_subsets_v2.csv", "label_counts_v2.csv"):
            lines.append(f"- `manifests/{p}`:`{sha256(SPLIT / p)}`")
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
