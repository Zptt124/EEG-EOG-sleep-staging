"""Step W2-B10: 缺项表——按 W2 文档矩阵逐格对比"应有 run 数"与"实际找到的 run 数"。

"找到"的依据有三类:本机 runs/ 下的 W2_* 目录、指定 git 引用里名字含 W2_<成员>_ 的 run 目录,以及
run_registry.csv 里状态为 completed/reused 的行。复用的 run 按 W2 矩阵格计入(A 的两个 E1 run 占
label100 seed0)。D 按名字里的 D0/D1/D2 计数;计数为 0 就如实写 0,不推断"可能在对方本地"。
C 的 80 个 run 来自 reports/week2/external_C。

用法:
  python scripts/18_gap_table.py origin/main origin/feat/a-data-baseline origin/feat/b-evaluation \
      origin/feat/b-week2 origin/feat/c-eegdm origin/feat/d-fusion --out reports/week2/b_gap_table.csv
"""
import argparse
import csv
import re
import subprocess
from pathlib import Path

import pandas as pd

from common import BASE, RUNS

BUDGETS = ["label10", "label25", "label50", "label100"]
FIELDS = ["owner", "cell", "expected_runs", "found_runs", "status", "blocker_or_note"]


def names_in_refs(refs):
    names = set()
    for ref in refs:
        out = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref], cwd=BASE, capture_output=True, text=True).stdout
        for line in out.splitlines():
            names.update(re.findall(r"W2_[A-D]_[A-Za-z0-9_+\-]*?seed\d+_v\d+", line))
    return names


def names_in_registry(path):
    names = set()
    for r in pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records"):
        if r["status"] not in ("completed", "reused"):
            continue
        reused = re.fullmatch(r"E1_(EEG|EOG)_(label\d+)_seed(\d+)_v\d+", r["run_id"])
        names.add(f"W2_A_{reused[1]}_{reused[2]}_sseed2024_seed{reused[3]}_v01" if reused else r["run_id"])
    return names


def count(names, pattern):
    rx = re.compile(pattern)
    return len({n for n in names if rx.fullmatch(n)})


def status(found, expected):
    return "complete" if found >= expected else ("partial" if found else "missing")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("refs", nargs="*")
    ap.add_argument("--runs-root", default=str(RUNS))
    ap.add_argument("--registry", default=str(BASE / "reports/week2/run_registry.csv"))
    ap.add_argument("--external-c", default=str(BASE / "reports/week2/external_C/metrics_summary.csv"))
    ap.add_argument("--out", default="reports/week2/b_gap_table.csv")
    args = ap.parse_args(argv)

    names = names_in_refs(args.refs)
    names |= {p.name for p in Path(args.runs_root).glob("W2_*") if p.is_dir()}
    names |= names_in_registry(args.registry)
    rows = []

    def add(owner, cell, expected, found, note=""):
        rows.append(dict(owner=owner, cell=cell, expected_runs=expected, found_runs=found,
                         status=status(found, expected), blocker_or_note=note))

    # --- A:主矩阵最多 50 个 CNN run
    for mod, budgets in (("EEGEOG", BUDGETS), ("EEG", BUDGETS[:2] + BUDGETS[3:]), ("EOG", BUDGETS[:2] + BUDGETS[3:])):
        for b in budgets:
            add("A", f"CNN {mod} {b} seed0-4", 5, count(names, rf"W2_A_{mod}_{b}_sseed2024_seed[0-4]_v\d+"),
                "B 已用官方 evaluate v1 重算(reports/week2/external_A);复用的 2 个 E1 run 核验见 a_reuse_check_w1.md")
    # --- B:监督曲线的 EEGDM 头与上下文
    for b in BUDGETS:
        add("B", f"EEGDM EEG+EOG 线性头 {b} seed0-4", 5, count(names, rf"W2_B_EEGDMhead_{b}_sseed2024_seed[0-4]_v\d+"),
            "需要 EEG/EOG 冻结(池化)特征;25%/100% 旧结果须逐项核对后才可登记复用")
    for rep in ("EEGDM", "CNN"):
        for L in (1, 3):
            add("B", f"上下文 长度{L} {rep} label25 seed0-4", 5,
                count(names, rf"W2_B_CTX{L}_{rep}_label25_sseed2024_seed[0-4]_v\d+"),
                "需要 EEGDM 特征" if rep == "EEGDM" else "需要 A 的 25% 双通道 CNN 编码器特征")
    # --- C:S0/S1 对照(已并入 B 的统一评价)
    c = pd.read_csv(args.external_c) if Path(args.external_c).exists() else pd.DataFrame(columns=["run_id", "representation"])
    for rep, label, exp in (("C_S0_EEGDM_raw", "S0 原始冻结", 20), ("C_S1_EEGDM_adapted", "S1 适配(3 个适配 seed)", 60)):
        add("C", f"{label} 10%/25% × EEG、EEG+EOG × 下游 seed0-4", exp, int((c.representation == rep).sum()),
            "已由 B 用官方 evaluate v1 重算;C 重建的 EOG 与 D 的 EOG 在 25%/10% 线性头上结果一致,未做逐值比对")
    # --- D:半监督 D0/D1/D2
    for setting in ("D0", "D1", "D2"):
        add("D", f"{setting} 2 表征 × 2 预算 × seed0-4", 20, count(names, rf"W2_D_.*{setting}.*seed\d+_v\d+"),
            "B 已用官方 evaluate v1 重算并检查伪标签表(reports/week2/external_D)")

    out = Path(args.out)
    pd.DataFrame(rows, columns=FIELDS).to_csv(out, index=False)
    t = pd.DataFrame(rows)
    print(t.groupby("owner")[["expected_runs", "found_runs"]].sum().to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
