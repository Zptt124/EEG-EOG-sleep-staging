#aggregate_pseudo_stats.py —— 汇总所有 D1/D2 run 的 pseudo_label_stats.csv，生成 pseudo_label_by_stage.csv
#每行：表征、预算、方法、伪标签类别；给出 5 个种子的均值（预测数、候选数、接收数、平均置信度、模态分歧）
#并附上校准阶段在有标签者内部（LOSO）估计的该类伪标签精确率 calib_precision_est（只来自有标签者，不用无标签者真标签）
#用法（Ubuntu）：python -u aggregate_pseudo_stats.py --runs-root runs文件夹 --calibration-root calibration文件夹 --output 输出csv

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ssl_common import STAGES, read_csv

NUM = ["n_predicted", "n_candidates", "n_accepted", "mean_conf_accepted", "frac_eeg_disagree",
       "frac_eog_disagree", "frac_both_agree_among_candidates"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--calibration-root", type=Path, required=True)
    ap.add_argument("--version", default="v01")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        print(f"ERROR: {args.output} already exists")
        sys.exit(1)

    index = [r for r in read_csv(args.runs_root / "run_index.csv")
             if r["method"] in ("D1", "D2") and r["run_id"].endswith(args.version)]
    groups = defaultdict(list)
    for r in index:
        stats = read_csv(args.runs_root / r["run_id"] / "pseudo_label_stats.csv")
        if [s["pseudo_class"] for s in stats] != STAGES:
            print(f"ERROR: unexpected classes in {r['run_id']}")
            sys.exit(1)
        for s in stats:
            groups[(r["repr"], r["budget"], r["sampling_seed"], r["method"], s["pseudo_class"])].append(s)

    #校准估计：同预算、同方法、锁定的 p 那一行的每类精确率
    calib = {}
    for d in args.calibration_root.glob("*/calibration_decision.json"):
        dec = json.loads(d.read_text(encoding="utf-8"))
        for row in read_csv(d.parent / "threshold_calibration.csv"):
            if row["method"] in ("D1", "D2") and abs(float(row["p"]) - float(dec["chosen_p"])) < 1e-9:
                for s in STAGES:
                    calib[(dec["budget"], dec["sampling_seed"], row["method"], s)] = row[f"prec_{s}"]

    out = []
    for (rep, budget, sseed, method, cls), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], STAGES.index(kv[0][4]))):
        row = {"repr": rep, "budget": budget, "sampling_seed": sseed, "method": method, "pseudo_class": cls,
               "n_seeds": len(rs)}
        for k in NUM:
            vals = [float(x[k]) for x in rs if x[k] != ""]
            row[f"{k}_mean"] = round(float(np.mean(vals)), 4) if vals else ""
        total_acc = sum(float(x["n_accepted"]) for x in rs)
        row["share_of_accepted"] = ""  #下面统一算
        row["calib_precision_est"] = calib.get((budget, sseed, method, cls), "")
        out.append(row)
    #每类占全部接收伪标签的比例
    tot = defaultdict(float)
    for r in out:
        tot[(r["repr"], r["budget"], r["method"])] += r["n_accepted_mean"]
    for r in out:
        r["share_of_accepted"] = round(r["n_accepted_mean"] / tot[(r["repr"], r["budget"], r["method"])], 4)

    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"{'repr':4s} {'budget':8s} {'meth':4s} {'class':5s} {'accepted':>9s} {'share':>6s} {'conf':>6s} {'calibP':>7s} {'EEGdis':>7s} {'EOGdis':>7s}")
    for r in out:
        print(f"{r['repr']:4s} {r['budget']:8s} {r['method']:4s} {r['pseudo_class']:5s} {r['n_accepted_mean']:9.0f} "
              f"{r['share_of_accepted']:6.3f} {r['mean_conf_accepted_mean']:6.3f} {str(r['calib_precision_est']):>7s} "
              f"{r['frac_eeg_disagree_mean']:7.3f} {r['frac_eog_disagree_mean']:7.3f}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
