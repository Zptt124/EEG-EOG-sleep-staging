#compare_ssl.py —— 第二周 实验④ 汇总：D0/D1/D2 多种子对比、N1 去向、按人分析、按受试者 bootstrap
#只读各 run 的 predictions_val.csv 和 run_index.csv，不训练任何东西
#输出（--output 文件夹）：
#  ssl_comparison.csv        每个方法的均值±标准差；D1-D0、D2-D1、D2-D0 逐种子配对差与为正的种子数
#  n1_direction.csv          每个 run 及方法均值：真 N1 被判成各阶段的比例；判成 N1 的段真实来自哪里；真 W 判成 N1 的比例
#  per_subject.csv           每个验证者、每个方法（5 种子平均）的指标，以及两级差值
#  bootstrap_paired.csv      按受试者 bootstrap：把 5 个种子的逐人混淆矩阵相加后，对受试者有放回重采样，算配对差的 95% 区间
#用法（Ubuntu）：python -u compare_ssl.py --runs-root runs文件夹 --repr R1 --budget label25 --output 输出文件夹

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from ssl_common import STAGES, load_official_evaluate, read_csv

METHODS = ["D0", "D1", "D2"]
PAIRS = [("D1", "D0"), ("D2", "D1"), ("D2", "D0")]
N_BOOT = 1000


def per_class(cm):
    #由混淆矩阵（行真实、列预测）算每类 precision/recall/F1
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(5), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(5), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(5), where=(prec + rec) > 0)
    return prec, rec, f1


def metric_dict(cm):
    prec, rec, f1 = per_class(cm)
    d = {"macro_f1": f1.mean(), "accuracy": np.trace(cm) / cm.sum()}
    for j, s in enumerate(STAGES):
        d[f"f1_{s}"] = f1[j]
    d["n1_precision"], d["n1_recall"] = prec[1], rec[1]
    d["W_to_N1"] = cm[0, 1] / max(cm[0].sum(), 1)  #真 W 被判成 N1 的比例
    return d


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--repr", default="R1")
    ap.add_argument("--budget", required=True)
    ap.add_argument("--version", default="v01")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if (args.output / "ssl_comparison.csv").exists():
        print(f"ERROR: {args.output} already has results, choose a new folder")
        sys.exit(1)
    args.output.mkdir(parents=True, exist_ok=True)
    ev = load_official_evaluate()

    #---------- 找到全部 run，读预测（官方 load_predictions 会做格式检查） ----------
    index = [r for r in read_csv(args.runs_root / "run_index.csv")
             if r["repr"] == args.repr and r["budget"] == args.budget and r["run_id"].endswith(args.version)]
    sseeds = {r["sampling_seed"] for r in index}
    if len(sseeds) != 1:
        print(f"ERROR: expected one sampling_seed, found {sseeds}")
        sys.exit(1)
    seeds = {m: sorted(int(r["seed"]) for r in index if r["method"] == m) for m in METHODS}
    common = sorted(set(seeds["D0"]) & set(seeds["D1"]) & set(seeds["D2"]))
    print(f"Seeds per method: {seeds}; paired comparisons use seeds {common}")
    if len(common) < 2:
        print("ERROR: need at least 2 seeds shared by D0, D1, D2")
        sys.exit(1)

    cms = {}           #(method, seed) -> 总混淆矩阵
    subj_cms = {}      #(method, seed) -> (受试者列表, [S,5,5])
    ref_ids = None
    for r in index:
        m, s = r["method"], int(r["seed"])
        if s not in common:
            continue
        df = ev.load_predictions(args.runs_root / r["run_id"] / "predictions_val.csv")
        df = df.sort_values("sample_id").reset_index(drop=True)
        ids = df["sample_id"].to_numpy()
        if ref_ids is None:
            ref_ids, ref_y = ids, df["y_true"].to_numpy()
        elif not (np.array_equal(ids, ref_ids) and np.array_equal(df["y_true"].to_numpy(), ref_y)):
            print(f"ERROR: {r['run_id']} has a different sample set or labels")
            sys.exit(1)
        cms[(m, s)] = ev.confusion(df["y_true"], df["y_pred"])
        subj_cms[(m, s)] = ev.per_subject_confusions(df)
        #核对：重新算的 Macro-F1 必须等于 run_index 里登记的值
        if abs(metric_dict(cms[(m, s)])["macro_f1"] - float(r["macro_f1"])) > 1e-4:
            print(f"ERROR: recomputed macro-F1 differs from run_index for {r['run_id']}")
            sys.exit(1)
    print(f"Loaded {len(cms)} prediction files, all with identical sample_id/y_true")

    keys = ["macro_f1", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM", "n1_precision", "n1_recall", "W_to_N1", "accuracy"]
    met = {k: metric_dict(v) for k, v in cms.items()}
    reg = {(r["method"], int(r["seed"])): r for r in index}

    #---------- 1. 方法均值与逐种子配对差 ----------
    comp = []
    for m in METHODS:
        row = {"row_type": "method", "name": m, "n_seeds": len(common)}
        for k in keys:
            v = [met[(m, s)][k] for s in common]
            row[f"{k}_mean"] = round(float(np.mean(v)), 4)
            row[f"{k}_std"] = round(float(np.std(v, ddof=1)), 4)
        for k in ("n_pseudo_labels_used", "n_updates", "best_epoch"):
            row[f"{k}_mean"] = round(float(np.mean([float(reg[(m, s)][k]) for s in common])), 1)
        comp.append(row)
    for a, b in PAIRS:
        row = {"row_type": "paired_diff", "name": f"{a}-{b}", "n_seeds": len(common)}
        for k in keys:
            d = [met[(a, s)][k] - met[(b, s)][k] for s in common]
            row[f"{k}_mean"] = round(float(np.mean(d)), 4)
            row[f"{k}_std"] = round(float(np.std(d, ddof=1)), 4)
            row[f"{k}_n_pos"] = int(sum(x > 0 for x in d))
            row[f"{k}_per_seed"] = " ".join(f"{x:+.4f}" for x in d)
        comp.append(row)
    fields = list(dict.fromkeys(k for r in comp for k in r))
    comp = [{k: r.get(k, "") for k in fields} for r in comp]
    write_csv(args.output / "ssl_comparison.csv", comp)

    #---------- 2. N1 去向 ----------
    direc = []
    for m in METHODS:
        for s in common + ["mean"]:
            cm = cms[(m, s)] if s != "mean" else sum(cms[(m, x)] for x in common)
            row = {"method": m, "seed": s}
            for j, st in enumerate(STAGES):
                row[f"trueN1_to_{st}"] = round(cm[1, j] / cm[1].sum(), 4)
            for j, st in enumerate(STAGES):
                row[f"predN1_from_{st}"] = round(cm[j, 1] / max(cm[:, 1].sum(), 1), 4)
            row["n_pred_N1"] = int(cm[:, 1].sum()) if s != "mean" else round(cm[:, 1].sum() / len(common), 1)
            row["trueW_to_N1"] = round(cm[0, 1] / cm[0].sum(), 4)
            row["trueREM_to_N1"] = round(cm[4, 1] / cm[4].sum(), 4)
            direc.append(row)
    write_csv(args.output / "n1_direction.csv", direc)

    #---------- 3. 按人（5 种子混淆矩阵相加） ----------
    subs = subj_cms[("D0", common[0])][0]
    summed = {m: sum(subj_cms[(m, s)][1] for s in common) for m in METHODS}  #[S,5,5]
    per_subj = []
    for i, sub in enumerate(subs):
        row = {"subject_id": sub, "n_epochs": int(summed["D0"][i].sum() / len(common)),
               "n_true_N1": int(summed["D0"][i][1].sum() / len(common))}
        md = {m: metric_dict(summed[m][i]) for m in METHODS}
        for m in METHODS:
            for k in ("macro_f1", "f1_N1", "n1_recall", "n1_precision", "f1_W", "f1_REM"):
                row[f"{m}_{k}"] = round(float(md[m][k]), 4)
        for a, b in PAIRS[:2]:
            for k in ("macro_f1", "f1_N1", "f1_W", "f1_REM"):
                row[f"{a}-{b}_{k}"] = round(float(md[a][k] - md[b][k]), 4)
        per_subj.append(row)
    write_csv(args.output / "per_subject.csv", per_subj)

    #---------- 4. 按受试者 bootstrap 配对差（同一套重采样） ----------
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(subs), size=(N_BOOT, len(subs)))
    bkeys = ["macro_f1", "f1_N1", "f1_W", "f1_REM", "n1_precision", "n1_recall"]
    boot = []
    for a, b in PAIRS:
        pa, pb = metric_dict(summed[a].sum(0)), metric_dict(summed[b].sum(0))
        deltas = {k: [] for k in bkeys}
        for d in draws:
            ma, mb = metric_dict(summed[a][d].sum(0)), metric_dict(summed[b][d].sum(0))
            for k in bkeys:
                deltas[k].append(ma[k] - mb[k])
        for k in bkeys:
            lo, hi = np.percentile(deltas[k], [2.5, 97.5])
            n_sub_pos = sum(metric_dict(summed[a][i])[k] > metric_dict(summed[b][i])[k] for i in range(len(subs)))
            boot.append({"pair": f"{a}-{b}", "metric": k, "point": round(float(pa[k] - pb[k]), 4),
                         "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                         "excludes_zero": bool(lo > 0 or hi < 0),
                         "frac_boot_positive": round(float(np.mean(np.array(deltas[k]) > 0)), 3),
                         "n_subjects_positive": f"{n_sub_pos}/{len(subs)}",
                         "note": f"confusions summed over seeds {common}; {N_BOOT} subject resamples, rng seed 0"})
    write_csv(args.output / "bootstrap_paired.csv", boot)

    #---------- 屏幕摘要 ----------
    print("\n=== method mean +- std over seeds ===")
    for r in comp[:3]:
        print(f"{r['name']}: macro {r['macro_f1_mean']:.4f}+-{r['macro_f1_std']:.4f}  N1 {r['f1_N1_mean']:.4f}+-{r['f1_N1_std']:.4f}  "
              f"W {r['f1_W_mean']:.4f}  REM {r['f1_REM_mean']:.4f}  W->N1 {r['W_to_N1_mean']:.4f}  pseudo {r['n_pseudo_labels_used_mean']}")
    print("\n=== N1 direction (seed-summed) ===")
    for r in direc:
        if r["seed"] == "mean":
            print(f"{r['method']}: true N1 -> " + " ".join(f"{s}:{r['trueN1_to_' + s]:.3f}" for s in STAGES)
                  + f" | pred N1 from " + " ".join(f"{s}:{r['predN1_from_' + s]:.3f}" for s in STAGES)
                  + f" | W->N1 {r['trueW_to_N1']:.3f}")
    print("\n=== per-subject (seed-summed) D1-D0 / D2-D1 ===")
    for r in per_subj:
        print(f"{r['subject_id']} N1 n={r['n_true_N1']:4d}  D1-D0 macro {r['D1-D0_macro_f1']:+.3f} N1 {r['D1-D0_f1_N1']:+.3f} W {r['D1-D0_f1_W']:+.3f}"
              f"   D2-D1 macro {r['D2-D1_macro_f1']:+.3f} N1 {r['D2-D1_f1_N1']:+.3f} REM {r['D2-D1_f1_REM']:+.3f}")
    print("\n=== subject bootstrap of paired differences (95% CI) ===")
    for r in boot:
        print(f"{r['pair']:6s} {r['metric']:12s} {r['point']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
              f"excl0={r['excludes_zero']}  subjects+ {r['n_subjects_positive']}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
