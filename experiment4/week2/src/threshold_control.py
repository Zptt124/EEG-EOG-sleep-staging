#threshold_control.py —— 机制分析：D1/D2 的 N1 提升是否只是"把 N1 判定门槛调低"
#方法 1（匹配召回）：把 D0 的 p_N1 乘以系数 alpha 后重新取 argmax，选 alpha 使 D0 的 N1 召回最接近同种子 D1（或 D2），
#                   再比较两者的 N1 精确率 / N1-F1 / W-F1 / Macro-F1。若调整后的 D0 与 D1 相当，说明提升只是门槛移动
#方法 2（与门槛无关）：N1 的 average precision、N1 AUROC、真 W vs 真 N1 的区分 AUROC
#注意：alpha 是在验证集上匹配出来的，只用于解释机制，不作为任何方法的报告结果
#用法（Ubuntu）：python -u threshold_control.py --runs-root runs文件夹 --repr R1 --budget label25 --output 输出csv

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ssl_common import STAGES, load_official_evaluate, read_csv

PROB = ["p_W", "p_N1", "p_N2", "p_N3", "p_REM"]
ALPHAS = np.logspace(-1, 1.5, 801)  #alpha 从 0.1 到约 31.6


def hard_metrics(y, pred):
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(5), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(5), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(5), where=(prec + rec) > 0)
    return {"macro_f1": f1.mean(), "f1_N1": f1[1], "n1_precision": prec[1], "n1_recall": rec[1],
            "f1_W": f1[0], "f1_N2": f1[2], "f1_REM": f1[4], "W_to_N1": cm[0, 1] / cm[0].sum()}


def shifted_pred(p, alpha):
    q = p.copy()
    q[:, 1] *= alpha
    return q.argmax(1)


def ranking_metrics(y, p):
    #与门槛无关：N1 一对其余的 AP、AUROC；只在真 W/真 N1 上看 p_N1/(p_W+p_N1) 的区分能力
    is_n1 = (y == 1).astype(int)
    wn = (y == 0) | (y == 1)
    score_wn = p[wn, 1] / np.maximum(p[wn, 0] + p[wn, 1], 1e-12)
    #N1 vs N2 同理
    n2 = (y == 2) | (y == 1)
    score_n2 = p[n2, 1] / np.maximum(p[n2, 2] + p[n2, 1], 1e-12)
    return {"N1_AP": average_precision_score(is_n1, p[:, 1]), "N1_AUROC": roc_auc_score(is_n1, p[:, 1]),
            "W_vs_N1_AUROC": roc_auc_score((y[wn] == 1).astype(int), score_wn),
            "N2_vs_N1_AUROC": roc_auc_score((y[n2] == 1).astype(int), score_n2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--repr", default="R1")
    ap.add_argument("--budget", required=True)
    ap.add_argument("--version", default="v01")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        print(f"ERROR: {args.output} already exists")
        sys.exit(1)
    ev = load_official_evaluate()

    index = [r for r in read_csv(args.runs_root / "run_index.csv")
             if r["repr"] == args.repr and r["budget"] == args.budget and r["run_id"].endswith(args.version)]
    by = {(r["method"], int(r["seed"])): r["run_id"] for r in index}
    seeds = sorted(s for (m, s) in by if m == "D0" and ("D1", s) in by and ("D2", s) in by)
    data, ref = {}, None
    for (m, s), rid in by.items():
        if s not in seeds:
            continue
        df = ev.load_predictions(args.runs_root / rid / "predictions_val.csv").sort_values("sample_id")
        if ref is None:
            ref = (df["sample_id"].to_numpy(), df["y_true"].to_numpy())
        elif not (np.array_equal(df["sample_id"].to_numpy(), ref[0]) and np.array_equal(df["y_true"].to_numpy(), ref[1])):
            print(f"ERROR: {rid} has a different sample set or labels")
            sys.exit(1)
        data[(m, s)] = df[PROB].to_numpy(dtype=np.float64)
    y = ref[1].astype(int)
    print(f"Loaded seeds {seeds} for D0/D1/D2")

    rows = []
    for s in seeds:
        p0 = data[("D0", s)]
        #D0 在各个 alpha 下的 N1 召回（一次算好）
        curve = [hard_metrics(y, shifted_pred(p0, a)) for a in ALPHAS]
        rec_curve = np.array([c["n1_recall"] for c in curve])
        goals = {}
        for m in ("D0", "D1", "D2"):
            hm = hard_metrics(y, data[(m, s)].argmax(1))
            goals[m] = hm["n1_recall"]
            rows.append({"seed": s, "model": m, "alpha": 1.0, **hm, **ranking_metrics(y, data[(m, s)])})
        for target in ("D1", "D2"):
            goal = goals[target]
            k = int(np.argmin(np.abs(rec_curve - goal)))
            rows.append({"seed": s, "model": f"D0_shift_to_{target}_recall", "alpha": float(ALPHAS[k]), **curve[k],
                         **{key: "" for key in ("N1_AP", "N1_AUROC", "W_vs_N1_AUROC", "N2_vs_N1_AUROC")}})

    keys = ["macro_f1", "f1_N1", "n1_precision", "n1_recall", "f1_W", "f1_N2", "f1_REM", "W_to_N1",
            "N1_AP", "N1_AUROC", "W_vs_N1_AUROC", "N2_vs_N1_AUROC"]
    models = ["D0", "D1", "D2", "D0_shift_to_D1_recall", "D0_shift_to_D2_recall"]
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        mean = {"seed": "mean", "model": m, "alpha": round(float(np.mean([r["alpha"] for r in rs])), 4)}
        for k in keys:
            vals = [r[k] for r in rs if r[k] != ""]
            mean[k] = float(np.mean(vals)) if vals else ""
        rows.append(mean)
    for r in rows:
        for k in keys + ["alpha"]:
            if r[k] != "":
                r[k] = round(float(r[k]), 4)

    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "model", "alpha"] + keys)
        w.writeheader()
        w.writerows(rows)

    print("\n=== mean over seeds ===")
    print(f"{'model':24s} {'alpha':>6s} {'macro':>7s} {'N1F1':>6s} {'N1P':>6s} {'N1R':>6s} {'W_F1':>6s} {'W->N1':>6s} "
          f"{'N1_AP':>6s} {'N1AUC':>6s} {'W/N1':>6s} {'N2/N1':>6s}")
    for r in rows:
        if r["seed"] != "mean":
            continue
        fmt = lambda v: f"{v:6.4f}" if v != "" else f"{'-':>6s}"
        print(f"{r['model']:24s} {r['alpha']:6.3f} {r['macro_f1']:7.4f} {fmt(r['f1_N1'])} {fmt(r['n1_precision'])} "
              f"{fmt(r['n1_recall'])} {fmt(r['f1_W'])} {fmt(r['W_to_N1'])} {fmt(r['N1_AP'])} {fmt(r['N1_AUROC'])} "
              f"{fmt(r['W_vs_N1_AUROC'])} {fmt(r['N2_vs_N1_AUROC'])}")
    print("\n=== per seed: D1 vs D0 shifted to the same N1 recall ===")
    for s in seeds:
        d1 = next(r for r in rows if r["seed"] == s and r["model"] == "D1")
        d0s = next(r for r in rows if r["seed"] == s and r["model"] == "D0_shift_to_D1_recall")
        d0 = next(r for r in rows if r["seed"] == s and r["model"] == "D0")
        print(f"seed {s}: recall D1 {d1['n1_recall']:.3f} / D0shift {d0s['n1_recall']:.3f} (alpha {d0s['alpha']:.2f})  "
              f"N1F1 D1 {d1['f1_N1']:.4f} vs D0shift {d0s['f1_N1']:.4f}  macro D1 {d1['macro_f1']:.4f} vs D0shift {d0s['macro_f1']:.4f}  "
              f"N1_AP D1 {d1['N1_AP']:.4f} vs D0 {d0['N1_AP']:.4f}  W/N1 AUC D1 {d1['W_vs_N1_AUROC']:.4f} vs D0 {d0['W_vs_N1_AUROC']:.4f}")
    print(f"\nSaved: {args.output}")
    print("NOTE: alpha is matched on the validation set for mechanism analysis only; not a reportable method result.")


if __name__ == "__main__":
    main()
