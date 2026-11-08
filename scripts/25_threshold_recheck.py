"""Step W2-B16: 独立复算 D 的结论「D1 的 N1 提升主要来自判 N1 倾向的移动」。

只读 60 份 predictions_val.csv 重算,不读 D 的 threshold_control.csv 作为输入,算完才与它比对。
做法与 D 相同:D0 的 p_N1 乘 alpha 后重新 argmax,取 N1 召回最接近同 seed D1 的 alpha(同一网格,取最先命中)。
把 D1-D0 拆成两段:
  threshold_part = D0调整后 - D0   只移动 N1 门槛能得到的部分
  residual       = D1 - D0调整后   相同召回下 D1 仍多出的部分
另算与门槛无关的 N1 AP / N1 AUROC / W vs N1 / N2 vs N1 AUROC。
按受试者 bootstrap 时 8 人的重采样在 5 个 seed 间共用,alpha 固定不重选,区间不含 alpha 的不确定性。
alpha 在验证集上匹配,只解释机制,不是可报告的方法结果。

输出 reports/week2/external_D/threshold_recheck_seed.csv、threshold_recheck_summary.csv、threshold_recheck_vs_D.csv。

用法:
  python scripts/25_threshold_recheck.py --d-runs experiment4/week2/runs --d-analysis experiment4/week2/analysis
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import BASE, RUNS
import w2_common as w

ev = w.ev
PROB = ["p_W", "p_N1", "p_N2", "p_N3", "p_REM"]
ALPHAS = np.logspace(-1, 1.5, 801)
SEEDS = range(5)
CELLS = [("R1", "label10"), ("R1", "label25"), ("R2", "label10"), ("R2", "label25")]
COMPARE_TOL = 1.01e-4                     # D 的表四舍五入到 4 位


def cm_of(y, pred):
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def scores(cm):
    """cm 形状 (..., 5, 5),行真实、列预测。返回 macro_f1、n1_f1、n1_precision、n1_recall,保留前置维度。"""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    pred, true = cm.sum(-2), cm.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * tp / (pred + true))
        return dict(macro_f1=f1.mean(-1), n1_f1=f1[..., 1], n1_precision=np.nan_to_num(tp[..., 1] / pred[..., 1]),
                    n1_recall=np.nan_to_num(tp[..., 1] / true[..., 1]))


def ranking(y, p):
    wn, n2 = (y == 0) | (y == 1), (y == 2) | (y == 1)
    return dict(n1_ap=average_precision_score(y == 1, p[:, 1]), n1_auroc=roc_auc_score(y == 1, p[:, 1]),
                w_vs_n1_auroc=roc_auc_score(y[wn] == 1, p[wn, 1] / np.maximum(p[wn, 0] + p[wn, 1], 1e-12)),
                n2_vs_n1_auroc=roc_auc_score(y[n2] == 1, p[n2, 1] / np.maximum(p[n2, 2] + p[n2, 1], 1e-12)))


def match_alpha(y, p0, goal):
    rec = np.array([scores(cm_of(y, (p0 * np.array([1, a, 1, 1, 1])).argmax(1)))["n1_recall"] for a in ALPHAS])
    return float(ALPHAS[int(np.argmin(np.abs(rec - goal)))])


def load(path):
    df = ev.load_predictions(path).sort_values("sample_id").reset_index(drop=True)
    return df, df[PROB].to_numpy(dtype=np.float64)


def per_subject_cm(df, pred):
    y = df["y_true"].to_numpy()
    return np.stack([cm_of(y[g], pred[g]) for g in df.groupby("subject_id").indices.values()])


def analyse_seed(d0_path, d1_path):
    df0, p0 = load(d0_path)
    df1, p1 = load(d1_path)
    if not (df0["sample_id"].equals(df1["sample_id"]) and df0["y_true"].equals(df1["y_true"])):
        raise ValueError(f"样本或标签不同: {d0_path} vs {d1_path}")
    y = df0["y_true"].to_numpy().astype(int)
    goal = scores(cm_of(y, p1.argmax(1)))["n1_recall"]
    alpha = match_alpha(y, p0, goal)
    shifted = (p0 * np.array([1, alpha, 1, 1, 1])).argmax(1)
    preds = dict(d0=p0.argmax(1), d1=p1.argmax(1), shift=shifted)
    sm = {k: scores(cm_of(y, v)) for k, v in preds.items()}
    r0, r1 = ranking(y, p0), ranking(y, p1)
    row = dict(alpha=alpha)
    for name, s in sm.items():
        row.update({f"{name}_{k}": float(v) for k, v in s.items()})
    row.update({f"delta_{k}": r1[k] - r0[k] for k in r0})
    return row, {k: per_subject_cm(df0, v) for k, v in preds.items()}


def decompose(rows, metric):
    f = lambda k: np.mean([r[f"{k}_{metric}"] for r in rows])
    return f("d1") - f("d0"), f("shift") - f("d0"), f("d1") - f("shift")


def boot_ci(cms, metric, n_boot, seed=0):
    """cms:每个 seed 的 per_subject_cm。返回 threshold_part 与 residual 的 95% 区间。"""
    rng = np.random.default_rng(seed)
    n = cms[0]["d0"].shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    weights = np.stack([np.bincount(i, minlength=n) for i in idx])
    mean_of = lambda k: np.mean([scores(np.einsum("bs,sij->bij", weights, c[k]))[metric] for c in cms], axis=0)
    d0, d1, sh = mean_of("d0"), mean_of("d1"), mean_of("shift")
    q = lambda v: np.percentile(v, [2.5, 97.5])
    return q(sh - d0), q(d1 - sh)


def summarise(label, rows, cms, n_boot):
    out = dict(cell=label, n_seeds=len(rows), mean_alpha=np.mean([r["alpha"] for r in rows]),
               max_recall_gap=max(abs(r["shift_n1_recall"] - r["d1_n1_recall"]) for r in rows))
    for metric, tag in (("n1_f1", "n1"), ("macro_f1", "macro")):
        total, part, resid = decompose(rows, metric)
        ci_part, ci_resid = boot_ci(cms, metric, n_boot)
        out.update({f"{tag}_d1_minus_d0": total, f"{tag}_threshold_part": part, f"{tag}_residual": resid,
                    f"{tag}_share": part / total if abs(total) > 1e-9 else float("nan"),
                    f"{tag}_threshold_ci_low": ci_part[0], f"{tag}_threshold_ci_high": ci_part[1],
                    f"{tag}_residual_ci_low": ci_resid[0], f"{tag}_residual_ci_high": ci_resid[1],
                    f"{tag}_seeds_d1_above_shift": sum(r[f"d1_{metric}"] > r[f"shift_{metric}"] for r in rows)})
    out["precision_d1_minus_shift"] = np.mean([r["d1_n1_precision"] - r["shift_n1_precision"] for r in rows])
    for k in ("n1_ap", "n1_auroc", "w_vs_n1_auroc", "n2_vs_n1_auroc"):
        out[f"delta_{k}"] = np.mean([r[f"delta_{k}"] for r in rows])
        out[f"delta_{k}_seeds_up"] = sum(r[f"delta_{k}"] > 0 for r in rows)
    return out


def compare_with_d(rows, analysis_root):
    bad, checked = [], 0
    pairs = [("d0", "D0"), ("d1", "D1"), ("shift", "D0_shift_to_D1_recall")]
    cols = [("macro_f1", "macro_f1"), ("n1_f1", "f1_N1"), ("n1_precision", "n1_precision"), ("n1_recall", "n1_recall")]
    for r in rows:
        t = pd.read_csv(Path(analysis_root) / f"{r['repr']}_{r['budget']}_v01" / "threshold_control.csv")
        t = t[t["seed"].astype(str) == str(r["seed"])].set_index("model")
        for ours, theirs in pairs:
            for k, c in cols:
                checked += 1
                gap = abs(r[f"{ours}_{k}"] - t.loc[theirs, c])
                if gap > COMPARE_TOL:
                    bad.append(dict(repr=r["repr"], budget=r["budget"], seed=r["seed"], model=theirs, metric=k,
                                    recomputed_by_B=r[f"{ours}_{k}"], reported_by_D=t.loc[theirs, c]))
        if abs(r["alpha"] - t.loc["D0_shift_to_D1_recall", "alpha"]) > COMPARE_TOL * max(1, r["alpha"]):
            bad.append(dict(repr=r["repr"], budget=r["budget"], seed=r["seed"], model="D0_shift_to_D1_recall",
                            metric="alpha", recomputed_by_B=r["alpha"], reported_by_D=t.loc["D0_shift_to_D1_recall", "alpha"]))
        checked += 1
    return bad, checked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--d-runs", required=True)
    ap.add_argument("--d-analysis", required=True)
    ap.add_argument("--b-runs", default=str(RUNS))
    ap.add_argument("--out-dir", default=str(BASE / "reports/week2/external_D"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    root, out = Path(args.d_runs), Path(args.out_dir)
    run = lambda m, rep, b, s: root / f"W2_D_{m}_{rep}_{b}_sseed2024_seed{s}_v01" / "predictions_val.csv"

    rows, cms, summary = [], {}, []
    for rep, budget in CELLS:
        cell, cell_cms = [], []
        for s in SEEDS:
            r, cm = analyse_seed(run("D0", rep, budget, s), run("D1", rep, budget, s))
            cell.append({"repr": rep, "budget": budget, "seed": s, **r})
            cell_cms.append(cm)
        rows += cell
        cms[(rep, budget)] = (cell, cell_cms)
        summary.append(summarise(f"{rep}_{budget}", cell, cell_cms, args.n_boot))
    summary.append(summarise("all_4_cells", rows, [c for v in cms.values() for c in v[1]], args.n_boot))

    # 敏感性:R1 10% 的 D0 seed0 换成 B 同配置结果(与 sensitivity_D0_label10_seed0.csv 同一问题)
    cell, cell_cms = cms[("R1", "label10")]
    r, cm = analyse_seed(Path(args.b_runs) / "W2_B_EEGDMhead_label10_sseed2024_seed0_v01" / "predictions_val.csv",
                         run("D1", "R1", "label10", 0))
    summary.append(summarise("R1_label10_D0seed0_from_B", [{**cell[0], **r}] + cell[1:], [cm] + cell_cms[1:], args.n_boot))
    swapped = [{**cell[0], **r}] + [x for k in CELLS for x in cms[k][0]][1:]
    swapped_cms = [cm] + [x for k in CELLS for x in cms[k][1]][1:]
    summary.append(summarise("all_4_cells_D0seed0_from_B", swapped, swapped_cms, args.n_boot))

    bad, checked = compare_with_d(rows, args.d_analysis)
    pd.DataFrame(rows).round(6).to_csv(out / "threshold_recheck_seed.csv", index=False)
    pd.DataFrame(summary).round(6).to_csv(out / "threshold_recheck_summary.csv", index=False)
    pd.DataFrame(bad, columns=["repr", "budget", "seed", "model", "metric", "recomputed_by_B", "reported_by_D"]).to_csv(
        out / "threshold_recheck_vs_D.csv", index=False)
    print(f"{len(rows)} 组 seed;与 D 的 threshold_control.csv 比对 {checked} 项,不一致 {len(bad)} 项")
    show = ["cell", "n1_d1_minus_d0", "n1_threshold_part", "n1_residual", "n1_share", "n1_residual_ci_low", "n1_residual_ci_high",
            "n1_seeds_d1_above_shift", "macro_d1_minus_d0", "macro_threshold_part", "macro_residual", "macro_residual_ci_low",
            "macro_residual_ci_high", "macro_seeds_d1_above_shift", "precision_d1_minus_shift"]
    print(pd.DataFrame(summary)[show].round(4).T.to_string())
    print(pd.DataFrame(summary)[["cell"] + [c for c in summary[0] if c.startswith("delta_")]].round(4).T.to_string())


if __name__ == "__main__":
    main()
