"""Step W3-B3: N1 分数调整对照(系数只在训练侧选,固定后用于开发集)。

预先固定:14 人预算(label25)、下游 seed 0;std 与 mean 各一组(--arms)。
  1. 读 27_b_w3_linear_probe.py 写出的 oof_train.csv(训练名单内按受试者留出的预测)。
  2. p_N1 乘 alpha 后重新 argmax;alpha 网格与 25_threshold_recheck.py 相同(logspace(-1,1.5,801),含 1)。
     取留出 N1 F1 最高的 alpha,并列取 |log alpha| 最小者。开发集不参与选择。
  3. 同一 alpha 用于同 run 的 predictions_val.csv,报告调整前后的 N1 P/R/F1、Macro-F1、N1→W/N2 去向,
     以及按受试者配对 bootstrap 的 调整后 − 调整前。
  4. 原始 p_N1 的 AP/AUROC 不受 alpha 影响,用来比较 std 与 mean 的排序。
注意:留出预测来自折内模型(少一折受试者),开发集预测来自全名单重训模型,两者校准可能略有差别。

输出 reports/week3/b_n1_adjust_summary.csv、b_n1_adjust_by_subject.csv。

用法:
  python scripts/28_b_n1_score_adjust.py --arms std_t1 mean_t1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import BASE, RUNS
import w2_common as w

ev = w.ev
PROB = w.PROB_COLS
ALPHAS = np.logspace(-1, 1.5, 801)
OUT = BASE / "reports" / "week3"


def adjusted(df, alpha):
    p = df[PROB].to_numpy().copy()
    p[:, 1] *= alpha
    return df.assign(y_pred=p.argmax(1))


def pick_alpha(oof):
    y, p = oof["y_true"].to_numpy(), oof[PROB].to_numpy()
    scores = []
    for a in ALPHAS:
        q = p.copy()
        q[:, 1] *= a
        scores.append(ev.compute_metrics(y, q.argmax(1))["n1_f1"])
    scores = np.array(scores)
    cand = np.flatnonzero(scores == scores.max())
    i = cand[np.argmin(np.abs(np.log(ALPHAS[cand])))]
    return float(ALPHAS[i]), float(scores[i]), float(scores[np.argmin(np.abs(np.log(ALPHAS)))])


def describe(df):
    m = ev.compute_metrics(df["y_true"], df["y_pred"])
    n1 = df[df["y_true"] == 1]
    return dict(macro_f1=m["macro_f1"], n1_precision=m["n1_precision"], n1_recall=m["n1_recall"], n1_f1=m["n1_f1"],
                n1_to_W=float((n1["y_pred"] == 0).mean()), n1_to_N2=float((n1["y_pred"] == 2).mean()),
                pred_n1=int((df["y_pred"] == 1).sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--budget", default="label25")
    ap.add_argument("--seed", type=int, default=0, help="预先固定的 seed")
    ap.add_argument("--runs-root", default=str(RUNS))
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, subj = [], []
    for arm in args.arms:
        run_id = f"W3_B_{arm}_{args.budget}_sseed2024_seed{args.seed}_v01"
        d = Path(args.runs_root) / run_id
        oof = pd.read_csv(d / "oof_train.csv")
        val = ev.load_predictions(d / "predictions_val.csv")
        ev.check_against_reference(val, pd.read_csv(w.VAL_REFERENCE))
        alpha, oof_f1, oof_f1_raw = pick_alpha(oof)
        after = adjusted(val, alpha)
        before_m, after_m = describe(val), describe(after)
        y1 = (val["y_true"] == 1).to_numpy()
        boot = ev.paired_bootstrap_delta(after, val, 1000, 0)
        row = dict(arm=arm, run_id=run_id, alpha=alpha, oof_n1_f1_raw=oof_f1_raw, oof_n1_f1_adjusted=oof_f1,
                   dev_n1_ap=average_precision_score(y1, val["p_N1"]), dev_n1_auroc=roc_auc_score(y1, val["p_N1"]))
        for k in before_m:
            row[f"before_{k}"], row[f"after_{k}"] = before_m[k], after_m[k]
        for k in ("n1_f1", "macro_f1"):
            row[f"delta_{k}"], row[f"delta_{k}_lo"], row[f"delta_{k}_hi"] = (boot[k]["point"], boot[k]["lo"], boot[k]["hi"])
        rows.append(row)
        for s, g in val.groupby("subject_id"):
            b, a = describe(g), describe(after[after["subject_id"] == s])
            subj.append(dict(arm=arm, subject_id=s, n1_support=int((g["y_true"] == 1).sum()),
                             before_n1_f1=b["n1_f1"], after_n1_f1=a["n1_f1"],
                             before_n1_recall=b["n1_recall"], after_n1_recall=a["n1_recall"],
                             before_n1_precision=b["n1_precision"], after_n1_precision=a["n1_precision"]))
        print(f"{arm}: alpha={alpha:.3f}  OOF N1 F1 {oof_f1_raw:.3f}->{oof_f1:.3f}  dev N1 P/R/F1 "
              f"{before_m['n1_precision']:.3f}/{before_m['n1_recall']:.3f}/{before_m['n1_f1']:.3f} -> "
              f"{after_m['n1_precision']:.3f}/{after_m['n1_recall']:.3f}/{after_m['n1_f1']:.3f}  "
              f"macro-F1 {before_m['macro_f1']:.4f}->{after_m['macro_f1']:.4f}  AP {row['dev_n1_ap']:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "b_n1_adjust_summary.csv", index=False, float_format="%.6g")
    pd.DataFrame(subj).to_csv(OUT / "b_n1_adjust_by_subject.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
