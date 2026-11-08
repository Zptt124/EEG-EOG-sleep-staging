"""Step W3-B4: 第三周统一评估(全组共用)。

登记表 reports/week3/b_eval_registry_w3.csv 列:run_id,group,arm,budget,seed,predictions
  - runs/W3_B_* 自动登记(arm/budget/seed 从 run_id 解析);A/C/D 的预测用 --extra 追加同列 CSV。
  - predictions 为 11 列 predictions_val.csv(sample_id,subject_id,record_id,epoch_index,y_true,y_pred,p_W..p_REM),
    相对仓库根目录的路径或绝对路径。
每份预测先用官方 evaluate.py v1 对照 manifests/val_reference_v1.csv 核对 sample_id 覆盖与 y_true。

输出 reports/week3:
  b_w3_metrics.csv          每个 run:Macro-F1、各阶段 F1、N1 P/R/F1、N1 AP/AUROC、N1 真值去向与 N1 预测来源
  b_w3_summary.csv          group×arm×budget:seed 数与均值/最小/最大
  b_w3_paired.csv           --pairs 指定的配对(comparison:reference,同 budget 同 seed 才配对):
                            seed 合并后按受试者 bootstrap 的差值与 95% 区间(Macro-F1、N1 F1、N1 AP),及逐 seed 差值
  b_w3_paired_by_subject.csv  每名验证者的配对差(各 seed 平均)
开发集 8 人反复使用;区间只反映受试者抽样,不含 seed 与选模的不确定性。

用法:
  python scripts/29_b_w3_evaluate.py --pairs mean_t1:std_t1 std_t10:std_t1 std_t25:std_t1
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import BASE, RUNS
import w2_common as w

ev = w.ev
OUT = BASE / "reports" / "week3"
COLS = ["run_id", "group", "arm", "budget", "seed", "predictions"]
N_BOOT = 1000
KEYS = ["macro_f1", "n1_f1", "n1_ap"]


def registry(runs_root, extra):
    rows = []
    for d in sorted(Path(runs_root).glob("W3_B_*")):
        m = re.fullmatch(r"W3_B_(.+)_(label\d+)_sseed2024_seed(\d+)_v\d+", d.name)
        if m and (d / "predictions_val.csv").exists():
            rows.append(dict(run_id=d.name, group="B", arm=m[1], budget=m[2], seed=int(m[3]),
                             predictions=str((d / "predictions_val.csv").relative_to(BASE))))
    reg = pd.DataFrame(rows, columns=COLS)
    for path in extra:
        e = pd.read_csv(path)
        if list(e.columns) != COLS:
            raise ValueError(f"{path} 列应为 {COLS}")
        reg = pd.concat([reg, e], ignore_index=True)
    if reg["run_id"].duplicated().any():
        raise ValueError("run_id 重复")
    return reg


def load(path):
    p = Path(path)
    df = ev.load_predictions(p if p.is_absolute() else BASE / p)
    return df, ev.check_against_reference(df, pd.read_csv(w.VAL_REFERENCE))


def run_metrics(r, df, sha):
    m = ev.compute_metrics(df["y_true"], df["y_pred"])
    y1 = (df["y_true"] == 1).to_numpy()
    n1, p1 = df[y1], df[df["y_pred"] == 1]
    out = dict(r, sample_set_sha256=sha, n_epochs=len(df), macro_f1=m["macro_f1"], accuracy=m["accuracy"])
    out.update({f"f1_{s}": m["per_class_f1"][s] for s in ev.STAGE_NAMES})
    out.update(n1_precision=m["n1_precision"], n1_recall=m["n1_recall"], n1_f1=m["n1_f1"],
               n1_ap=average_precision_score(y1, df["p_N1"]), n1_auroc=roc_auc_score(y1, df["p_N1"]),
               n1_true_to_W=(n1["y_pred"] == 0).mean(), n1_true_to_N2=(n1["y_pred"] == 2).mean(),
               n1_true_to_REM=(n1["y_pred"] == 4).mean(),
               n1_pred_from_W=(p1["y_true"] == 0).mean(), n1_pred_from_N2=(p1["y_true"] == 2).mean(),
               n1_pred_from_REM=(p1["y_true"] == 4).mean())
    return out


def pooled_stats(frames, subjects):
    """多个 seed 的预测在指定受试者(可重复)上合并后的 Macro-F1、N1 F1、N1 AP。"""
    cm = sum(frames["cms"][s] for s in subjects)
    m = ev.metrics_from_confusion(cm)
    y = np.concatenate([frames["y1"][s] for s in subjects])
    p = np.concatenate([frames["p1"][s] for s in subjects])
    return dict(macro_f1=m["macro_f1"], n1_f1=m["n1_f1"], n1_ap=average_precision_score(y, p))


def per_subject(dfs):
    """seed 列表 -> 每名受试者的合并混淆矩阵与 N1 标签/分数。"""
    out = dict(cms={}, y1={}, p1={})
    for s in sorted(dfs[0]["subject_id"].unique()):
        gs = [d[d["subject_id"] == s] for d in dfs]
        out["cms"][s] = sum(ev.confusion(g["y_true"], g["y_pred"]) for g in gs)
        out["y1"][s] = np.concatenate([(g["y_true"] == 1).to_numpy() for g in gs])
        out["p1"][s] = np.concatenate([g["p_N1"].to_numpy() for g in gs])
    return out


def paired(reg, preds, comp, ref):
    rows, subj_rows = [], []
    for budget in sorted(set(reg["budget"])):
        a = reg[(reg["arm"] == comp) & (reg["budget"] == budget)].set_index("seed")
        b = reg[(reg["arm"] == ref) & (reg["budget"] == budget)].set_index("seed")
        seeds = sorted(set(a.index) & set(b.index))
        if not seeds:
            continue
        da = [preds[a.loc[s, "run_id"]].sort_values("sample_id").reset_index(drop=True) for s in seeds]
        db = [preds[b.loc[s, "run_id"]].sort_values("sample_id").reset_index(drop=True) for s in seeds]
        for x, y in zip(da, db):
            if not (x["sample_id"].equals(y["sample_id"]) and x["y_true"].equals(y["y_true"])):
                raise ValueError(f"{comp} vs {ref} {budget}: sample_id 或 y_true 不一致")
        fa, fb = per_subject(da), per_subject(db)
        subs = sorted(fa["cms"])
        pa, pb = pooled_stats(fa, subs), pooled_stats(fb, subs)
        rng = np.random.default_rng(0)
        boot = {k: [] for k in KEYS}
        for draw in rng.integers(0, len(subs), size=(N_BOOT, len(subs))):
            chosen = [subs[i] for i in draw]
            sa, sb = pooled_stats(fa, chosen), pooled_stats(fb, chosen)
            for k in KEYS:
                boot[k].append(sa[k] - sb[k])
        row = dict(comparison=comp, reference=ref, budget=budget, seeds=" ".join(map(str, seeds)), n_subjects=len(subs))
        for k in KEYS:
            lo, hi = np.percentile(boot[k], [2.5, 97.5])
            row.update({f"delta_{k}": pa[k] - pb[k], f"delta_{k}_lo": lo, f"delta_{k}_hi": hi,
                        f"delta_{k}_frac_boot_pos": float(np.mean(np.array(boot[k]) > 0))})
        per_seed = [run_metrics({}, x, None)["macro_f1"] - run_metrics({}, y, None)["macro_f1"] for x, y in zip(da, db)]
        per_seed_n1 = [ev.compute_metrics(x["y_true"], x["y_pred"])["n1_f1"] - ev.compute_metrics(y["y_true"], y["y_pred"])["n1_f1"]
                       for x, y in zip(da, db)]
        row.update(per_seed_delta_macro_f1=" ".join(f"{v:+.4f}" for v in per_seed),
                   per_seed_delta_n1_f1=" ".join(f"{v:+.4f}" for v in per_seed_n1))
        rows.append(row)
        for s in subs:
            ma, mb = ev.metrics_from_confusion(fa["cms"][s]), ev.metrics_from_confusion(fb["cms"][s])
            subj_rows.append(dict(comparison=comp, reference=ref, budget=budget, subject_id=s,
                                  n1_support=int(fa["cms"][s][1].sum() // len(seeds)),
                                  delta_macro_f1=ma["macro_f1"] - mb["macro_f1"], delta_n1_f1=ma["n1_f1"] - mb["n1_f1"],
                                  delta_n1_recall=ma["n1_recall"] - mb["n1_recall"],
                                  delta_n1_precision=ma["n1_precision"] - mb["n1_precision"]))
    return rows, subj_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs-root", default=str(RUNS))
    ap.add_argument("--extra", nargs="*", default=[], help="A/C/D 的登记 CSV(列同 b_eval_registry_w3.csv)")
    ap.add_argument("--pairs", nargs="*", default=[], help="comparison:reference,按 arm 名")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    reg = registry(args.runs_root, args.extra)
    reg.to_csv(OUT / "b_eval_registry_w3.csv", index=False)
    preds, metrics = {}, []
    for r in reg.to_dict("records"):
        df, sha = load(r["predictions"])
        preds[r["run_id"]] = df
        metrics.append(run_metrics(r, df, sha))
    met = pd.DataFrame(metrics)
    met.to_csv(OUT / "b_w3_metrics.csv", index=False, float_format="%.6g")
    show = ["macro_f1", "n1_f1", "n1_precision", "n1_recall", "n1_ap", "f1_REM", "n1_true_to_W", "n1_true_to_N2"]
    summ = met.groupby(["group", "arm", "budget"])[show].agg(["mean", "min", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.insert(0, "n_seeds", met.groupby(["group", "arm", "budget"]).size())
    summ.reset_index().to_csv(OUT / "b_w3_summary.csv", index=False, float_format="%.6g")
    print(met.groupby(["group", "arm", "budget"])[show].mean().round(4).to_string())

    rows, subj = [], []
    for pair in args.pairs:
        comp, ref = pair.split(":")
        r, s = paired(reg, preds, comp, ref)
        rows += r
        subj += s
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "b_w3_paired.csv", index=False, float_format="%.6g")
        pd.DataFrame(subj).to_csv(OUT / "b_w3_paired_by_subject.csv", index=False, float_format="%.6g")
        cols = ["comparison", "reference", "budget", "seeds"] + [f"delta_{k}{x}" for k in KEYS for x in ("", "_lo", "_hi")]
        print(pd.DataFrame(rows)[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
