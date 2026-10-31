"""Step W2-B13: 按人分析——把每个 run 的 by_subject.csv 汇成 8 名验证者的对比表。

输出到 reports/week2:
  b_by_subject_summary.csv  每个条件 × 每名验证者:5 seed 的 Macro-F1、N1 F1 均值与种子间范围
  b_by_subject_delta.csv    每名验证者的配对差(comparison 减 reference,同 seed 相减再汇总):均值、范围、涨的 seed 数
  b_n1_error_direction.csv  25% 上下文四个条件的 N1 真值去向(5 seed 平均的段数与占比)
单人 N1 支持数很小,单人 N1 F1 波动大;8 人是开发集,这些表只用于找线索,不用于下结论。

用法:
  python scripts/21_by_subject_analysis.py --runs-root runs --a-by-subject reports/week2/external_A/by_subject.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from common import BASE, RUNS

CONDITIONS = {"EEGDM_linear_14": r"W2_B_EEGDMhead_label25_", "CNN_dual_14": r"W2_A_EEGEOG_label25_",
              "EEGDM_ctx1": r"W2_B_CTX1_EEGDM_", "EEGDM_ctx3": r"W2_B_CTX3_EEGDM_",
              "CNN_ctx1": r"W2_B_CTX1_CNN_", "CNN_ctx3": r"W2_B_CTX3_CNN_"}
PAIRS = [("EEGDM_ctx3", "EEGDM_ctx1"), ("CNN_ctx3", "CNN_ctx1"), ("EEGDM_ctx3", "CNN_ctx3"), ("EEGDM_linear_14", "CNN_dual_14")]


def load_all(runs_root, a_table):
    frames = [pd.read_csv(a_table)]
    for d in sorted(Path(runs_root).glob("W2_B_*")):
        frames.append(pd.read_csv(d / "by_subject.csv").assign(run_id=d.name))
    df = pd.concat(frames, ignore_index=True)
    df["seed"] = df["run_id"].str.extract(r"_seed(\d+)_v\d+$")[0].astype(int)
    out = []
    for cond, prefix in CONDITIONS.items():
        g = df[df["run_id"].str.startswith(prefix)].assign(condition=cond)
        if g["seed"].nunique() != 5 or len(g) != 40:
            raise ValueError(f"{cond}: 应为 5 seed × 8 人,实际 {len(g)} 行")
        out.append(g)
    return pd.concat(out, ignore_index=True)


def summary(df):
    g = df.groupby(["condition", "subject_id"], sort=False)
    return g.agg(n_epochs=("n_epochs", "first"), n1_support=("n1_support", "first"), macro_f1_mean=("macro_f1", "mean"),
                 macro_f1_min=("macro_f1", "min"), macro_f1_max=("macro_f1", "max"), n1_f1_mean=("n1_f1", "mean"),
                 n1_f1_min=("n1_f1", "min"), n1_f1_max=("n1_f1", "max")).reset_index().round(4)


def deltas(df):
    rows = []
    for comp, ref in PAIRS:
        a = df[df.condition == comp].set_index(["subject_id", "seed"])
        b = df[df.condition == ref].set_index(["subject_id", "seed"])
        d = (a[["macro_f1", "n1_f1"]] - b[["macro_f1", "n1_f1"]]).reset_index()
        for subject, g in d.groupby("subject_id"):
            rows.append(dict(comparison=comp, reference=ref, subject_id=subject,
                             delta_macro_f1_mean=g.macro_f1.mean(), delta_macro_f1_min=g.macro_f1.min(),
                             delta_macro_f1_max=g.macro_f1.max(), seeds_macro_up=int((g.macro_f1 > 0).sum()),
                             delta_n1_f1_mean=g.n1_f1.mean(), delta_n1_f1_min=g.n1_f1.min(),
                             delta_n1_f1_max=g.n1_f1.max(), seeds_n1_up=int((g.n1_f1 > 0).sum())))
    return pd.DataFrame(rows).round(4)


def n1_direction(runs_root):
    rows = []
    for cond, prefix in CONDITIONS.items():
        if "ctx" not in cond:
            continue
        t = pd.concat([pd.read_csv(p / "n1_errors.csv") for p in sorted(Path(runs_root).glob(prefix + "*"))]).mean()
        rows.append({"condition": cond, **{k: round(float(v), 1) for k, v in t.items()},
                     **{f"{k}_share": round(float(t[k] / t["n1_support"]), 4) for k in t.index if k != "n1_support"}})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs-root", default=str(RUNS))
    ap.add_argument("--a-by-subject", default=str(BASE / "reports/week2/external_A/by_subject.csv"))
    ap.add_argument("--out-dir", default=str(BASE / "reports/week2"))
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    df = load_all(args.runs_root, args.a_by_subject)
    summary(df).to_csv(out / "b_by_subject_summary.csv", index=False)
    deltas(df).to_csv(out / "b_by_subject_delta.csv", index=False)
    n1_direction(args.runs_root).to_csv(out / "b_n1_error_direction.csv", index=False)
    print(f"{df.condition.nunique()} 个条件、{df.subject_id.nunique()} 名验证者 -> {out}")


if __name__ == "__main__":
    main()
