"""Step W2-B7: 监督预算曲线(Macro-F1 与 N1 F1,并附 N1 precision/recall 表)。

读 seed_summary.csv(由 14_w2_aggregate.py 生成)与 --extra-summary(A/D 的同结构表)。横轴是"有标签受试者人数"(5/14/27/54),不是片段比例;
误差条是训练种子间标准差,**不含**受试者不确定性(8 人开发集)。缺失预算(没有跑)留空,不插值;
单一 seed 的点不画误差条并在图例标注。半监督(D)带 unlabeled_signal_subjects>0 的点用空心标记另列,
不与只用有标签者信号的监督点混称。D 的点(表征名以 D_ 开头)单独画一张 <out>_semi.png,不进监督曲线。

用法:
  python scripts/15_plot_budget_curves.py --summary reports/week2/seed_summary.csv \
      --extra-summary reports/week2/external_A/seed_summary.csv --out reports/week2/budget_curves
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--summary", default="reports/week2/seed_summary.csv")
    ap.add_argument("--extra-summary", nargs="*", default=[], help="其他成员的 seed_summary.csv")
    ap.add_argument("--out", default="reports/week2/budget_curves")
    ap.add_argument("--context-length", type=int, default=1, help="监督曲线用长度 1(无上下文)的结果")
    args = ap.parse_args(argv)
    df = pd.concat([pd.read_csv(p) for p in [args.summary, *args.extra_summary]], ignore_index=True)
    df = df[(df["head"] == "linear") & (df.context_length == args.context_length)    # 预算曲线只用线性头
            & (df.modalities == "EEG+EOG")]                                          # 单模态与概率平均不进曲线
    if df.empty:
        raise SystemExit("没有符合条件的行")
    semi = df[df.representation.str.startswith("D_")]
    sup = df[~df.representation.str.startswith("D_")]
    out = Path(args.out)
    draw(sup, "Supervised label-budget curves (8-subject dev set; bars = std over training seeds)", out.with_suffix(".png"))
    if not semi.empty:
        draw(semi, "Semi-supervised D0/D1/D2 (5+49 and 14+40; hollow = uses unlabeled signals; bars = seed std)",
             out.with_name(out.name + "_semi.png"))
    cols = ["representation", "labeled_subjects", "unlabeled_signal_subjects", "n_seeds",
            "N1_precision_mean", "N1_recall_mean", "N1_f1_mean"]
    df[cols].sort_values(["representation", "labeled_subjects"]).to_csv(out.with_name(out.name + "_n1_table.csv"), index=False)
    print(f"wrote {out.with_suffix('.png')}")


def draw(df, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True)
    for ax, (col, ylabel) in zip(axes, [("macro_f1", "Macro-F1"), ("N1_f1", "N1 F1")]):
        for rep, g in df.groupby("representation"):
            g = g.sort_values("labeled_subjects")
            unlabeled = sorted(g.unlabeled_signal_subjects.unique())
            semi = max(unlabeled) > 0
            label = rep + (" (+" + "/".join(str(int(u)) for u in unlabeled[::-1]) + " unlabeled)" if semi else "")
            multi = g.n_seeds.max() > 1
            ax.errorbar(g.labeled_subjects, g[f"{col}_mean"], yerr=g.get(f"{col}_std") if multi else None,
                        marker="o", mfc="white" if semi else None, capsize=3, label=label + ("" if multi else " (1 seed)"))
        ax.set_xlabel("labeled subjects (count of people, not epoch fraction)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df.labeled_subjects.unique()))
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
