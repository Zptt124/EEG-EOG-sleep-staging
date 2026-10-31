"""Step B4: 汇总 25% 与 100% 标签的对比与不确定性 -> reports/b_summary.md + b_summary.csv。

只读验证集预测(测试集本周封存)。所有指标都用官方 src/evaluation/evaluate.py 计算。
不确定性来源分开呈现,不把它们混成一个数字:
  (a) 验证受试者抽样:按受试者 bootstrap 的 95% 区间(验证集仅 8 人,区间会很宽);
  (b) 标签名单抽样:同一训练种子下换抽样种子(2024/2025/2026)的 25% 重复;
  (c) 训练随机性:本周未重复(各 run 只有 seed 0),因此 (b) 中包含 (c) 的混合,无法拆开。

用法: python scripts/08_summarize_fewlabel.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))
import evaluate as ev  # noqa: E402
from common import BASE, DOCS, RUNS  # noqa: E402

PKG = BASE / "交付包" / "实验①交付包_20261018" / "3_验证集预测_给D"
SEEDS = [2024, 2025, 2026]
PRIMARY = 2024


def full_path(ch):
    return PKG / f"predictions_val_{ch}.csv"


def few_path(ch, s):
    return RUNS / f"E2_{ch}_label25_sseed{s}_seed0_v01" / "predictions_val.csv"


def fmt_ci(d):
    return f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"


def main():
    L = ["# B:少标注(25% vs 100%)汇总与不确定性", "",
         "- 评估:官方 `src/evaluation/evaluate.py` " + ev.EVALUATE_VERSION + ";仅验证集(8 人 / 21,186 段),测试集未使用",
         f"- 主结果:抽样种子 {PRIMARY}、训练种子 0;100% 标签引用 A 的 `E1_*_label100_seed0_v01`(CUDA),25% 在 B 的机器(MPS)上训练",
         "- 区间:按**受试者**重采样的 bootstrap(1000 次,95% 百分位),不是按片段", ""]
    rows = []
    paired = {}
    for ch in ["EEG", "EOG"]:
        full = ev.load_predictions(full_path(ch))
        preds = {s: ev.load_predictions(few_path(ch, s)) for s in SEEDS if few_path(ch, s).exists()}
        if PRIMARY not in preds:
            sys.exit(f"缺少 {few_path(ch, PRIMARY)}:先跑 07_train_cnn_fewlabel.py")
        entries = [("100%", "A:E1_%s_label100_seed0_v01" % ch, full)] + \
                  [(f"25% (抽样种子 {s})", f"B:E2_{ch}_label25_sseed{s}_seed0_v01", d) for s, d in preds.items()]
        for label, run, df in entries:
            m = ev.compute_metrics(df["y_true"], df["y_pred"])
            b = ev.bootstrap_subjects(df, 1000, 0)
            rows.append({"modality": ch, "budget": label, "run_id": run, "macro_f1": m["macro_f1"],
                         "macro_f1_lo": b["macro_f1"]["lo"], "macro_f1_hi": b["macro_f1"]["hi"],
                         "n1_f1": m["n1_f1"], "n1_f1_lo": b["n1_f1"]["lo"], "n1_f1_hi": b["n1_f1"]["hi"],
                         "accuracy": m["accuracy"], **{f"f1_{k}": v for k, v in m["per_class_f1"].items()}})
        paired[ch] = {s: ev.paired_bootstrap_delta(d, full, 1000, 0) for s, d in preds.items()}
        paired[(ch, "df")] = (full, preds)
    tab = pd.DataFrame(rows)
    tab.to_csv(DOCS / "b_summary.csv", index=False)

    L += ["## 1. 主对比(验证集)", "",
          "| 模态 | 标签预算 | Macro-F1 [95% 区间] | N1 F1 [95% 区间] | Acc | run_id |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['modality']} | {r['budget']} | {r['macro_f1']:.4f} [{r['macro_f1_lo']:.4f}, {r['macro_f1_hi']:.4f}] "
                 f"| {r['n1_f1']:.4f} [{r['n1_f1_lo']:.4f}, {r['n1_f1_hi']:.4f}] | {r['accuracy']:.4f} | `{r['run_id']}` |")

    L += ["", "## 2. 25% 相对 100% 的差(同一批验证片段上的配对 bootstrap,Δ = 25% − 100%)", "",
          "| 模态 | 抽样种子 | ΔMacro-F1 [95%] | ΔN1 F1 [95%] | ΔMacro-F1>0 的 bootstrap 占比 |", "|---|---|---|---|---|"]
    for ch in ["EEG", "EOG"]:
        for s, d in paired[ch].items():
            L.append(f"| {ch} | {s} | {d['macro_f1']['point']:+.4f} [{d['macro_f1']['lo']:+.4f}, {d['macro_f1']['hi']:+.4f}] "
                     f"| {d['n1_f1']['point']:+.4f} [{d['n1_f1']['lo']:+.4f}, {d['n1_f1']['hi']:+.4f}] "
                     f"| {d['macro_f1']['frac_boot_positive']:.2f} |")

    L += ["", "## 3. 换一批 25% 受试者的波动(标签名单抽样 + 训练随机性混合,各 1 次训练)", "",
          "| 模态 | 抽样种子数 | 各 Macro-F1 | 均值 | 标准差(n−1) | 极差 | 100% 基线 |", "|---|---|---|---|---|---|---|"]
    for ch in ["EEG", "EOG"]:
        v = tab[(tab.modality == ch) & tab.budget.str.startswith("25%")]["macro_f1"].to_numpy()
        base = tab[(tab.modality == ch) & (tab.budget == "100%")]["macro_f1"].iloc[0]
        sd = f"{v.std(ddof=1):.4f}" if len(v) > 1 else "n/a"
        L.append(f"| {ch} | {len(v)} | {', '.join(f'{x:.4f}' for x in v)} | {v.mean():.4f} | {sd} | "
                 f"{v.max() - v.min():.4f} | {base:.4f} |")
    L += ["", "> n=3 的标准差本身很不稳,只作量级参考;不能当作置信区间。", ""]

    L += ["## 4. 每类 F1(主结果,抽样种子 %d)" % PRIMARY, "",
          "| 模态 | 预算 | " + " | ".join(ev.STAGE_NAMES) + " |", "|---|---|" + "---|" * 5]
    for r in rows:
        if r["budget"] in ("100%", f"25% (抽样种子 {PRIMARY})"):
            L.append(f"| {r['modality']} | {r['budget']} | " + " | ".join(f"{r['f1_' + n]:.4f}" for n in ev.STAGE_NAMES) + " |")

    L += ["", "## 5. 验证受试者逐人 Macro-F1(主结果)", ""]
    for ch in ["EEG", "EOG"]:
        full, preds = paired[(ch, "df")]
        t100 = ev.per_subject_table(full).set_index("subject_id")
        t25 = ev.per_subject_table(preds[PRIMARY]).set_index("subject_id")
        j = pd.DataFrame({"n_epochs": t100["n_epochs"], "100% Macro-F1": t100["macro_f1"],
                          "25% Macro-F1": t25["macro_f1"], "Δ": t25["macro_f1"] - t100["macro_f1"],
                          "N1 support": t100["n1_support"]})
        L += [f"**{ch}**", "", "| 受试者 | 片段数 | N1 片段数 | 100% | 25% | Δ |", "|---|---|---|---|---|---|"]
        for sid, r in j.iterrows():
            L.append(f"| {sid} | {int(r['n_epochs'])} | {int(r['N1 support'])} | {r['100% Macro-F1']:.4f} | "
                     f"{r['25% Macro-F1']:.4f} | {r['Δ']:+.4f} |")
        L.append("")

    L += ["## 6. 限制(解读前必读)", "",
          "- 验证集只有 8 个受试者,所有区间都建立在这 8 人之上;换验证人群,结论可能变。",
          "- 25% 档每个抽样种子只训练 1 次(训练种子 0),训练随机性与名单抽样的影响无法分开。",
          "- 100% 基线在 CUDA 上训练,25% 在 Apple MPS 上训练,浮点实现不同,不保证逐位可比;早停/选模规则相同(验证集 Macro-F1)。",
          "- 25% 档归一化统计量只用有标签受试者;100% 档用全部训练受试者。这是预算差异的一部分,不单独剥离。",
          "- 没有半监督:未标注训练受试者完全未使用。结果只回答'标签变少时纯监督小 CNN 掉多少',不回答别的方法。",
          "- 验证集上选 epoch 与比较方案属同一份数据;测试集本周封存,以上数字不是最终泛化估计。",
          "- N1 在验证集中占 12.4%(2,623 段),跨受试者分布不均,N1 指标的区间最宽。", ""]
    out = DOCS / "b_summary.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
