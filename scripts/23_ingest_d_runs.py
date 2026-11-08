"""Step W2-B15: 把 D 的 60 个半监督 run 并入全组统一口径,并核对伪标签没有泄漏。

每个 predictions_val.csv 重新过官方 evaluate.py v1 + 验证参考清单,指标以重算为准,与 D 的 metrics.json 逐项比对,
不一致写入 mismatch_vs_reported.csv。40 份 pseudo_label_trace.csv(Release d-ssl-w2-v01-20261101)逐份检查:
没有真标签列、池内只有本预算无标签者且每段一次、D2 接收项三位教师一致、接收数与 pseudo_label_stats.csv 相同、
拒绝项权重为 0,任何一项不符都报错。

输出到 reports/week2/external_D,配对 delta 为 D1-D0、D2-D0、D2-D1(同表征、同预算、同 seed),按受试者 bootstrap。
sensitivity_D0_label10_seed0.csv:R1 10% 的 D0 seed0 早停结果不稳定(B、C 与 D 只差第 17 轮 7e-5 的验证分),
这里分别给出原配对、去掉 seed0、把 D0 seed0 换成 B 同配置结果三种均值。

用法:
  python scripts/23_ingest_d_runs.py --d-runs experiment4/week2/runs --trace-root features/d_w2_v01/extracted/runs \
      --roles experiment4/week2/manifests/v2/labeled_unlabeled_subjects.csv --out-dir reports/week2/external_D
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

from common import BASE, RUNS
import w2_common as w

ev = w.ev
spec = importlib.util.spec_from_file_location("agg", Path(__file__).with_name("14_w2_aggregate.py"))
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)

TOL = 5e-6
NAME_RX = re.compile(r"W2_D_(D[012])_(R[12])_(label\d+)_sseed2024_seed(\d+)_v\d+")
LABELED = {"label10": 5, "label25": 14}
UNLABELED = {"label10": 49, "label25": 40}
FORBIDDEN = {"y", "y_true", "label", "labels", "true_label", "stage"}


def subject_of(sample_id):
    return sample_id[:5]       # SC4001_0961 -> SC400


def check_trace(trace, stats, roles, method, budget, seed):
    """trace、stats 为 DataFrame,roles 为 subject->角色。返回 (池大小, 接收数),不符即抛错。"""
    tag = f"{method} {budget} seed{seed}"
    if FORBIDDEN & {c.lower() for c in trace.columns}:
        raise ValueError(f"{tag}: 伪标签表含真标签列")
    if trace["sample_id"].duplicated().any():
        raise ValueError(f"{tag}: sample_id 重复")
    subjects = set(trace["sample_id"].map(subject_of))
    expected = {s for s, r in roles.items() if r == "unlabeled"}
    if subjects != expected:
        raise ValueError(f"{tag}: 伪标签池的人与无标签名单不同,多 {sorted(subjects - expected)[:3]},少 {sorted(expected - subjects)[:3]}")
    if {method} != set(trace["method"]) or {budget} != set(trace["budget"]) or {seed} != set(trace["seed"]):
        raise ValueError(f"{tag}: 表内方法/预算/seed 与 run 不符")
    accepted = trace[trace["accepted"] == 1]
    if not (accepted["weight"] > 0).all() or not (trace.loc[trace["accepted"] == 0, "weight"] == 0).all():
        raise ValueError(f"{tag}: 接收与权重不一致")
    if method == "D2" and not ((accepted["eeg_prediction"] == accepted["pseudo_label"])
                               & (accepted["eog_prediction"] == accepted["pseudo_label"])).all():
        raise ValueError(f"{tag}: D2 接收项不是三位教师一致")
    if len(accepted) != int(stats["n_accepted"].sum()):
        raise ValueError(f"{tag}: 接收数 {len(accepted)} 与 pseudo_label_stats.csv 的 {int(stats['n_accepted'].sum())} 不同")
    return len(trace), len(accepted)


def ingest(run_dir, commit):
    method, rep, budget, seed = NAME_RX.fullmatch(run_dir.name).groups()
    f = run_dir / "predictions_val.csv"
    m = ev.evaluate(f, reference=w.VAL_REFERENCE)
    df = ev.load_predictions(f)
    g = m["per_class_f1"]
    unlabeled = 0 if method == "D0" else UNLABELED[budget]
    cfg = dict(run_id=run_dir.name, representation=f"D_{rep}_{method}", modalities="EEG+EOG", context_length=1,
               nominal_budget=budget, labeled_subjects=LABELED[budget], unlabeled_signal_subjects=unlabeled, downstream_seed=int(seed))
    history = pd.read_csv(run_dir / "history.csv")
    row = {k: cfg[k] for k in ("run_id", "representation", "modalities", "context_length", "nominal_budget", "labeled_subjects",
                               "unlabeled_signal_subjects", "downstream_seed")}
    row.update(adaptation_version="NA", adaptation_seed="NA", teacher_id="NA" if method == "D0" else "dual_D0_same_seed",   # seed 不进 ID,否则 seed_summary 会按 seed 拆开
              
               best_epoch=int(history.set_index("epoch")["val_macro_f1"].idxmax()), macro_f1=f"{m['macro_f1']:.6f}", W_f1=f"{g['W']:.6f}",
               N1_precision=f"{m['n1_precision']:.6f}", N1_recall=f"{m['n1_recall']:.6f}", N1_f1=f"{m['n1_f1']:.6f}", N2_f1=f"{g['N2']:.6f}",
               N3_f1=f"{g['N3']:.6f}", REM_f1=f"{g['REM']:.6f}", sample_count=m["n_epochs"], subject_count=m["n_subjects"],
               metrics_path=str(f.parent.relative_to(BASE) / "metrics.json"), predictions_path=str(f.relative_to(BASE)),
               predictions_sha256=w.sha256_file(f), by_subject_path="by_subject.csv", n1_errors_path="NA",
               evaluator_commit=commit, sample_set_sha256=m["sample_set_sha256"])
    reported = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    bad = [(run_dir.name, k, m[k], reported[k]) for k in ("macro_f1", "n1_precision", "n1_recall", "n1_f1", "accuracy")
           if abs(m[k] - reported[k]) > TOL]
    return (cfg, m, df, f), row, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--d-runs", required=True)
    ap.add_argument("--trace-root", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--b-runs", default=str(RUNS))
    ap.add_argument("--out-dir", default=str(BASE / "reports/week2/external_D"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    commit = agg.evaluator_commit()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    role_table = pd.read_csv(args.roles)
    runs, rows, subj_rows, mismatches, traces = {}, [], [], [], []
    for d in sorted(p for p in Path(args.d_runs).resolve().iterdir() if NAME_RX.fullmatch(p.name)):
        run, row, bad = ingest(d, commit)
        runs[d.name] = run
        rows.append(row)
        mismatches += bad
        subj_rows += [{"run_id": d.name, **t} for t in ev.per_subject_table(run[2]).to_dict("records")]
        method, rep, budget, seed = NAME_RX.fullmatch(d.name).groups()
        if method != "D0":
            trace = pd.read_csv(Path(args.trace_root) / d.name / "pseudo_label_trace.csv")
            stats = pd.read_csv(d / "pseudo_label_stats.csv")
            roles = dict(zip(role_table["subject_id"], role_table[f"role_{budget}"]))
            pool, accepted = check_trace(trace, stats, roles, method, budget, int(seed))
            traces.append(dict(run_id=d.name, pool_epochs=pool, accepted=accepted, true_label_columns=0, status="pass"))
    if len(rows) != 60 or len(traces) != 40:
        raise ValueError(f"应为 60 个 run、40 份伪标签表,实际 {len(rows)}、{len(traces)}")

    pd.DataFrame(rows, columns=agg.SUMMARY_FIELDS).to_csv(out / "metrics_summary.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(out / "by_subject.csv", index=False)
    agg.seed_summary(rows).to_csv(out / "seed_summary.csv", index=False)
    pd.DataFrame(traces).to_csv(out / "pseudo_trace_check.csv", index=False)
    pd.DataFrame(mismatches, columns=["run_id", "metric", "recomputed_by_B", "reported_by_owner"]).to_csv(
        out / "mismatch_vs_reported.csv", index=False)

    pairs = []
    for name in runs:
        method, rep, budget, seed = NAME_RX.fullmatch(name).groups()
        refs = {"D1": ["D0"], "D2": ["D0", "D1"]}.get(method, [])
        for r in refs:
            p = agg.paired_row(runs[name.replace(f"_{method}_", f"_{r}_")], runs[name], out, args.n_boot)
            p["by_subject_path"] = "by_subject.csv"
            p["limitations"] = agg.LIMITS + ";验证集同时用于早停;半监督额外使用无标签信号,与监督点不混称"
            pairs.append(p)
    pairs = pd.DataFrame(pairs, columns=agg.PAIR_FIELDS)
    pairs.to_csv(out / "paired_comparison.csv", index=False)

    b_ref = agg.load_run(Path(args.b_runs) / "W2_B_EEGDMhead_label10_sseed2024_seed0_v01")
    sens = []
    for method in ("D1", "D2"):
        sub = pairs[pairs.comparison_run_id.str.startswith(f"W2_D_{method}_R1_label10") & pairs.reference_run_id.str.contains("_D0_")]
        seed0 = runs[f"W2_D_{method}_R1_label10_sseed2024_seed0_v01"][1]
        for metric, col, key in (("macro_f1", "delta_macro_f1", "macro_f1"), ("n1_f1", "delta_N1_f1", "n1_f1")):
            d = sub[col].astype(float)
            others = sub[~sub.comparison_run_id.str.contains("_seed0_")][col].astype(float)
            swapped = (others.sum() + seed0[key] - b_ref[1][key]) / 5
            sens.append(dict(comparison=f"{method}-D0", metric=metric, original_5seed=round(d.mean(), 6),
                             without_seed0=round(others.mean(), 6), d0_seed0_from_B=round(swapped, 6)))
    pd.DataFrame(sens).to_csv(out / "sensitivity_D0_label10_seed0.csv", index=False)
    print(f"{len(rows)} 个 run、{len(traces)} 份伪标签表通过、{len(pairs)} 组配对;与 D 登记值不一致 {len(mismatches)} 项 -> {out}")


if __name__ == "__main__":
    main()
