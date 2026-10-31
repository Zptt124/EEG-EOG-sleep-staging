"""Step W2-B12: 把 A 的 run 并入全组统一口径,并做 EEGDM 线性头 vs CNN 双通道的同样本配对。

输入是解压后的 w2_a_runs_v01(Release a-cnn-w2-v01-20261031)。每个 predictions_val.csv 都重新过官方
evaluate.py v1 + 验证参考清单,指标以重算为准,并与 A 登记的 a_supervised_comparison.csv 逐项比对,
不一致写入 mismatch_vs_reported.csv,不静默覆盖。输出到 reports/week2/external_A,不改 A 的表。

复用的两个第一周 run(E1_EEG/EOG_label100_seed0)的预测不在 Release 内,这里不重算,核验见
reports/week2/a_reuse_check_w1.md。

配对表 delta 为 EEGDM 减 CNN(同预算、同下游 seed、同 8 人)。两者只有 seed 编号相同,没有共享权重;
CNN 只见过本预算有标签者,EEGDM 主干由外部数据预训练,所以这是表征比较,不是纯架构比较。

用法:
  python scripts/20_ingest_a_runs.py --a-runs features/a_w2_v01/extracted --b-runs runs --out-dir reports/week2/external_A
"""
import argparse
import importlib.util
import re
from pathlib import Path

import pandas as pd

from common import BASE, RUNS
import w2_common as w

ev = w.ev
spec = importlib.util.spec_from_file_location("agg", Path(__file__).with_name("14_w2_aggregate.py"))
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)

TOL = 5e-6      # A 的表保留全精度,这里只容忍浮点误差
BUDGET_SUBJECTS = {"label10": 5, "label25": 14, "label50": 27, "label100": 54}
MODALITY = {"EEGEOG": "EEG+EOG", "EEG": "EEG", "EOG": "EOG", "PROBAVG": "prob_avg"}
REPRESENTATION = {"EEGEOG": "A_CNN_dual_channel", "EEG": "A_CNN_EEG", "EOG": "A_CNN_EOG", "PROBAVG": "A_CNN_prob_avg"}
NAME_RX = re.compile(r"W2_A_(EEGEOG|EEG|EOG|PROBAVG)_(label\d+)_sseed2024_seed(\d+)_v\d+")


def ingest(run_dir, commit, compared):
    """返回 (cfg, 指标, 预测表, 路径, 汇总行, 与 A 登记值的不一致项)。"""
    kind, budget, seed = NAME_RX.fullmatch(run_dir.name).groups()
    f = run_dir / "predictions_val.csv"
    m = ev.evaluate(f, reference=w.VAL_REFERENCE)
    df = ev.load_predictions(f)
    history = run_dir / "history.csv"       # 概率平均目录没有训练过程
    best_epoch = int(pd.read_csv(history).set_index("epoch")["val_macro_f1"].idxmax()) if history.exists() else "NA"
    cfg = dict(run_id=run_dir.name, representation=REPRESENTATION[kind], modalities=MODALITY[kind], context_length=1,
               nominal_budget=budget, labeled_subjects=BUDGET_SUBJECTS[budget], unlabeled_signal_subjects=0,
               downstream_seed=int(seed))
    g = m["per_class_f1"]
    rel = f"w2_a_runs_v01.zip#{run_dir.name}/"
    row = {"run_id": run_dir.name, **{k: cfg[k] for k in ("representation", "modalities", "context_length", "nominal_budget",
                                                          "labeled_subjects", "unlabeled_signal_subjects", "downstream_seed")},
           "adaptation_version": "NA", "adaptation_seed": "NA", "teacher_id": "NA", "best_epoch": best_epoch,
           "macro_f1": f"{m['macro_f1']:.6f}", "W_f1": f"{g['W']:.6f}", "N1_precision": f"{m['n1_precision']:.6f}",
           "N1_recall": f"{m['n1_recall']:.6f}", "N1_f1": f"{m['n1_f1']:.6f}", "N2_f1": f"{g['N2']:.6f}",
           "N3_f1": f"{g['N3']:.6f}", "REM_f1": f"{g['REM']:.6f}", "sample_count": m["n_epochs"],
           "subject_count": m["n_subjects"], "metrics_path": rel + "metrics.json", "predictions_path": rel + "predictions_val.csv",
           "predictions_sha256": w.sha256_file(f), "by_subject_path": "by_subject.csv", "n1_errors_path": "NA",
           "evaluator_commit": commit, "sample_set_sha256": m["sample_set_sha256"]}
    reported = compared[run_dir.name]
    ours = dict(macro_f1=m["macro_f1"], n1_precision=m["n1_precision"], n1_recall=m["n1_recall"], n1_f1=m["n1_f1"], rem_f1=g["REM"])
    bad = [(run_dir.name, k, v, float(reported[k])) for k, v in ours.items() if abs(v - float(reported[k])) > TOL]
    return cfg, m, df, f, row, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a-runs", required=True)
    ap.add_argument("--b-runs", default=str(RUNS))
    ap.add_argument("--comparison", default=str(BASE / "reports/week2/a_supervised_comparison.csv"))
    ap.add_argument("--out-dir", default=str(BASE / "reports/week2/external_A"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    compared = {r["run_id"]: r for r in w.read_csv_rows(args.comparison)}
    commit = agg.evaluator_commit()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs, rows, subj_rows, mismatches = {}, [], [], []
    for d in sorted(p for p in Path(args.a_runs).iterdir() if NAME_RX.fullmatch(p.name)):
        cfg, m, df, f, row, bad = ingest(d, commit, compared)
        runs[d.name] = (cfg, m, df, f)
        rows.append(row)
        mismatches += bad
        subj_rows += [{"run_id": d.name, **t} for t in ev.per_subject_table(df).to_dict("records")]

    pd.DataFrame(rows, columns=agg.SUMMARY_FIELDS).to_csv(out / "metrics_summary.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(out / "by_subject.csv", index=False)
    agg.seed_summary(rows).to_csv(out / "seed_summary.csv", index=False)
    pd.DataFrame(mismatches, columns=["run_id", "metric", "recomputed_by_B", "reported_by_owner"]).to_csv(
        out / "mismatch_vs_reported.csv", index=False)

    pairs = []
    for budget in BUDGET_SUBJECTS:
        for seed in range(5):
            ref = f"W2_A_EEGEOG_{budget}_sseed2024_seed{seed}_v01"
            comp = Path(args.b_runs) / f"W2_B_EEGDMhead_{budget}_sseed2024_seed{seed}_v01"
            p = agg.paired_row(runs[ref], agg.load_run(comp), out, args.n_boot)
            p["by_subject_path"] = "by_subject.csv"
            p["limitations"] = agg.LIMITS + ";EEGDM 主干由外部数据预训练,CNN 只见本预算有标签者,是表征比较不是纯架构比较"
            pairs.append(p)
    pd.DataFrame(pairs, columns=agg.PAIR_FIELDS).to_csv(out / "paired_comparison.csv", index=False)
    print(f"{len(rows)} 个 run、{len(pairs)} 组配对;与 A 登记值不一致 {len(mismatches)} 项 -> {out}")


if __name__ == "__main__":
    main()
