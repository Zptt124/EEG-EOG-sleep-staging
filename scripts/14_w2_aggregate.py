"""Step W2-B6: 滚动汇总——metrics_summary.csv、seed_summary.csv、paired_comparison.csv、run_registry(B 的行)。

输入:runs 根目录下的 W2_* run(须有 config.yaml / metrics.json / predictions_val.csv);
每个 run 的预测表都会重新过官方 evaluate.py v1 + 验证参考清单,指标以重算为准,不信任旧 metrics.json。
输出(可反复重算,同名文件会被覆盖——它们由 run 目录完全决定):
  metrics_summary.csv  每个 run 一行,同一最佳权重的 Macro-F1、各类 F1、N1 P/R/F1
  seed_summary.csv     按(表征,上下文,预算,…)分组的五 seed 均值/标准差(种子波动,不含人群不确定性)
  paired_comparison.csv  同一批 8 人上的配对差(comparison 减 reference),按受试者 bootstrap
  每个 run 目录内:by_subject.csv、n1_errors.csv(N1 误判去向)
种子波动与受试者不确定性分开:seed_summary 只回答"换训练种子",paired_comparison 的区间只回答"换这 8 人",
二者都不能把 5 seed × 8 人当作 40 个独立样本;区间含 0 不代表方法等效,8 人仅为开发集探索证据。

用法:
  python scripts/14_w2_aggregate.py --runs-root runs --out-dir reports/week2 \
      [--pair-context] [--pair REF_RUN_ID COMP_RUN_ID ...] [--update-registry]
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import BASE, RUNS
import w2_common as w

ev = w.ev
SUMMARY_FIELDS = ["run_id", "representation", "modalities", "context_length", "nominal_budget", "labeled_subjects",
                  "unlabeled_signal_subjects", "adaptation_version", "adaptation_seed", "teacher_id", "downstream_seed",
                  "best_epoch", "macro_f1", "W_f1", "N1_precision", "N1_recall", "N1_f1", "N2_f1", "N3_f1", "REM_f1",
                  "sample_count", "subject_count", "metrics_path", "predictions_path", "predictions_sha256",
                  "by_subject_path", "n1_errors_path", "evaluator_commit", "sample_set_sha256"]
PAIR_FIELDS = ["comparison_id", "reference_run_id", "comparison_run_id", "nominal_budget", "downstream_seed",
               "adaptation_seed", "sample_set_sha256", "n_subjects", "delta_macro_f1", "delta_N1_precision",
               "delta_N1_recall", "delta_N1_f1", "delta_REM_f1", "uncertainty_unit", "ci_method", "ci_level",
               "delta_macro_f1_ci_low", "delta_macro_f1_ci_high", "delta_N1_f1_ci_low", "delta_N1_f1_ci_high",
               "training_seed_summary", "by_subject_path", "limitations"]
LIMITS = "8人开发集探索性证据;区间含0不代表等效;训练种子波动另见seed_summary,不与按人区间混合"
# head 必须分组:线性头与长度 1 的 BiGRU 头 context_length 都是 1,但不是同一种模型
GROUP_KEYS = ["head", "representation", "modalities", "context_length", "nominal_budget", "labeled_subjects",
              "unlabeled_signal_subjects", "adaptation_version", "adaptation_seed", "teacher_id"]


def evaluator_commit():
    path = Path(ev.__file__)
    try:
        git = lambda *a: subprocess.run(["git", *a], cwd=BASE, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=BASE).returncode != 0
        tracked = git("log", "-1", "--format=%H", "--", str(path.relative_to(BASE)))
        return tracked + ("+dirty" if dirty else "") if tracked else f"untracked-sha256:{w.sha256_file(path)[:12]}"
    except Exception:
        return f"sha256:{w.sha256_file(path)[:12]}"


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(BASE))
    except ValueError:
        return str(path)


def na(v):
    return "NA" if v is None else v


def load_run(run_dir):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    pred_path = run_dir / "predictions_val.csv"
    m = ev.evaluate(pred_path, reference=w.VAL_REFERENCE)         # 重新核验并重算
    m["best_epoch"] = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["best_epoch"]
    df = ev.load_predictions(pred_path)
    return cfg, m, df, pred_path


def write_run_tables(run_dir, df, m):
    run_dir = Path(run_dir)
    ev.per_subject_table(df).to_csv(run_dir / "by_subject.csv", index=False)
    cm = np.array(m["confusion_matrix"])
    pd.DataFrame([cm[1]], columns=[f"to_{s}" for s in ev.STAGE_NAMES]).assign(
        n1_support=int(cm[1].sum())).to_csv(run_dir / "n1_errors.csv", index=False)


def summary_row(run_dir, cfg, m, commit):
    run_dir = Path(run_dir)
    f1 = m["per_class_f1"]
    return {"run_id": cfg["run_id"], "representation": cfg["representation"], "modalities": cfg.get("modalities", "EEG+EOG"),
            "context_length": cfg["context_length"], "nominal_budget": cfg["nominal_budget"],
            "labeled_subjects": cfg["labeled_subjects"], "unlabeled_signal_subjects": cfg["unlabeled_signal_subjects"],
            "adaptation_version": "NA", "adaptation_seed": "NA", "teacher_id": "NA",
            "downstream_seed": cfg["downstream_seed"], "best_epoch": m["best_epoch"],
            "macro_f1": f"{m['macro_f1']:.6f}", "W_f1": f"{f1['W']:.6f}", "N1_precision": f"{m['n1_precision']:.6f}",
            "N1_recall": f"{m['n1_recall']:.6f}", "N1_f1": f"{m['n1_f1']:.6f}", "N2_f1": f"{f1['N2']:.6f}",
            "N3_f1": f"{f1['N3']:.6f}", "REM_f1": f"{f1['REM']:.6f}", "sample_count": m["n_epochs"],
            "subject_count": m["n_subjects"], "metrics_path": rel(run_dir / "metrics.json"),
            "predictions_path": rel(run_dir / "predictions_val.csv"),
            "predictions_sha256": w.sha256_file(run_dir / "predictions_val.csv"),
            "by_subject_path": rel(run_dir / "by_subject.csv"), "n1_errors_path": rel(run_dir / "n1_errors.csv"),
            "evaluator_commit": commit, "sample_set_sha256": m["sample_set_sha256"]}


def paired_row(ref, comp, out_dir, n_boot=1000):
    (cfg_r, m_r, df_r, _), (cfg_c, m_c, df_c, _) = ref, comp
    if m_r["sample_set_sha256"] != m_c["sample_set_sha256"]:
        raise ValueError(f"样本集合不同,不能配对: {cfg_r['run_id']} vs {cfg_c['run_id']}")
    d = ev.paired_bootstrap_delta(df_c, df_r, n_boot=n_boot, seed=0)      # comparison - reference
    delta = lambda k: m_c[k] - m_r[k]
    return {"comparison_id": f"{cfg_c['run_id']}__minus__{cfg_r['run_id']}", "reference_run_id": cfg_r["run_id"],
            "comparison_run_id": cfg_c["run_id"], "nominal_budget": cfg_c["nominal_budget"],
            "downstream_seed": cfg_c["downstream_seed"], "adaptation_seed": "NA",
            "sample_set_sha256": m_c["sample_set_sha256"], "n_subjects": d["n_subjects"],
            "delta_macro_f1": f"{delta('macro_f1'):+.6f}", "delta_N1_precision": f"{delta('n1_precision'):+.6f}",
            "delta_N1_recall": f"{delta('n1_recall'):+.6f}", "delta_N1_f1": f"{delta('n1_f1'):+.6f}",
            "delta_REM_f1": f"{m_c['per_class_f1']['REM'] - m_r['per_class_f1']['REM']:+.6f}",
            "uncertainty_unit": "subject", "ci_method": f"paired_bootstrap_subjects_n{n_boot}_seed0", "ci_level": d["level"],
            "delta_macro_f1_ci_low": f"{d['macro_f1']['lo']:+.6f}", "delta_macro_f1_ci_high": f"{d['macro_f1']['hi']:+.6f}",
            "delta_N1_f1_ci_low": f"{d['n1_f1']['lo']:+.6f}", "delta_N1_f1_ci_high": f"{d['n1_f1']['hi']:+.6f}",
            "training_seed_summary": rel(Path(out_dir) / "seed_summary.csv"),
            "by_subject_path": "see run directories", "limitations": LIMITS}


def seed_summary(rows):
    df = pd.DataFrame(rows)
    df["head"] = ["bigru_context" if "_CTX" in r else "linear" for r in df["run_id"]]
    for c in ("macro_f1", "N1_precision", "N1_recall", "N1_f1", "REM_f1"):
        df[c] = df[c].astype(float)
    g = df.groupby(GROUP_KEYS, dropna=False)
    out = g.agg(n_seeds=("downstream_seed", "nunique"), seeds=("downstream_seed", lambda s: ";".join(map(str, sorted(s)))),
                **{f"{c}_mean": (c, "mean") for c in ("macro_f1", "N1_precision", "N1_recall", "N1_f1", "REM_f1")},
                **{f"{c}_std": (c, lambda s: s.std(ddof=1) if len(s) > 1 else float("nan"))
                   for c in ("macro_f1", "N1_f1")}).reset_index()
    out["note"] = "std 为训练种子间样本标准差(ddof=1);只有 1 个种子时为 NaN;不含受试者不确定性"
    return out.round(6)


def registry_row(run_dir, cfg, m):
    run_dir = Path(run_dir)
    na_ = "NA"
    return {"run_id": cfg["run_id"], "owner": "B", "status": "completed", "experiment": cfg.get("experiment", "②"),
            "representation": cfg["representation"], "modalities": cfg.get("modalities", "EEG+EOG"),
            "context_length": cfg["context_length"], "nominal_budget": cfg["nominal_budget"],
            "labeled_subjects": cfg["labeled_subjects"], "unlabeled_signal_subjects": cfg["unlabeled_signal_subjects"],
            "subset_version": "v2", "subset_sha256": cfg.get("subset_sha256", na_), "sampling_seed": cfg["sampling_seed"],
            "adaptation_version": na_, "adaptation_seed": na_, "teacher_id": na_, "teacher_seed": na_,
            "downstream_seed": cfg["downstream_seed"], "code_commit": cfg.get("code_commit", na_),
            "config_path": rel(run_dir / "config.yaml"), "normalization_path": rel(run_dir / "normalization.npz"),
            "backbone_sha256": na_, "encoder_sha256": (w.sha256_file(cfg["encoder_features"]) if cfg.get("encoder_features")
                                                       and Path(cfg["encoder_features"]).exists() else na_),
            "feature_version": cfg.get("eeg_index_sha256", na_), "sample_set_path": rel(w.VAL_REFERENCE),
            "sample_set_sha256": m["sample_set_sha256"], "artifact_index_path": "reports/week2/artifact_index.csv", "run_storage_uri": rel(run_dir),
            "reused_from_run_id": na_, "reuse_check_path": na_,
            "evaluation_status": "evaluate_v1_with_reference_passed", "last_successful_step": "predictions_val.csv evaluated",
            "failure_reason": na_}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs-root", default=str(RUNS))
    ap.add_argument("--out-dir", default=str(BASE / "reports" / "week2"))
    ap.add_argument("--pair-context", action="store_true", help="同表征同 seed:长度 3 减 长度 1")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("REF", "COMP"), default=[],
                    help="额外配对:参照 run_id 与比较 run_id(比较减参照)")
    ap.add_argument("--update-registry", action="store_true", help="更新 run_registry.csv 中 B 的行(保留他人行)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    runs = {}
    for d in sorted(Path(args.runs_root).glob("W2_*")):
        if (d / "config.yaml").exists() and (d / "predictions_val.csv").exists():
            runs[d.name] = (d, *load_run(d))
    if not runs:
        sys.exit("错误: 没有找到可汇总的 W2_* run")
    commit = evaluator_commit()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, (d, cfg, m, df, _) in runs.items():
        write_run_tables(d, df, m)
        rows.append(summary_row(d, cfg, m, commit))
    pd.DataFrame(rows, columns=SUMMARY_FIELDS).to_csv(out_dir / "metrics_summary.csv", index=False)
    seed_summary(rows).to_csv(out_dir / "seed_summary.csv", index=False)

    pairs = [tuple(p) for p in args.pair]
    if args.pair_context:
        for name, (_, cfg, *_rest) in runs.items():
            if cfg["context_length"] == 3:
                ref = name.replace("_CTX3_", "_CTX1_")
                if ref in runs:
                    pairs.append((ref, name))
    pair_rows = [paired_row(tuple(runs[r][1:]), tuple(runs[c][1:]), out_dir, args.n_boot) for r, c in pairs]
    pd.DataFrame(pair_rows, columns=PAIR_FIELDS).to_csv(out_dir / "paired_comparison.csv", index=False)

    if args.update_registry:
        path = out_dir / "run_registry.csv"
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields, old = reader.fieldnames, [r for r in reader if r["owner"] != "B"]
        new = [registry_row(d, cfg, m) for d, cfg, m, *_ in runs.values()]
        with open(path, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            wr.writeheader()
            wr.writerows(old + [{k: r.get(k, "NA") for k in fields} for r in new])
    print(f"{len(rows)} 个 run -> metrics_summary.csv;{len(pair_rows)} 组配对 -> paired_comparison.csv;evaluator={commit}")


if __name__ == "__main__":
    main()
