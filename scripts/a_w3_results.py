"""a_w3_results.py — 第三周实验A:主矩阵/配对差/EOG对照/曲线/登记汇总(W3 v01)。

输入:9 个 ContraWR run(exp1/runs/W3_A_CONTRAWR_*)与 12 个复用 W2 CNN run
(exp1/runs/W2_A_*),全部六件套 + B 官方 metrics.json。
产出 reports/week3/:
  - a_w3_main_grid.csv        18 格主矩阵(方法×预算×seed,Macro-F1/N1 F1 等)
  - a_w3_paired_deltas.csv    配对差(ContraWR − CNN,官方 --compare 按人配对 bootstrap)
  - a_w3_eog_contrast.csv     EOG 对照(dual − eeg,label25,seed0-2)
  - by_subject/*.csv          逐人宏F1(每对比较)
  - budget_curves_w3.png      5/14/54 人 Macro-F1 与 N1 F1 曲线(3 seed 均值±seed 散布)
  - run_registry.csv / metrics_summary.csv / resource_profile.csv / paired_comparison.csv
  - a_w3_results.md           结果叙述与限制

用法:
    D:/python/python.exe scripts/a_w3_results.py ^
        --runs-dir C:/Users/34364/Desktop/睡眠/exp1/runs
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
VAL_REF = REPO / "manifests" / "val_reference_v1.csv"
EVALUATOR = REPO / "src" / "evaluation" / "evaluate.py"
OUT = REPO / "reports" / "week3"

BUDGETS = ["label10", "label25", "label100"]          # 5/14/54 人
BUDGET_PEOPLE = {"label10": 5, "label25": 14, "label100": 54}
SEEDS = [0, 1, 2]
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

CONTRAWR_ID = lambda b, s: f"W3_A_CONTRAWR_{b}_sseed2024_seed{s}_v01"
CNN_DUAL_ID = lambda b, s: f"W2_A_EEGEOG_{b}_sseed2024_seed{s}_v01"
CNN_EEG_ID = lambda b, s: f"W2_A_EEG_{b}_sseed2024_seed{s}_v01"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_eval(preds: Path, compare: Path | None, out_json: Path) -> dict:
    cmd = [sys.executable, str(EVALUATOR), str(preds), "--reference", str(VAL_REF),
           "--out", str(out_json), "--bootstrap", "1000", "--seed", "0"]
    if compare is not None:
        cmd += ["--compare", str(compare)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"官方评估失败({preds}): {res.stderr.strip()}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def per_subject_macro_f1(preds: Path) -> pd.DataFrame:
    """逐人 Macro-F1(与官方同定义:行真实列预测混淆,五类 F1 平均,零分母 0)。"""
    df = pd.read_csv(preds)
    rows = []
    for sid, g in df.groupby("subject_id", sort=True):
        cm = np.zeros((5, 5), dtype=np.int64)
        np.add.at(cm, (g["y_true"].values.astype(int), g["y_pred"].values.astype(int)), 1)
        tp = np.diag(cm).astype(float)
        prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
        rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
        rows.append({"subject_id": sid, "n_epochs": int(cm.sum()), "macro_f1": float(f1.mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", required=True, help="本地 run 根目录(含 W2 复用与 W3 ContraWR)")
    ap.add_argument("--skip-eval", action="store_true", help="跳过官方 --compare 重跑(调试用)")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    (OUT / "by_subject").mkdir(parents=True, exist_ok=True)
    for p in (runs, VAL_REF, EVALUATOR):
        if not p.exists():
            sys.exit(f"缺少输入: {p}")
    for b in BUDGETS:
        for s in SEEDS:
            for rid in (CONTRAWR_ID(b, s), CNN_DUAL_ID(b, s)):
                if not (runs / rid).exists():
                    sys.exit(f"缺少 run: {runs / rid}")
    for s in SEEDS:
        if not (runs / CNN_EEG_ID("label25", s)).exists():
            sys.exit(f"缺少 run: {runs / CNN_EEG_ID('label25', s)}")

    # ---------------- 主矩阵 + 配对差 + EOG 对照(官方评估器) ----------------
    grid_rows, paired_rows, eog_rows = [], [], []
    for b in BUDGETS:
        for s in SEEDS:
            cw, cn = runs / CONTRAWR_ID(b, s), runs / CNN_DUAL_ID(b, s)
            mc, mn = load_metrics(cw), load_metrics(cn)
            grid_rows.append({
                "method": "ContraWR", "modality": "dual", "budget": b,
                "n_people": BUDGET_PEOPLE[b], "seed": s,
                "run_id": CONTRAWR_ID(b, s),
                "best_epoch": yaml.safe_load((cw / "config.yaml").read_text(encoding="utf-8"))["epochs_run"],
                "macro_f1": mc["macro_f1"], "n1_f1": mc["n1_f1"],
                "n1_precision": mc["n1_precision"], "n1_recall": mc["n1_recall"],
                "f1_W": mc["per_class_f1"]["W"], "f1_N2": mc["per_class_f1"]["N2"],
                "f1_N3": mc["per_class_f1"]["N3"], "f1_REM": mc["per_class_f1"]["REM"],
                "accuracy": mc["accuracy"],
            })
            grid_rows.append({
                "method": "CNN(复用W2)", "modality": "dual", "budget": b,
                "n_people": BUDGET_PEOPLE[b], "seed": s,
                "run_id": CNN_DUAL_ID(b, s),
                "best_epoch": yaml.safe_load((cn / "config.yaml").read_text(encoding="utf-8"))["epochs_run"],
                "macro_f1": mn["macro_f1"], "n1_f1": mn["n1_f1"],
                "n1_precision": mn["n1_precision"], "n1_recall": mn["n1_recall"],
                "f1_W": mn["per_class_f1"]["W"], "f1_N2": mn["per_class_f1"]["N2"],
                "f1_N3": mn["per_class_f1"]["N3"], "f1_REM": mn["per_class_f1"]["REM"],
                "accuracy": mn["accuracy"],
            })
            # 配对差:comparison=ContraWR, reference=CNN(官方 --compare)
            cid = f"W3_CONTRAWR_dual_minus_CNN_dual_{b}_seed{s}"
            bs_csv = OUT / "by_subject" / f"{cid}.csv"
            pa = per_subject_macro_f1(cw / "predictions_val.csv")
            pb = per_subject_macro_f1(cn / "predictions_val.csv")
            by = pa.merge(pb, on="subject_id", suffixes=("_contrawr", "_cnn"))
            by["delta_macro_f1"] = by["macro_f1_contrawr"] - by["macro_f1_cnn"]
            by.to_csv(bs_csv, index=False)
            if args.skip_eval:
                paired_rows.append({"comparison_id": cid})
                continue
            tmp = OUT / f"_tmp_{cid}.json"
            m = run_eval(cw / "predictions_val.csv", cn / "predictions_val.csv", tmp)
            d = m["paired_delta_vs"]
            ci_m = {k: d[k] for k in ("macro_f1", "n1_f1", "accuracy")}
            paired_rows.append({
                "comparison_id": cid,
                "reference_run_id": CNN_DUAL_ID(b, s),
                "comparison_run_id": CONTRAWR_ID(b, s),
                "nominal_budget": b,
                "downstream_seed": s,
                "adaptation_seed": "NA",
                "sample_set_sha256": m["sample_set_sha256"],
                "n_subjects": m["n_subjects"],
                "delta_macro_f1": ci_m["macro_f1"]["point"],
                "delta_macro_f1_ci_low": ci_m["macro_f1"]["lo"],
                "delta_macro_f1_ci_high": ci_m["macro_f1"]["hi"],
                "delta_N1_precision": mc["n1_precision"] - mn["n1_precision"],
                "delta_N1_recall": mc["n1_recall"] - mn["n1_recall"],
                "delta_N1_f1": ci_m["n1_f1"]["point"],
                "delta_N1_f1_ci_low": ci_m["n1_f1"]["lo"],
                "delta_N1_f1_ci_high": ci_m["n1_f1"]["hi"],
                "delta_REM_f1": mc["per_class_f1"]["REM"] - mn["per_class_f1"]["REM"],
                "uncertainty_unit": "subject",
                "ci_method": "paired bootstrap by subject (official evaluate.py, n=1000, 95%)",
                "ci_level": 0.95,
                "training_seed_summary": "reports/week3/a_w3_results.md#种子波动",
                "by_subject_path": str(bs_csv.relative_to(REPO)),
                "limitations": "8 人开发集反复使用;区间含 0 不代表等效;开发集证据",
            })
            tmp.unlink(missing_ok=True)

    for s in SEEDS:
        dual, eeg = runs / CNN_DUAL_ID("label25", s), runs / CNN_EEG_ID("label25", s)
        md, me = load_metrics(dual), load_metrics(eeg)
        cid = f"W3_EOG_dual_minus_eeg_label25_seed{s}"
        bs_csv = OUT / "by_subject" / f"{cid}.csv"
        pa = per_subject_macro_f1(dual / "predictions_val.csv")
        pb = per_subject_macro_f1(eeg / "predictions_val.csv")
        by = pa.merge(pb, on="subject_id", suffixes=("_dual", "_eeg"))
        by["delta_macro_f1"] = by["macro_f1_dual"] - by["macro_f1_eeg"]
        by.to_csv(bs_csv, index=False)
        if args.skip_eval:
            eog_rows.append({"comparison_id": cid})
            continue
        tmp = OUT / f"_tmp_{cid}.json"
        m = run_eval(dual / "predictions_val.csv", eeg / "predictions_val.csv", tmp)
        d = m["paired_delta_vs"]
        ci_m = {k: d[k] for k in ("macro_f1", "n1_f1", "accuracy")}
        eog_rows.append({
            "comparison_id": cid,
            "reference_run_id": CNN_EEG_ID("label25", s),
            "comparison_run_id": CNN_DUAL_ID("label25", s),
            "nominal_budget": "label25",
            "downstream_seed": s,
            "adaptation_seed": "NA",
            "sample_set_sha256": m["sample_set_sha256"],
            "n_subjects": m["n_subjects"],
            "delta_macro_f1": ci_m["macro_f1"]["point"],
            "delta_macro_f1_ci_low": ci_m["macro_f1"]["lo"],
            "delta_macro_f1_ci_high": ci_m["macro_f1"]["hi"],
            "delta_N1_precision": md["n1_precision"] - me["n1_precision"],
            "delta_N1_recall": md["n1_recall"] - me["n1_recall"],
            "delta_N1_f1": ci_m["n1_f1"]["point"],
            "delta_N1_f1_ci_low": ci_m["n1_f1"]["lo"],
            "delta_N1_f1_ci_high": ci_m["n1_f1"]["hi"],
            "delta_REM_f1": md["per_class_f1"]["REM"] - me["per_class_f1"]["REM"],
            "uncertainty_unit": "subject",
            "ci_method": "paired bootstrap by subject (official evaluate.py, n=1000, 95%)",
            "ci_level": 0.95,
            "training_seed_summary": "reports/week3/a_w3_results.md#种子波动",
            "by_subject_path": str(bs_csv.relative_to(REPO)),
            "limitations": "8 人开发集反复使用;区间含 0 不代表等效;开发集证据",
        })
        tmp.unlink(missing_ok=True)

    # ---------------- 曲线 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    people = [BUDGET_PEOPLE[b] for b in BUDGETS]
    for ax, key, title in ((axes[0], "macro_f1", "Macro-F1"),
                           (axes[1], "n1_f1", "N1 F1")):
        for method, rid_fn, color, marker in (
                ("ContraWR (dual)", CONTRAWR_ID, "C0", "o"),
                ("CNN (dual, W2 复用)", CNN_DUAL_ID, "C1", "s")):
            means, seed_pts = [], []
            for b in BUDGETS:
                vals = [load_metrics(runs / rid_fn(b, s))[key] for s in SEEDS]
                means.append(float(np.mean(vals)))
                seed_pts.append(vals)
            ax.plot(people, means, marker=marker, label=method, color=color)
            for x, vs in zip(people, seed_pts):
                if len(vs) > 1:
                    ax.plot([x, x], [min(vs), max(vs)], color=color, lw=1.2, alpha=0.8)
        ax.set_xlabel("有标签受试者人数")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(title)
        ax.set_xticks(people)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("实验A 第三周:CNN 与 ContraWR 标签效率(8人验证集,3 seed 均值,竖线=seed 范围)")
    fig.tight_layout()
    fig.savefig(OUT / "budget_curves_w3.png", dpi=150)
    plt.close(fig)

    # ---------------- 登记表 ----------------
    grid_df = pd.DataFrame(grid_rows)
    paired_df = pd.DataFrame(paired_rows)
    eog_df = pd.DataFrame(eog_rows)

    reg_rows = []
    ms_rows = []
    rp_rows = []
    for b in BUDGETS:
        for s in SEEDS:
            for rid, rep, kind in ((CONTRAWR_ID(b, s), "ContraWR STFT CNN (W3 external baseline)", "trained"),
                                   (CNN_DUAL_ID(b, s), "fair supervised SmallCNN", "reused"),
                                   (CNN_EEG_ID(b, s) if b == "label25" else None,
                                    "fair supervised SmallCNN", "reused")):
                if rid is None:
                    continue
                d = runs / rid
                cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
                m = load_metrics(d)
                res = json.loads((d / "resource.json").read_text(encoding="utf-8"))
                pred_p = d / "predictions_val.csv"
                reg_rows.append({
                    "run_id": rid, "owner": "A",
                    "status": "completed" if kind == "trained" else "reused",
                    "experiment": "W3_E1" if kind == "trained" else "W2_E1",
                    "representation": rep,
                    "modalities": "dual" if "EEGEOG" in rid or "CONTRAWR" in rid else "eeg",
                    "context_length": 1,
                    "nominal_budget": cfg["nominal_budget"],
                    "labeled_subjects": ";".join(cfg["labeled_subjects"]),
                    "unlabeled_signal_subjects": 0,
                    "subset_version": cfg.get("subset_version", "v2"),
                    "subset_sha256": cfg.get("subset_sha256", ""),
                    "sampling_seed": cfg.get("sampling_seed", 2024),
                    "adaptation_version": "NA", "adaptation_seed": "NA",
                    "teacher_id": "NA", "teacher_seed": "NA",
                    "downstream_seed": cfg["downstream_seed"],
                    "code_commit": cfg.get("code_commit", ""),
                    "config_path": "", "normalization_path": "",
                    "backbone_sha256": "", "encoder_sha256": "",
                    "feature_version": cfg.get("code_version", "v01"),
                    "sample_set_path": "manifests/val_reference_v1.csv",
                    "sample_set_sha256": m["sample_set_sha256"],
                    "artifact_index_path": "",
                    "run_storage_uri": "",
                    "reused_from_run_id": "" if kind == "trained" else rid,
                    "reuse_check_path": "" if kind == "trained" else "reports/week3/a_w3_reuse_check.md",
                    "evaluation_status": "official_v1_with_reference",
                    "last_successful_step": "six-piece set + official metrics + val predictions",
                    "failure_reason": "",
                })
                ms_rows.append({
                    "run_id": rid,
                    "representation": rep,
                    "modalities": "dual" if "EEGEOG" in rid or "CONTRAWR" in rid else "eeg",
                    "context_length": 1,
                    "nominal_budget": cfg["nominal_budget"],
                    "labeled_subjects": ";".join(cfg["labeled_subjects"]),
                    "unlabeled_signal_subjects": 0,
                    "adaptation_version": "NA", "adaptation_seed": "NA",
                    "teacher_id": "NA",
                    "downstream_seed": cfg["downstream_seed"],
                    "best_epoch": cfg.get("epochs_run", ""),
                    "macro_f1": m["macro_f1"],
                    "W_f1": m["per_class_f1"]["W"],
                    "N1_precision": m["n1_precision"], "N1_recall": m["n1_recall"],
                    "N1_f1": m["n1_f1"],
                    "N2_f1": m["per_class_f1"]["N2"], "N3_f1": m["per_class_f1"]["N3"],
                    "REM_f1": m["per_class_f1"]["REM"],
                    "sample_count": m["n_epochs"],
                    "subject_count": m["n_subjects"],
                    "metrics_path": "", "predictions_path": "",
                    "predictions_sha256": sha256_file(pred_p),
                    "by_subject_path": "", "n1_errors_path": "",
                    "evaluator_commit": "",
                    "sample_set_sha256": m["sample_set_sha256"],
                })
                rp_rows.append({
                    "run_id": rid, "phase": "train",
                    "device": res["device"],
                    "gpu_memory_gib": 8,
                    "driver": res.get("driver", ""),
                    "pytorch": res["torch"],
                    "cuda": res["cuda"],
                    "precision": res["precision"],
                    "determinism_settings": res["determinism"],
                    "trainable_parameters": res["trainable_parameters"],
                    "actual_updates": res["actual_updates"],
                    "actual_epochs": res["actual_epochs"],
                    "elapsed_seconds": res["elapsed_seconds"],
                    "peak_memory_mib": res["peak_memory_mib"],
                    "examples_per_second": res.get("inference_val_epochs_per_second", ""),
                    "labeled_examples": res["labeled_examples"],
                    "pseudo_candidates": "", "pseudo_accepted": "",
                    "pseudo_training_examples": "",
                    "notes": "W2 复用" if kind == "reused" else "ContraWR 监督,W3",
                })

    def write_csv(name, rows, fieldnames=None):
        if not rows:
            return
        fn = fieldnames or list(rows[0].keys())
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            for r in rows:
                w.writerow({k: "" if v is None else v for k, v in r.items()})

    write_csv("a_w3_main_grid.csv", grid_rows)
    write_csv("a_w3_paired_deltas.csv", paired_rows)
    write_csv("a_w3_eog_contrast.csv", eog_rows)
    write_csv("run_registry.csv", reg_rows)
    write_csv("metrics_summary.csv", ms_rows)
    write_csv("resource_profile.csv", rp_rows)
    write_csv("paired_comparison.csv", paired_rows + eog_rows,
              fieldnames=list(paired_rows[0].keys()))

    print(f"wrote reports/week3/{{a_w3_main_grid,a_w3_paired_deltas,a_w3_eog_contrast,"
          f"run_registry,metrics_summary,resource_profile,paired_comparison}}.csv + "
          f"budget_curves_w3.png + by_subject/")
    print("done")


if __name__ == "__main__":
    main()
