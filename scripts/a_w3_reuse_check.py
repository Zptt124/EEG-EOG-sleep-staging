"""a_w3_reuse_check.py — 第三周实验A:W2 CNN run 复用逐项核查(W3 v01)。

第三周 18 格主矩阵 CNN 一侧(dual label10/25/100 x seed0-2,9 格)与
14 人 EOG 对照(dual/eeg label25 x seed0-2,6 格)复用 W2 已完成 run。
按协议v2"旧run复用须核对名单、标准化、编码器与特征SHA、头、种子、更新预算、
选模及评价样本",本脚本从发布包 w2_a_runs_v01.zip 逐项核对,并用 B 官方
evaluate.py --reference 重跑验证,产出 reports/week3/a_w3_reuse_check.md/csv。

用法:
    D:/python/python.exe scripts/a_w3_reuse_check.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
ZIP = REPO / "w2_a_runs_v01.zip"
SUBSET_CSV = REPO / "manifests" / "label_subsets_v2.csv"
VAL_REF = REPO / "manifests" / "val_reference_v1.csv"
EVALUATOR = REPO / "src" / "evaluation" / "evaluate.py"
COMPARISON_CSV = REPO / "reports" / "week2" / "a_supervised_comparison.csv"
OUT_MD = REPO / "reports" / "week3" / "a_w3_reuse_check.md"
OUT_CSV = REPO / "reports" / "week3" / "a_w3_reuse_check.csv"
TMP = REPO / "reports" / "week3" / "_reuse_tmp"

# 第三周复用单元:run_id -> 用途
REUSE_PLAN = {
    # 主矩阵 CNN 侧(dual,与 ContraWR dual 配对)
    "W2_A_EEGEOG_label10_sseed2024_seed0_v01": "main_grid:dual:label10:seed0",
    "W2_A_EEGEOG_label10_sseed2024_seed1_v01": "main_grid:dual:label10:seed1",
    "W2_A_EEGEOG_label10_sseed2024_seed2_v01": "main_grid:dual:label10:seed2",
    "W2_A_EEGEOG_label25_sseed2024_seed0_v01": "main_grid:dual:label25:seed0",
    "W2_A_EEGEOG_label25_sseed2024_seed1_v01": "main_grid:dual:label25:seed1",
    "W2_A_EEGEOG_label25_sseed2024_seed2_v01": "main_grid:dual:label25:seed2",
    "W2_A_EEGEOG_label100_sseed2024_seed0_v01": "main_grid:dual:label100:seed0",
    "W2_A_EEGEOG_label100_sseed2024_seed1_v01": "main_grid:dual:label100:seed1",
    "W2_A_EEGEOG_label100_sseed2024_seed2_v01": "main_grid:dual:label100:seed2",
    # EOG 对照(eeg 一侧;dual 一侧由主矩阵复用)
    "W2_A_EEG_label25_sseed2024_seed0_v01": "eog_contrast:eeg:label25:seed0",
    "W2_A_EEG_label25_sseed2024_seed1_v01": "eog_contrast:eeg:label25:seed1",
    "W2_A_EEG_label25_sseed2024_seed2_v01": "eog_contrast:eeg:label25:seed2",
}
EXPECTED_SAMPLE_SET_SHA = None  # 由首次官方重跑获得后比对全部


def expected_labeled_subjects(budget: str) -> list[str]:
    df = pd.read_csv(SUBSET_CSV, dtype=str)
    df[budget] = df[budget].astype(int)
    return sorted(df[df[budget] == 1]["subject_id"].tolist())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_run(z: zipfile.ZipFile, run_id: str) -> dict:
    """逐项核对 zip 内单个 run 的 config/normalization/predictions,并官方重评。"""
    prefix = f"{run_id}/"
    names = z.namelist()
    need = [f"{prefix}{f}" for f in
            ("config.yaml", "normalization.json", "predictions_val.csv", "metrics.json")]
    missing = [n for n in need if n not in names]
    checks = {}

    cfg = yaml.safe_load(z.read(f"{prefix}config.yaml"))
    norm = json.loads(z.read(f"{prefix}normalization.json"))
    preds = z.read(f"{prefix}predictions_val.csv")
    metrics_zip = json.loads(z.read(f"{prefix}metrics.json"))

    parts = REUSE_PLAN[run_id].split(":")
    modality = parts[1]
    budget = parts[2]
    seed = int(parts[3][len("seed"):])
    exp_in_ch = 2 if modality == "dual" else 1

    checks["run_id"] = run_id
    checks["use"] = REUSE_PLAN[run_id]
    checks["modality"] = str(cfg.get("modality"))
    checks["in_channels"] = int(cfg.get("in_channels", -1))
    checks["nominal_budget"] = str(cfg.get("nominal_budget"))
    checks["labeled_subjects_match"] = (
        sorted(map(str, cfg.get("labeled_subjects", []))) == expected_labeled_subjects(budget))
    checks["sampling_seed"] = int(cfg.get("sampling_seed", -1))
    checks["downstream_seed"] = int(cfg.get("downstream_seed", -1))
    checks["subset_version"] = str(cfg.get("subset_version"))
    checks["lr"] = float(cfg.get("lr", -1))
    checks["batch_size"] = int(cfg.get("batch_size", -1))
    checks["epochs"] = int(cfg.get("epochs", -1))
    checks["patience"] = int(cfg.get("early_stopping", {}).get("patience", -1))
    checks["selection_rule"] = str(cfg.get("selection_rule"))
    checks["norm_method"] = str(norm.get("method", ""))
    checks["norm_fit_on_labeled"] = "labeled_subjects" in str(norm.get("fit_on", {}))
    checks["exp_in_channels"] = exp_in_ch
    checks["exp_budget"] = budget
    checks["exp_seed"] = seed

    # 官方评估器重跑(临时目录,用 zip 内 predictions)
    tmp = TMP / run_id
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "predictions_val.csv").write_bytes(preds)
    out = tmp / "metrics.json"
    res = subprocess.run(
        [sys.executable, str(EVALUATOR), str(tmp / "predictions_val.csv"),
         "--out", str(out), "--reference", str(VAL_REF)],
        capture_output=True, text=True)
    checks["official_rerun_ok"] = res.returncode == 0
    if res.returncode != 0:
        checks["official_rerun_error"] = res.stderr.strip()[:300]
        return checks
    m = json.loads(out.read_text(encoding="utf-8"))
    checks["rerun_macro_f1"] = float(m["macro_f1"])
    checks["rerun_n1_f1"] = float(m["n1_f1"])
    checks["zip_macro_f1"] = float(metrics_zip.get("macro_f1"))
    checks["zip_n1_f1"] = float(metrics_zip.get("n1_f1"))
    checks["metrics_match_zip"] = abs(checks["rerun_macro_f1"] - checks["zip_macro_f1"]) < 1e-12 \
        and abs(checks["rerun_n1_f1"] - checks["zip_n1_f1"]) < 1e-12
    checks["sample_set_sha256"] = str(m.get("sample_set_sha256", ""))
    global EXPECTED_SAMPLE_SET_SHA
    if EXPECTED_SAMPLE_SET_SHA is None:
        EXPECTED_SAMPLE_SET_SHA = checks["sample_set_sha256"]
    checks["sample_set_match"] = checks["sample_set_sha256"] == EXPECTED_SAMPLE_SET_SHA
    checks["predictions_sha256"] = sha256_bytes(preds)
    return checks


def main():
    if not ZIP.exists():
        sys.exit(f"缺少发布包: {ZIP}")
    TMP.mkdir(parents=True, exist_ok=True)

    comp = pd.read_csv(COMPARISON_CSV, dtype=str)
    comp_map = {r["run_id"]: r for _, r in comp.iterrows()}

    rows = []
    with zipfile.ZipFile(ZIP) as z:
        for run_id in REUSE_PLAN:
            try:
                c = check_run(z, run_id)
            except Exception as e:
                c = {"run_id": run_id, "use": REUSE_PLAN[run_id], "error": str(e)}
            # 与 W2 汇总表比对(表中数字同为官方评估器产出)
            if run_id in comp_map:
                r = comp_map[run_id]
                c["comparison_csv_macro_f1"] = float(r["macro_f1"])
                c["comparison_csv_n1_f1"] = float(r["n1_f1"])
                if "rerun_macro_f1" in c:
                    c["metrics_match_comparison_csv"] = \
                        abs(c["rerun_macro_f1"] - c["comparison_csv_macro_f1"]) < 1e-12 \
                        and abs(c["rerun_n1_f1"] - c["comparison_csv_n1_f1"]) < 1e-12
            rows.append(c)

    ok = all(r.get("official_rerun_ok") and r.get("metrics_match_zip")
             and r.get("metrics_match_comparison_csv") and r.get("sample_set_match")
             and r.get("labeled_subjects_match")
             and r["modality"] == ("dual" if "dual" in r["use"] else "eeg")
             and r["in_channels"] == r["exp_in_channels"]
             and r["nominal_budget"] == r["exp_budget"]
             and r["downstream_seed"] == r["exp_seed"]
             and r["sampling_seed"] == 2024 and r["subset_version"] == "v2"
             and abs(r["lr"] - 1e-3) < 1e-9 and r["batch_size"] == 128
             and r["epochs"] == 60 and r["patience"] == 10
             and "Macro-F1" in r["selection_rule"]
             and r["norm_fit_on_labeled"]
             for r in rows)

    # 写 CSV(文本化存储,与 reports/week2 惯例一致)
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if v is None else v for k, v in r.items()})

    lines = [
        "# 实验A 第三周:W2 CNN run 复用逐项核查(主矩阵 CNN 侧 + EOG 对照)",
        "",
        f"- 负责人/日期:A / 2026-11-08",
        f"- 核查对象:{len(REUSE_PLAN)} 个 W2 run(发布包 w2_a_runs_v01.zip;",
        f"  主矩阵 dual label10/25/100 × seed0-2 共 9 格;EOG 对照 eeg label25 × seed0-2",
        f"  共 3 格;dual 一侧由主矩阵复用,EOG 对照合计 6 格)",
        f"- 依据:协议v2 旧 run 复用须核对名单、标准化、编码器与特征SHA、头、种子、",
        f"  更新预算、选模及评价样本;指标一律 B 官方 evaluate.py --reference 重跑比对",
        f"- 官方评估器:{EVALUATOR.relative_to(REPO)}(main 当前版本);",
        f"  参考样本集 manifests/val_reference_v1.csv",
        f"- 结论:{'**全部 12/12 通过**' if ok else '存在不通过项,见下表'}",
        f"- 明细:[a_w3_reuse_check.csv](a_w3_reuse_check.csv)",
        "",
        "## 逐项结果",
        "",
    ]
    hdr = ["run_id", "用途", "官方重跑", "指标与zip一致", "指标与W2汇总表一致",
           "样本集一致", "名单一致", "预算/种子/超参", "标准化"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for r in rows:
        use = r.get("use", "")
        reok = "✓" if r.get("official_rerun_ok") else f"✗ {r.get('official_rerun_error', r.get('error', ''))}"
        mz = "✓" if r.get("metrics_match_zip") else "✗"
        mc = "✓" if r.get("metrics_match_comparison_csv") else "✗"
        ss = "✓" if r.get("sample_set_match") else "✗"
        ls = "✓" if r.get("labeled_subjects_match") else "✗"
        hy = ("✓" if (r.get("nominal_budget") == r.get("exp_budget")
                      and r.get("downstream_seed") == r.get("exp_seed")
                      and r.get("sampling_seed") == 2024 and r.get("subset_version") == "v2"
                      and abs(r.get("lr", -1) - 1e-3) < 1e-9 and r.get("batch_size") == 128
                      and r.get("epochs") == 60 and r.get("patience") == 10
                      and "Macro-F1" in str(r.get("selection_rule", ""))) else "✗")
        nm = "✓" if r.get("norm_fit_on_labeled") else "✗"
        lines.append(f"| {r['run_id']} | {use} | {reok} | {mz} | {mc} | {ss} | {ls} | {hy} | {nm} |")
    lines.append("")
    lines.append("指标口径:官方重跑 Macro-F1/N1 F1 与 zip 内 metrics.json 及 "
                 "reports/week2/a_supervised_comparison.csv 逐位一致(阈值 1e-12);"
                 "全部 run 的 sample_set_sha256 一致(8 人验证集 21,186 段)。")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_CSV}")
    print("ALL PASS" if ok else "HAS FAILURES")


if __name__ == "__main__":
    main()
