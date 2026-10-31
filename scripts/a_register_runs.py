"""a_register_runs.py — 实验① A:把完成/复用的 run 登记进 reports/week2 各表。

按 RUN_TEMPLATE 与 reports/week2/README.md 的字段规范,从 run 文件夹的
config.yaml/metrics.json/resource.json 读取,追加(或更新)以下各表:
    run_registry.csv              每 run 一行(completed/reused 状态)
    resource_profile.csv          资源行(第一周复用 run 无资源行,在复用核查中注明)
    a_supervised_comparison.csv   A 的监督比较主表(含概率平均行)
    encoder_index.csv             已导出编码器行(按 encoder_id 去重)
CSV 全列按文本存储,数字读取后为字符串,消费方需自行 cast。
重复 id 默认跳过;--overwrite 覆盖同 id 旧行。

第一周 run 复用:--reuse-run-dir 按第一周 config 布局登记,status=reused,并用
B 官方 evaluate.py --reference 重算 metrics_official.json。复用登记要求
--data-dir:程序化复核 训练片段数==132938、channel∈{eeg,eog}、seed==0、
label_frac==1.0,并重算标准化统计与第一周 config 数值逐位比较。

用法:
    D:/python/python.exe scripts/a_register_runs.py \
        --run-dir <runs>/W2_A_EEG_label25_sseed2024_seed0_v01 [--run-dir ...] \
        --reuse-run-dir <runs>/E1_EEG_label100_seed0_v01 --data-dir D:/sleepedf/processed \
        --probavg-dir <runs>/W2_A_PROBAVG_label25_sseed2024_seed0_v01 \
        --storage-prefix team-storage://w2-a [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from train_cnn_w2 import file_sha256, load_records

REGISTRY_COLS = ["run_id", "owner", "status", "experiment", "representation",
                 "modalities", "context_length", "nominal_budget", "labeled_subjects",
                 "unlabeled_signal_subjects", "subset_version", "subset_sha256",
                 "sampling_seed", "adaptation_version", "adaptation_seed", "teacher_id",
                 "teacher_seed", "downstream_seed", "code_commit", "config_path",
                 "normalization_path", "backbone_sha256", "encoder_sha256",
                 "feature_version", "sample_set_path", "sample_set_sha256",
                 "artifact_index_path", "run_storage_uri", "reused_from_run_id",
                 "reuse_check_path", "evaluation_status", "last_successful_step",
                 "failure_reason"]
RESOURCE_COLS = ["run_id", "phase", "device", "gpu_memory_gib", "driver", "pytorch",
                 "cuda", "precision", "determinism_settings", "trainable_parameters",
                 "actual_updates", "actual_epochs", "elapsed_seconds", "peak_memory_mib",
                 "examples_per_second", "labeled_examples", "pseudo_candidates",
                 "pseudo_accepted", "pseudo_training_examples", "notes"]
COMPARISON_COLS = ["run_id", "row_type", "representation", "modalities",
                   "nominal_budget", "labeled_subjects", "downstream_seed",
                   "macro_f1", "accuracy", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM",
                   "n1_precision", "n1_recall", "n1_f1", "rem_f1", "n_val_epochs",
                   "source_eeg_run", "source_eog_run"]
ENCODER_COLS = ["encoder_id", "source_run_id", "modalities", "nominal_budget",
                "labeled_subjects", "subset_version", "subset_sha256", "sampling_seed",
                "downstream_seed", "code_commit", "weights_storage_uri",
                "weights_sha256", "normalization_path", "normalization_sha256",
                "input_shape", "sampling_rate_hz", "channel_units", "output_dim",
                "dtype", "inference_mode", "sample_index_path", "loading_example_path",
                "internal_fold_id"]
SIX_PIECE = ["README.md", "config.yaml", "history.csv", "best.ckpt",
             "metrics.json", "predictions_val.csv"]


def load_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=cols)


def upsert(df: pd.DataFrame, row: dict, overwrite: bool, key_col: str) -> pd.DataFrame:
    row = {k: "" if v is None else str(v) for k, v in row.items()}
    idx = df.index[df[key_col].eq(row[key_col])]
    if len(idx):
        if not overwrite:
            print(f"  skip duplicate {row[key_col]} (--overwrite to replace)")
            return df
        df = df.drop(idx)
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def official_metrics(repo_root: Path, pred: Path, val_ref: Path, out: Path) -> dict:
    """用 B 官方 evaluate.py --reference 重算并落盘,失败即退出。"""
    cmd = [sys.executable, str(repo_root / "src" / "evaluation" / "evaluate.py"),
           str(pred), "--out", str(out), "--reference", str(val_ref)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"B 官方评估失败({pred}): {res.stderr.strip()}")
    return json.loads(out.read_text(encoding="utf-8"))


def pairing_of(pred_csv: Path) -> tuple[str, str]:
    """从源 run 的 config.yaml 取 (nominal_budget, downstream_seed)。"""
    cfg_path = Path(pred_csv).parent / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"找不到源 run 配置: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("nominal_budget"):
        return str(cfg["nominal_budget"]), str(cfg["downstream_seed"])
    if cfg.get("label_frac") == 1.0:
        return "label100", str(cfg.get("seed"))
    sys.exit(f"无法从 {cfg_path} 解析预算/seed")


def register_run(run_dir: Path, repo_root: Path, storage_prefix: str,
                 tables: dict, overwrite: bool):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    run_id = str(cfg["run_id"])
    storage = storage_prefix.rstrip("/") + "/" + run_id
    missing = [f for f in SIX_PIECE if not (run_dir / f).exists()]
    if missing:
        sys.exit(f"{run_id} 六件套不完整,拒绝登记 completed: 缺 {missing}")

    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    res = json.loads((run_dir / "resource.json").read_text(encoding="utf-8")) \
        if (run_dir / "resource.json").exists() else {}
    n_val = sum(1 for _ in open(run_dir / "predictions_val.csv", encoding="utf-8")) - 1

    enc_dir = run_dir / "encoder_export"
    enc_sha, bkb_sha = "", file_sha256(run_dir / "best.ckpt")
    if (enc_dir / "encoder_weights.pt").exists():
        for line in (enc_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            sha, name = line.split("  ")
            if name == "encoder_weights.pt":
                enc_sha = sha
    # 字段语义:backbone_sha256=完整模型 ckpt;encoder_sha256=去分类层编码器产物
    row = {c: "" for c in REGISTRY_COLS}
    row.update({
        "run_id": run_id, "owner": str(cfg.get("owner", "A")), "status": "completed",
        "experiment": "W2_E1", "representation": "fair supervised SmallCNN",
        "modalities": cfg["modality"], "context_length": 1,
        "nominal_budget": cfg["nominal_budget"],
        "labeled_subjects": ";".join(cfg["labeled_subjects"]),
        "unlabeled_signal_subjects": "0",
        "subset_version": cfg["subset_version"], "subset_sha256": cfg["subset_sha256"],
        "sampling_seed": str(cfg["sampling_seed"]),
        "adaptation_version": "NA", "adaptation_seed": "NA",
        "teacher_id": "NA", "teacher_seed": "NA",
        "downstream_seed": str(cfg["downstream_seed"]),
        "code_commit": cfg.get("code_commit", ""),
        "config_path": f"{storage}/config.yaml",
        "normalization_path": f"{storage}/normalization.json",
        "backbone_sha256": bkb_sha, "encoder_sha256": enc_sha,
        "feature_version": cfg.get("code_version", "v01"),
        "sample_set_path": "manifests/val_reference_v1.csv",
        "sample_set_sha256": m.get("sample_set_sha256", ""),
        "artifact_index_path": "reports/week2/artifact_index.csv",
        "run_storage_uri": storage,
        "reused_from_run_id": "", "reuse_check_path": "",
        "evaluation_status": "official_v1_with_reference",
        "last_successful_step": "six-piece set + official metrics + val predictions",
        "failure_reason": "",
    })
    tables["registry"] = upsert(tables["registry"], row, overwrite, "run_id")

    rrow = {c: "" for c in RESOURCE_COLS}
    rrow.update({
        "run_id": run_id, "phase": "train",
        "device": res.get("device", ""), "gpu_memory_gib": "",
        "driver": res.get("driver", ""), "pytorch": res.get("torch", ""),
        "cuda": res.get("cuda", ""),
        "precision": res.get("precision", ""),
        "determinism_settings": res.get("determinism", ""),
        "trainable_parameters": res.get("trainable_parameters", ""),
        "actual_updates": res.get("actual_updates", ""),
        "actual_epochs": res.get("actual_epochs", ""),
        "elapsed_seconds": res.get("elapsed_seconds", ""),
        "peak_memory_mib": res.get("peak_memory_mib", ""),
        "examples_per_second": str(round(res.get("labeled_examples", 0)
                                         / res.get("elapsed_seconds", 1), 1)
                                   if res else ""),
        "labeled_examples": res.get("labeled_examples", ""),
        "pseudo_candidates": "0", "pseudo_accepted": "0",
        "pseudo_training_examples": "0",
        "notes": f"val_examples={res.get('val_examples', '')}",
    })
    tables["resource"] = upsert(tables["resource"], rrow, overwrite, "run_id")

    crow = {c: "" for c in COMPARISON_COLS}
    crow.update({
        "run_id": run_id, "row_type": "trained", "representation": "SmallCNN",
        "modalities": cfg["modality"], "nominal_budget": cfg["nominal_budget"],
        "labeled_subjects": len(cfg["labeled_subjects"]),
        "downstream_seed": cfg["downstream_seed"],
        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        "f1_W": m["per_class_f1"]["W"], "f1_N1": m["per_class_f1"]["N1"],
        "f1_N2": m["per_class_f1"]["N2"], "f1_N3": m["per_class_f1"]["N3"],
        "f1_REM": m["per_class_f1"]["REM"],
        "n1_precision": m["n1_precision"], "n1_recall": m["n1_recall"],
        "n1_f1": m["n1_f1"], "rem_f1": m["per_class_f1"]["REM"],
        "n_val_epochs": n_val, "source_eeg_run": "", "source_eog_run": "",
    })
    tables["comparison"] = upsert(tables["comparison"], crow, overwrite, "run_id")

    if (enc_dir / "encoder_weights.pt").exists():
        sums = {}
        for line in (enc_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            sha, name = line.split("  ")
            sums[name] = sha
        erow = {c: "" for c in ENCODER_COLS}
        erow.update({
            "encoder_id": f"{run_id}_enc", "source_run_id": run_id,
            "modalities": cfg["modality"], "nominal_budget": cfg["nominal_budget"],
            "labeled_subjects": ";".join(cfg["labeled_subjects"]),
            "subset_version": cfg["subset_version"], "subset_sha256": cfg["subset_sha256"],
            "sampling_seed": str(cfg["sampling_seed"]),
            "downstream_seed": str(cfg["downstream_seed"]),
            "code_commit": cfg.get("code_commit", ""),
            "weights_storage_uri": f"{storage}/encoder_export/encoder_weights.pt",
            "weights_sha256": sums.get("encoder_weights.pt", ""),
            "normalization_path": f"{storage}/encoder_export/normalization.json",
            "normalization_sha256": sums.get("normalization.json", ""),
            "input_shape": json.dumps(cfg["input_shape"]),
            "sampling_rate_hz": "100", "channel_units": "uV",
            "output_dim": "128", "dtype": "float32",
            "inference_mode": "eval (no BN update)",
            "sample_index_path": "manifests/val_reference_v1.csv",
            "loading_example_path": "scripts/load_encoder_example.py",
            "internal_fold_id": "NA",
        })
        tables["encoder"] = upsert(tables["encoder"], erow, overwrite, "encoder_id")
    print(f"  registered {run_id}")


def register_reuse(run_dir: Path, repo_root: Path, storage_prefix: str,
                   data_dir: Path, tables: dict, overwrite: bool):
    """第一周 run(EEG/EOG label100 seed0)按 W2 label100 逐项匹配后登记复用。"""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    run_id = str(cfg["run_id"])
    storage = storage_prefix.rstrip("/") + "/" + run_id

    # 程序化逐项复核:通道、seed、100% 标签、训练片段数、标准化数值
    assert cfg.get("channel") in ("eeg", "eog"), f"非单模态第一周 run: {cfg.get('channel')}"
    assert cfg.get("seed") == 0, f"非 seed0 第一周 run: {cfg.get('seed')}"
    assert cfg.get("label_frac") == 1.0, f"非 100% 标签第一周 run: {cfg.get('label_frac')}"
    assert cfg.get("n_train_epochs") == 132938, \
        f"训练片段数 {cfg.get('n_train_epochs')} != label100 的 132938"
    ch_idx = 0 if cfg["channel"] == "eeg" else 1
    split_df = pd.read_csv(repo_root / "manifests" / "split_subjects.csv", dtype=str)
    train_recs = sorted(r for _, row in split_df.iterrows()
                        for r in row["record_ids"].split(";") if row["split"] == "train")
    Xtr, _, _ = load_records(Path(data_dir), train_recs, [ch_idx])
    w1_mean, w1_std = cfg["normalization"]["mean"], cfg["normalization"]["std"]
    assert float(np.float32(Xtr.mean())) == w1_mean and \
        float(np.float32(Xtr.std())) == w1_std, \
        f"标准化统计与第一周 config 不一致: recomputed {Xtr.mean():.8f}/{Xtr.std():.8f} " \
        f"vs config {w1_mean}/{w1_std}"

    train_subjects = sorted(split_df[split_df["split"] == "train"]["subject_id"])
    val_ref = repo_root / "manifests" / "val_reference_v1.csv"
    subset_csv = repo_root / "manifests" / "label_subsets_v2.csv"
    pred = run_dir / "predictions_val.csv"
    off = run_dir / "metrics_official.json"
    if not off.exists() or overwrite:
        m = official_metrics(repo_root, pred, val_ref, off)
    else:
        m = json.loads(off.read_text(encoding="utf-8"))
    n_val = sum(1 for _ in open(pred, encoding="utf-8")) - 1

    row = {c: "" for c in REGISTRY_COLS}
    row.update({
        "run_id": run_id, "owner": "A", "status": "reused",
        "experiment": "W2_E1", "representation": "fair supervised SmallCNN (W1 code)",
        "modalities": cfg["channel"], "context_length": 1,
        "nominal_budget": "label100",
        "labeled_subjects": ";".join(train_subjects),
        "unlabeled_signal_subjects": "0",
        "subset_version": "v2", "subset_sha256": file_sha256(subset_csv),
        "sampling_seed": "2024",
        "adaptation_version": "NA", "adaptation_seed": "NA",
        "teacher_id": "NA", "teacher_seed": "NA",
        "downstream_seed": str(cfg["seed"]),
        "code_commit": "02f120ee80c1c7a7822c33746ce92017632f1faa",
        "config_path": f"{storage}/config.yaml",
        "normalization_path": f"{storage}/config.yaml#normalization",
        "backbone_sha256": file_sha256(run_dir / "best.ckpt"),
        "encoder_sha256": file_sha256(run_dir / "best.ckpt"),
        "feature_version": cfg.get("code_version", "v01"),
        "sample_set_path": "manifests/val_reference_v1.csv",
        "sample_set_sha256": m.get("sample_set_sha256", ""),
        "artifact_index_path": "reports/week2/artifact_index.csv",
        "run_storage_uri": storage,
        "reused_from_run_id": run_id,
        "reuse_check_path": "reports/week2/a_reuse_check_w1.md",
        "evaluation_status": "official_v1_with_reference",
        "last_successful_step": "W1 six-piece set; official re-evaluation passed",
        "failure_reason": "",
    })
    tables["registry"] = upsert(tables["registry"], row, overwrite, "run_id")

    crow = {c: "" for c in COMPARISON_COLS}
    crow.update({
        "run_id": run_id, "row_type": "reused", "representation": "SmallCNN (W1)",
        "modalities": cfg["channel"], "nominal_budget": "label100",
        "labeled_subjects": 54, "downstream_seed": cfg["seed"],
        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        "f1_W": m["per_class_f1"]["W"], "f1_N1": m["per_class_f1"]["N1"],
        "f1_N2": m["per_class_f1"]["N2"], "f1_N3": m["per_class_f1"]["N3"],
        "f1_REM": m["per_class_f1"]["REM"],
        "n1_precision": m["n1_precision"], "n1_recall": m["n1_recall"],
        "n1_f1": m["n1_f1"], "rem_f1": m["per_class_f1"]["REM"],
        "n_val_epochs": n_val, "source_eeg_run": "", "source_eog_run": "",
    })
    tables["comparison"] = upsert(tables["comparison"], crow, overwrite, "run_id")
    print(f"  registered reuse {run_id} (official macro_f1={m['macro_f1']:.4f})")


def register_probavg(prob_dir: Path, repo_root: Path, tables: dict, overwrite: bool):
    prob_dir = Path(prob_dir)
    m = json.loads((prob_dir / "metrics.json").read_text(encoding="utf-8"))
    df = pd.read_csv(prob_dir / "predictions_val.csv")
    readme = (prob_dir / "README.md").read_text(encoding="utf-8")
    budget, seed = "", ""
    for line in readme.splitlines():
        if line.startswith("- 预算/seed:"):
            budget, seed = line.split(":", 1)[1].strip().split("/ seed ")
    # 源表路径在 README 中列出(含 predictions_val.csv 的行)
    src_lines = [l for l in readme.splitlines() if "predictions_val.csv" in l]
    if len(src_lines) >= 2:
        eeg_src, eog_src = src_lines[0].strip(), src_lines[1].strip()
    else:
        eeg_src, eog_src = "", ""
    run_id = prob_dir.name
    crow = {c: "" for c in COMPARISON_COLS}
    crow.update({
        "run_id": run_id, "row_type": "prob_avg",
        "representation": "EEG+EOG single-modal prob average",
        "modalities": "eeg+eog", "nominal_budget": budget,
        "labeled_subjects": "", "downstream_seed": seed,
        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        "f1_W": m["per_class_f1"]["W"], "f1_N1": m["per_class_f1"]["N1"],
        "f1_N2": m["per_class_f1"]["N2"], "f1_N3": m["per_class_f1"]["N3"],
        "f1_REM": m["per_class_f1"]["REM"],
        "n1_precision": m["n1_precision"], "n1_recall": m["n1_recall"],
        "n1_f1": m["n1_f1"], "rem_f1": m["per_class_f1"]["REM"],
        "n_val_epochs": len(df),
        "source_eeg_run": Path(eeg_src).parent.name if eeg_src else "",
        "source_eog_run": Path(eog_src).parent.name if eog_src else "",
    })
    tables["comparison"] = upsert(tables["comparison"], crow, overwrite, "run_id")
    print(f"  registered probavg {run_id}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", action="append", default=[], help="W2 run,可多次给出")
    ap.add_argument("--reuse-run-dir", action="append", default=[], help="第一周 run,可多次")
    ap.add_argument("--probavg-dir", action="append", default=[])
    ap.add_argument("--data-dir", default=None, help="复用核验用 NPZ 目录(登记复用必填)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--storage-prefix", default=None,
                    help="团队存储 URI 前缀(必填,如 Release 下载 URL + #片段)")
    ap.add_argument("--reuse-storage-prefix", default="交付包/实验①交付包_20261018",
                    help="第一周复用 run 的位置前缀(仓库相对路径,默认交付包)")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()

    if args.reuse_run_dir and not args.data_dir:
        sys.exit("--reuse-run-dir 需要 --data-dir(程序化复核标准化统计)")
    if (args.run_dir or args.reuse_run_dir) and not args.storage_prefix:
        sys.exit("--storage-prefix 必填(协议:登记路径须为团队存储 URI 或仓库相对路径)")

    repo_root = Path(args.repo_root)
    w2 = repo_root / "reports" / "week2"
    tables = {
        "registry": load_csv(w2 / "run_registry.csv", REGISTRY_COLS),
        "resource": load_csv(w2 / "resource_profile.csv", RESOURCE_COLS),
        "comparison": load_csv(w2 / "a_supervised_comparison.csv", COMPARISON_COLS),
        "encoder": load_csv(w2 / "encoder_index.csv", ENCODER_COLS),
    }
    for rd in args.run_dir:
        register_run(Path(rd), repo_root, args.storage_prefix, tables, args.overwrite)
    for rd in args.reuse_run_dir:
        register_reuse(Path(rd), repo_root, args.reuse_storage_prefix,
                       Path(args.data_dir), tables, args.overwrite)
    for pd_ in args.probavg_dir:
        register_probavg(Path(pd_), repo_root, tables, args.overwrite)

    tables["registry"].to_csv(w2 / "run_registry.csv", index=False)
    tables["resource"].to_csv(w2 / "resource_profile.csv", index=False)
    tables["comparison"].to_csv(w2 / "a_supervised_comparison.csv", index=False)
    tables["encoder"].to_csv(w2 / "encoder_index.csv", index=False)
    print("wrote reports/week2/{run_registry,resource_profile,"
          "a_supervised_comparison,encoder_index}.csv")


if __name__ == "__main__":
    main()
