"""make_resource_profile.py - Week-2 Experiment 4 (D): resource table in the team's resource_profile.csv format.

Columns are identical to reports/week2/resource_profile.csv on main (A and B use the same header), so B can
append D's rows directly. One row per:
  - student run (D0/D1/D2, R1/R2)            phase = train_student
  - teacher head (dual/eeg/eog per seed)      phase = train_teacher   (dual teacher is the same model as D0)
  - R2 calibration fold encoder (A's script)  phase = calibration_fold_encoder (copied from A's resource.json)
  - calibration runs                          phase = calibration
  - feature caches (R1 pooling, R2 extraction) phase = feature_extraction
Linear heads ran on CPU (4 threads, deterministic); their peak RAM was not measured (reported as NA).

Usage (Ubuntu):
  python -u make_resource_profile.py --exp-dir <D_local_project>/week2/exp4_ssl --fold-runs <D_local_r2_fold_cnns>
      --cache-root <D_local_cache> --output <D_local_project>/week2/exp4_ssl/analysis/resource_profile_D.csv
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

FIELDS = ["run_id", "phase", "device", "gpu_memory_gib", "driver", "pytorch", "cuda", "precision",
          "determinism_settings", "trainable_parameters", "actual_updates", "actual_epochs", "elapsed_seconds",
          "peak_memory_mib", "examples_per_second", "labeled_examples", "pseudo_candidates", "pseudo_accepted",
          "pseudo_training_examples", "notes"]
CPU_DET = "torch.manual_seed before model init; torch.use_deterministic_algorithms(True); 4 CPU threads"


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip().split(",")
        return round(float(out[0]) / 1024, 2), out[1].strip()
    except Exception:
        return "NA", "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--fold-runs", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        print(f"ERROR: {args.output} already exists")
        sys.exit(1)
    gpu_gib, driver = gpu_info()
    base = {"pytorch": str(torch.__version__), "cuda": str(torch.version.cuda), "precision": "float32", "driver": driver}
    out = []

    # ---------- student runs ----------
    runs = read_csv(args.exp_dir / "runs" / "run_index.csv")
    for r in runs:
        d = args.exp_dir / "runs" / r["run_id"]
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        dim = 5120 if r["repr"] == "R1" else 256
        cand = acc = ""
        if r["method"] in ("D1", "D2"):
            st = read_csv(d / "pseudo_label_stats.csv")
            cand = sum(int(x["n_candidates"]) for x in st)
            acc = sum(int(x["n_accepted"]) for x in st)
        seen = int(m.get("pseudo_samples_seen", 0))
        n_lab = int(r["n_labeled_samples"])
        secs = float(r["train_seconds"])
        out.append({**base, "run_id": r["run_id"], "phase": "train_student", "device": "cpu", "gpu_memory_gib": "NA",
                    "determinism_settings": CPU_DET, "trainable_parameters": dim * 5 + 5,
                    "actual_updates": r["n_updates"], "actual_epochs": r["n_epochs_run"], "elapsed_seconds": secs,
                    "peak_memory_mib": "NA", "examples_per_second": round((n_lab * int(r["n_epochs_run"]) + seen) / max(secs, 1e-9), 1),
                    "labeled_examples": n_lab, "pseudo_candidates": cand, "pseudo_accepted": acc,
                    "pseudo_training_examples": seen,
                    "notes": "D0 student is the same model as the dual teacher" if r["method"] == "D0" else
                             "mixed batches: 128 labeled + 384 pseudo per step; steps per epoch = labeled batches"})

    # ---------- teacher heads ----------
    for t in read_csv(args.exp_dir / "runs" / "teacher_index.csv"):
        dim = {"dual": 5120, "eeg": 2560, "eog": 2560}[t["modality"]] if t["repr"] == "R1" else {"dual": 256, "eeg": 128, "eog": 128}[t["modality"]]
        out.append({**base, "run_id": t["teacher_id"], "phase": "train_teacher", "device": "cpu", "gpu_memory_gib": "NA",
                    "determinism_settings": CPU_DET, "trainable_parameters": dim * 5 + 5, "actual_updates": t["n_updates"],
                    "actual_epochs": "", "elapsed_seconds": t["train_seconds"], "peak_memory_mib": "NA",
                    "examples_per_second": "", "labeled_examples": t["n_labeled_samples"], "pseudo_candidates": "",
                    "pseudo_accepted": "", "pseudo_training_examples": 0,
                    "notes": "dual teacher identical to D0 of the same seed (time counted once in train_student)"
                             if t["modality"] == "dual" else "single-modality teacher used by D2 agreement rule"})

    # ---------- R2 calibration fold encoders (A's training script) ----------
    for f in read_csv(args.fold_runs / "fold_index.csv"):
        res = json.loads((args.fold_runs / f["run_id"] / "resource.json").read_text(encoding="utf-8"))
        out.append({**base, "run_id": f["run_id"], "phase": "calibration_fold_encoder", "device": res.get("device", ""),
                    "gpu_memory_gib": gpu_gib, "driver": res.get("driver", driver), "pytorch": res.get("torch", base["pytorch"]),
                    "cuda": res.get("cuda", base["cuda"]), "precision": res.get("precision", "float32"),
                    "determinism_settings": res.get("determinism", ""), "trainable_parameters": res.get("trainable_parameters", ""),
                    "actual_updates": res.get("actual_updates", ""), "actual_epochs": res.get("actual_epochs", ""),
                    "elapsed_seconds": res.get("elapsed_seconds", ""), "peak_memory_mib": res.get("peak_memory_mib", ""),
                    "examples_per_second": "", "labeled_examples": res.get("labeled_examples", ""),
                    "pseudo_candidates": "", "pseudo_accepted": "", "pseudo_training_examples": 0,
                    "notes": f"{f['budget']} fold {f['fold']} held-out {f['heldout_subject']} {f['modality']}; A's train_cnn_w2.py --subjects, seed 0"})

    # ---------- calibrations ----------
    for d in sorted((args.exp_dir / "calibration").glob("*/calibration_decision.json")):
        dec = json.loads(d.read_text(encoding="utf-8"))
        rep = dec.get("representation", "R1")
        out.append({**base, "run_id": f"W2_D_calibration_{rep}_{dec['budget']}", "phase": "calibration",
                    "device": "cuda+cpu" if rep == "R2" else "cpu", "gpu_memory_gib": gpu_gib if rep == "R2" else "NA",
                    "determinism_settings": CPU_DET, "trainable_parameters": "", "actual_updates": "", "actual_epochs": "",
                    "elapsed_seconds": round(float(dec["minutes"]) * 60, 1), "peak_memory_mib": "NA", "examples_per_second": "",
                    "labeled_examples": "", "pseudo_candidates": "", "pseudo_accepted": "", "pseudo_training_examples": 0,
                    "notes": f"LOSO {dec['n_folds']} folds x 3 linear teachers; chosen p {dec['chosen_p']}"
                             + ("; R2 fold-encoder training time is in calibration_fold_encoder rows" if rep == "R2" else "")})

    # ---------- feature caches ----------
    r1 = json.loads((args.cache_root / "R1_eegdm_cond14_pooled_v01" / "cache_manifest.json").read_text(encoding="utf-8"))
    out.append({**base, "run_id": "R1_eegdm_cond14_pooled_v01", "phase": "feature_extraction", "device": "cpu",
                "gpu_memory_gib": "NA", "determinism_settings": "no randomness (mean pooling of existing frozen features)",
                "trainable_parameters": 0, "actual_updates": 0, "actual_epochs": 0,
                "elapsed_seconds": round(float(r1["minutes"]) * 60, 1), "peak_memory_mib": "NA", "examples_per_second": "",
                "labeled_examples": "", "pseudo_candidates": "", "pseudo_accepted": "", "pseudo_training_examples": 0,
                "notes": f"time pooling of week-1 EEGDM caches to 2560 per modality, {r1['n_samples']} samples; "
                         "original EEGDM extraction time is in week-1 logs"})
    r2 = json.loads(next(args.cache_root.glob("R2_cnn_label25_seed0_v01/cache_manifest.json")).read_text(encoding="utf-8"))
    out.append({**base, "run_id": "R2_cnn_all_budgets_seeds_v01", "phase": "feature_extraction", "device": r2["device"],
                "gpu_memory_gib": gpu_gib, "determinism_settings": "eval mode, no_grad, cudnn deterministic",
                "trainable_parameters": 0, "actual_updates": 0, "actual_epochs": 0,
                "elapsed_seconds": round(float(r2["minutes_total_build"]) * 60, 1), "peak_memory_mib": "NA",
                "examples_per_second": "", "labeled_examples": "", "pseudo_candidates": "", "pseudo_accepted": "",
                "pseudo_training_examples": 0,
                "notes": f"20 A encoders (EEG/EOG x label10/label25 x seeds 0-4) on {r2['n_samples']} samples, one build for all 10 caches"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in FIELDS} for r in out])
    phases = {}
    for r in out:
        p = phases.setdefault(r["phase"], [0, 0.0])
        p[0] += 1
        p[1] += float(r["elapsed_seconds"]) if r["elapsed_seconds"] not in ("", None) else 0.0
    for ph, (n, s) in phases.items():
        print(f"{ph:26s} rows {n:4d}  total {s / 60:6.1f} min")
    print(f"Saved: {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()
