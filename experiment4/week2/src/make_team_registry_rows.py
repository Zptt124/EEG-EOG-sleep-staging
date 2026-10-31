"""make_team_registry_rows.py - Week-2 Experiment 4 (D): rows for the team's shared registries.

Builds D's rows for the three shared tables in reports/week2/ (same headers as A's and B's rows):
  run_registry.csv   one row per student run (60)
  artifact_index.csv one row per file in D's Release asset, plus one row for the asset itself
  resource_profile.csv  D's rows (from analysis/resource_profile_D.csv, already in the team format)
Before writing, every predictions_val.csv is re-checked with the team's current evaluate.py and
--reference manifests/val_reference_v1.csv (same check as B's runs); the recomputed macro-F1 must equal
the value in run_index.csv.

Step 1 (generate):  --exp-dir --delivery --team-repo --cache-root   -> <delivery>/team_rows/*.csv
Step 2 (append):    add --append-to <team repo checkout> to append the generated rows to
                    <repo>/reports/week2/{run_registry,artifact_index,resource_profile}.csv
                    (headers must match exactly; run_id / artifact_id must not already exist).
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml

RELEASE_TAG = "d-ssl-w2-v01-20261101"
ZIP_NAME = "w2_d_artifacts_v01.zip"
REPO_PREFIX = "experiment4/week2"
TABLES = {"run_registry": "run_registry_D.csv", "artifact_index": "artifact_index_D_team.csv",
          "resource_profile": "resource_profile_D.csv"}
KEY = {"run_registry": "run_id", "artifact_index": "artifact_id", "resource_profile": "run_id"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def header(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def write(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "NA") for k in fields} for r in rows])


def generate(args):
    exp, team = args.exp_dir, args.team_repo
    out = args.delivery / "team_rows"
    if out.exists():
        fail(f"{out} already exists")
    out.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("team_evaluate", team / "src" / "evaluation" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    ref = team / "manifests" / "val_reference_v1.csv"
    subset_sha = sha256(team / "manifests" / "label_subsets_v2.csv")
    ref_sha = sha256(ref)

    # ---------- run_registry rows (with --reference re-check) ----------
    reg_fields = header(team / "reports" / "week2" / "run_registry.csv")
    runs = read_csv(exp / "runs" / "run_index.csv")
    teachers = {(t["repr"], t["budget"], t["seed"], t["modality"]): t["teacher_id"] for t in read_csv(exp / "runs" / "teacher_index.csv")}
    uri = f"https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/{RELEASE_TAG}/{ZIP_NAME}#/"
    reg_rows, checks = [], []
    for r in runs:
        rid, rep, b, s, m = r["run_id"], r["repr"], r["budget"], r["seed"], r["method"]
        rdir = exp / "runs" / rid
        met = ev.evaluate(rdir / "predictions_val.csv", reference=str(ref))
        if abs(met["macro_f1"] - float(r["macro_f1"])) > 5e-5:
            fail(f"{rid}: macro-F1 with team evaluate.py {met['macro_f1']:.6f} != run_index {r['macro_f1']}")
        if met.get("sample_set_sha256") in (None, ""):
            fail(f"{rid}: team evaluate.py did not return sample_set_sha256")
        checks.append({"run_id": rid, "macro_f1_team_evaluate": round(met["macro_f1"], 6), "macro_f1_run_index": r["macro_f1"],
                       "sample_set_sha256": met.get("sample_set_sha256", ""), "reference": "manifests/val_reference_v1.csv",
                       "evaluate_version": met.get("evaluate_version", "")})
        labeled = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))["labeled_subjects"]
        if len(labeled) != {"label10": 5, "label25": 14}[b]:
            fail(f"{rid}: {len(labeled)} labeled subjects in config.yaml")
        if rep == "R1":
            man = json.loads((args.cache_root / "R1_eegdm_cond14_pooled_v01" / "cache_manifest.json").read_text(encoding="utf-8"))
            representation, enc_sha = "eegdm_raw_frozen_eegeog_5120", "NA"
            feature_version = f"R1_eegdm_cond14_pooled_v01 eeg:{man['files_sha256']['eeg_pooled.npy']};eog:{man['files_sha256']['eog_pooled.npy']}"
        else:
            man = json.loads((args.cache_root / f"R2_cnn_{b}_seed{s}_v01" / "cache_manifest.json").read_text(encoding="utf-8"))
            representation = f"cnn_single_modality_eeg_eog_{b}_encoders_256"
            enc_sha = f"eeg:{man['encoders']['eeg']['encoder_weights.pt']};eog:{man['encoders']['eog']['encoder_weights.pt']}"
            feature_version = f"R2_cnn_{b}_seed{s}_v01 eeg:{man['files_sha256']['eeg_feat.npy']};eog:{man['files_sha256']['eog_feat.npy']}"
        if m == "D0":
            teacher_id, teacher_seed, n_unl = "NA", "NA", 0
        else:
            ids = [teachers[(rep, b, s, "dual")]] + ([teachers[(rep, b, s, "eeg")], teachers[(rep, b, s, "eog")]] if m == "D2" else [])
            teacher_id, teacher_seed, n_unl = ";".join(ids), s, {"label10": 49, "label25": 40}[b]
        reg_rows.append({
            "run_id": rid, "owner": "D", "status": "completed",
            "experiment": {"D0": "W2 Exp4 semi-supervised: D0 supervised start (true labels only)",
                           "D1": "W2 Exp4 semi-supervised: D1 self-training (per-class top-p pseudo labels)",
                           "D2": "W2 Exp4 semi-supervised: D2 cross-modal self-training (EEG and EOG teachers agree)"}[m],
            "representation": representation, "modalities": "EEG+EOG", "context_length": 1, "nominal_budget": b,
            "labeled_subjects": ";".join(labeled), "unlabeled_signal_subjects": n_unl, "subset_version": "v2",
            "subset_sha256": subset_sha, "sampling_seed": 2024, "adaptation_version": "NA", "adaptation_seed": "NA",
            "teacher_id": teacher_id, "teacher_seed": teacher_seed, "downstream_seed": s,
            "code_commit": "NA (per-file code sha256 in config.yaml; experiment4/week2/code_sha_map.csv)",
            "config_path": f"{REPO_PREFIX}/runs/{rid}/config.yaml",
            "normalization_path": f"{uri}runs/{rid}/best.ckpt (standardizer stored in the checkpoint)",
            "backbone_sha256": "NA", "encoder_sha256": enc_sha, "feature_version": feature_version,
            "sample_set_path": "manifests/val_reference_v1.csv", "sample_set_sha256": met["sample_set_sha256"],
            "artifact_index_path": "reports/week2/artifact_index.csv", "run_storage_uri": f"{REPO_PREFIX}/runs/{rid}",
            "reused_from_run_id": "NA",
            "reuse_check_path": f"{REPO_PREFIX}/manifests/roles_v1_vs_v2_note.md" if rep == "R1" else "NA",
            "evaluation_status": "evaluate_v1_with_reference_passed",
            "last_successful_step": "predictions_val.csv evaluated", "failure_reason": "NA"})
    if len(reg_rows) != 60:
        fail(f"expected 60 runs, got {len(reg_rows)}")
    write(out / TABLES["run_registry"], reg_fields, reg_rows)
    write(out / "reference_check_D.csv", list(checks[0].keys()), checks)

    # ---------- artifact_index rows ----------
    art_fields = header(team / "reports" / "week2" / "artifact_index.csv")
    repo_part = args.delivery / "repo" / REPO_PREFIX
    info = dict(line.split(": ", 1) for line in (repo_part / "release_info.txt").read_text(encoding="utf-8").splitlines() if ": " in line)
    art_rows = [{"artifact_id": ZIP_NAME, "artifact_type": "run_artifacts_bundle", "owner": "D", "source_run_id": "NA",
                 "adaptation_version": "NA", "adaptation_seed": "NA", "teacher_id": "NA",
                 "storage_uri": f"https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/{RELEASE_TAG}/{ZIP_NAME}",
                 "sha256": info["asset_sha256"], "size_bytes": info["asset_size_bytes"], "code_commit": "NA", "config_path": "NA",
                 "index_path": f"{REPO_PREFIX}/artifact_index_D.csv", "normalization_path": "NA",
                 "notes": "D week-2: 60 student best.ckpt, 40 pseudo_label_trace.csv, 60 teacher heads + teacher val predictions, original code"}]
    type_map = {"student_weights": "head_weights", "teacher_weights": "teacher_weights",
                "pseudo_label_trace": "pseudo_label_trace", "teacher_predictions_val": "teacher_predictions_val",
                "original_code": "code_original"}
    for a in read_csv(repo_part / "artifact_index_D.csv"):
        parts = a["artifact_path"].split("/")
        src_run = parts[1] if parts[0] == "runs" and parts[1] != "teachers" else "NA"
        teacher = parts[2] if len(parts) > 2 and parts[1] == "teachers" else "NA"
        art_rows.append({"artifact_id": "D_" + a["artifact_path"].replace("/", "__"), "artifact_type": type_map[a["artifact_type"]],
                         "owner": "D", "source_run_id": src_run, "adaptation_version": "NA", "adaptation_seed": "NA",
                         "teacher_id": teacher, "storage_uri": a["storage_uri"], "sha256": a["sha256"], "size_bytes": a["size_bytes"],
                         "code_commit": "NA", "config_path": f"{REPO_PREFIX}/runs/{src_run}/config.yaml" if src_run != "NA" else "NA",
                         "index_path": f"{REPO_PREFIX}/artifact_index_D.csv",
                         "normalization_path": "inside best.ckpt" if a["artifact_type"] == "student_weights" else "NA", "notes": ""})
    write(out / TABLES["artifact_index"], art_fields, art_rows)

    # ---------- resource_profile rows ----------
    res_fields = header(team / "reports" / "week2" / "resource_profile.csv")
    res_rows = read_csv(exp / "analysis" / "resource_profile_D.csv")
    if header(exp / "analysis" / "resource_profile_D.csv") != res_fields:
        fail("resource_profile_D.csv header differs from the team table")
    write(out / TABLES["resource_profile"], res_fields, res_rows)

    for t, n in TABLES.items():
        print(f"{t:17s} {len(read_csv(out / n)):4d} rows -> {out / n}")
    print(f"OK: all 60 predictions passed team evaluate.py --reference; macro-F1 equal to run_index.csv")


def append(args):
    out = args.delivery / "team_rows"
    for t, n in TABLES.items():
        target = args.append_to / "reports" / "week2" / f"{t}.csv"
        if header(target) != header(out / n):
            fail(f"header of {target} differs from {out / n}")
        existing = {r[KEY[t]] for r in read_csv(target)}
        new = read_csv(out / n)
        dup = [r[KEY[t]] for r in new if r[KEY[t]] in existing]
        if dup:
            fail(f"{target}: {len(dup)} rows already present, e.g. {dup[:2]}")
        raw = target.read_bytes()
        with open(target, "a", encoding="utf-8", newline="") as f:
            if raw and not raw.endswith(b"\n"):
                f.write("\n")
            w = csv.DictWriter(f, fieldnames=header(target), lineterminator="\r\n" if b"\r\n" in raw[:4096] else "\n")
            w.writerows(new)
        print(f"appended {len(new)} rows to {target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--delivery", type=Path, required=True)
    ap.add_argument("--team-repo", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--append-to", type=Path, help="team repo checkout to append the generated rows to")
    args = ap.parse_args()
    if args.append_to:
        append(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
