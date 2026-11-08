"""run_fold_cnns.py - Week-2 Experiment 4 (D): retrain A's SmallCNN encoders for R2 internal calibration.

For each budget and each labeled subject h (leave-one-labeled-subject-out), train EEG-only and EOG-only
SmallCNNs with A's unified entry point (scripts/train_cnn_w2.py --subjects <labeled minus h>), seed 0.
These fold encoders never see the held-out subject's labels, as required by protocol v2 rule 4.
Early stopping inside train_cnn_w2.py uses the external 8-subject validation set (A's interface; the same
rule as A's own encoders), so the validation set is used only for epoch selection, never to choose p.

Resumable: a fold run whose metrics.json already exists is skipped. A run folder without metrics.json
(a crashed run) stops the script so it can be inspected and moved away.

Writes <runs-dir>/fold_index.csv: budget, fold, heldout_subject, modality, run_id, train_subjects, status.

Usage (Ubuntu):
  python -u run_fold_cnns.py --roles labeled_unlabeled_subjects.csv --budgets label10 label25
      --a-repo <team_repo> --data-dir <A_processed_npz>
      --runs-dir <D_local_r2_fold_cnns>
"""

import argparse
import csv
import hashlib
import subprocess
import sys
import time
from pathlib import Path

TAG = {"eeg": "EEG", "eog": "EOG"}
SEED = 0


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def expected_run_id(modality, subjects):
    # Same naming rule as train_cnn_w2.py for --subjects runs
    h = hashlib.sha256(",".join(sorted(subjects)).encode()).hexdigest()[:8]
    return f"W2_A_{TAG[modality]}_custom{h}_sseed2024_seed{SEED}_v01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", type=Path, required=True)
    ap.add_argument("--budgets", nargs="+", choices=["label10", "label25"], required=True)
    ap.add_argument("--a-repo", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    args = ap.parse_args()

    trainer = args.a_repo / "scripts" / "train_cnn_w2.py"
    if not trainer.exists():
        print(f"ERROR: {trainer} not found")
        sys.exit(1)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    roles = read_csv(args.roles)

    jobs = []
    for budget in args.budgets:
        labeled = sorted(r["subject_id"] for r in roles if r[f"role_{budget}"] == "labeled")
        if len(labeled) != {"label10": 5, "label25": 14}[budget]:
            print(f"ERROR: {budget} has {len(labeled)} labeled subjects in {args.roles}")
            sys.exit(1)
        for fold, held in enumerate(labeled):
            train_subjects = [s for s in labeled if s != held]
            for modality in ("eeg", "eog"):
                jobs.append((budget, fold, held, modality, train_subjects))
    print(f"{len(jobs)} fold encoders to train ({', '.join(args.budgets)}), seed {SEED}", flush=True)

    index_rows = []
    t_all = time.time()
    for k, (budget, fold, held, modality, train_subjects) in enumerate(jobs, start=1):
        run_id = expected_run_id(modality, train_subjects)
        run_dir = args.runs_dir / run_id
        row = {"budget": budget, "fold": fold, "heldout_subject": held, "modality": modality, "run_id": run_id,
               "train_subjects": ";".join(train_subjects)}
        if (run_dir / "metrics.json").exists():
            print(f"[{k}/{len(jobs)}] skip (already done): {budget} fold {fold} held={held} {modality} {run_id}", flush=True)
            index_rows.append({**row, "status": "done_before"})
            continue
        if run_dir.exists():
            print(f"ERROR: {run_dir} exists without metrics.json (crashed run?). Move it away and rerun.")
            sys.exit(1)
        print(f"\n[{k}/{len(jobs)}] train {budget} fold {fold} held-out={held} {modality} -> {run_id}", flush=True)
        cmd = [sys.executable, "-u", str(trainer), "--modality", modality, "--budget", budget, "--seed", str(SEED),
               "--subjects", ",".join(train_subjects), "--owner", "D",
               "--data-dir", str(args.data_dir), "--runs-dir", str(args.runs_dir)]
        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(args.a_repo / "scripts"))
        if res.returncode != 0 or not (run_dir / "metrics.json").exists():
            print(f"ERROR: training failed for {run_id} (return code {res.returncode})")
            sys.exit(1)
        print(f"[{k}/{len(jobs)}] done in {time.time() - t0:.0f}s, elapsed {(time.time() - t_all) / 60:.1f} min", flush=True)
        index_rows.append({**row, "status": "trained"})

    with (args.runs_dir / "fold_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)
    print(f"\nAll {len(jobs)} fold encoders ready in {(time.time() - t_all) / 60:.1f} min. Saved {args.runs_dir / 'fold_index.csv'}")


if __name__ == "__main__":
    main()
