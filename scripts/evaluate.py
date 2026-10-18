"""evaluate.py — PROVISIONAL local copy (B owns the official group script).

Interface per the team doc: input = a per-epoch prediction table
(sample_id, subject_id, record_id, epoch_index, y_true, y_pred, p_W..p_REM),
output = Macro-F1, per-class F1, N1 precision/recall, confusion matrix.

Usage:
  python evaluate.py runs/E1_EEG_label100_seed0_v01/predictions_val.csv
  python evaluate.py runs/E1_EEG_label100_seed0_v01/predictions_val.csv --out metrics.json
Replace with B's official evaluate.py once delivered; results must match.
"""
import argparse
import json
import sys

import pandas as pd

from metrics import compute_metrics, metrics_table, save_metrics

from common import STAGE_NAMES

PROB_COLS = ["p_W", "p_N1", "p_N2", "p_N3", "p_REM"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions_csv)
    required = ["sample_id", "subject_id", "record_id", "epoch_index", "y_true", "y_pred"] + PROB_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"missing columns: {missing}")
    if not df["sample_id"].is_unique:
        sys.exit(f"sample_id not unique: {df['sample_id'].duplicated().sum()} duplicates")
    psum = df[PROB_COLS].sum(axis=1)
    if (psum < 0.99).any() or (psum > 1.01).any():
        sys.exit(f"probability rows not summing to ~1: {psum.min():.4f}..{psum.max():.4f}")
    argmax_pred = df[PROB_COLS].values.argmax(1)
    if not (argmax_pred == df["y_pred"].values).all():
        sys.exit("y_pred must equal argmax of (p_W..p_REM) on every row")

    m = compute_metrics(df["y_true"].values, df["y_pred"].values, STAGE_NAMES)
    m["n_epochs"] = len(df)
    m["source"] = args.predictions_csv
    print(metrics_table(m, STAGE_NAMES))
    if args.out:
        save_metrics(m, args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
