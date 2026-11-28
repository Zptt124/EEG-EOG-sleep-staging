"""Recompute the archived BiGRU and budget-extension results from confusion counts."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

from extend_budget import analyze
from run import ARMS, macro, paired, sha256, summarize

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "week4_budget_external_ying"))
from verify_results import read_json, verify_value

DEVELOPMENT = HERE.parent / "week3_budget_replication_20261128" / "results"
BIGRU = HERE.parent / "week5_budget_bigru_20261128" / "results"
EXTENSION = HERE / "results"


def verify_bigru(plan, working):
    protocol = read_json(BIGRU / "protocol.json")
    with np.load(BIGRU / "confusion_counts.npz", allow_pickle=False) as archive:
        counts = archive["counts"]
    summarize(counts, plan, working)
    measured = read_json(working / "summary.json")
    primary = protocol["primary"]
    contrast = measured["combined"][primary]
    lower, upper = contrast["interval_95_pp"]
    if upper < 1:
        verdict = "below_practical_target"
    elif lower > 0 and contrast["mean_pp"] >= 1 and contrast["low_overlap_mean_pp"] > 0:
        verdict = "advance_to_external_bigru"
    else:
        verdict = "inconclusive"
    seeded = paired(macro(counts[:, :3].sum(axis=-2)))
    measured.update(
        primary=primary,
        decision=verdict,
        arm_f1=dict(zip(ARMS, macro(counts[0].sum(axis=-2)).mean(axis=(0, 1)).tolist())),
        seed_check={"first_three_rosters_new_minus_deepen_pp": seeded[..., 2].mean(axis=(1, 2)).tolist()},
    )
    expected = read_json(BIGRU / "summary.json")
    expected.pop("elapsed_seconds")
    verify_value(expected, measured, "BiGRU")


def verify_extension(working):
    protocol = read_json(EXTENSION / "protocol.json")
    for path, expected in (
        (DEVELOPMENT / "plan.json", protocol["locked_inputs_sha256"]["plan.json"]),
        (HERE / "extend_budget.py", protocol["code_sha256"]["extend_budget.py"]),
        (EXTENSION / "extended_blocks.csv", protocol["extended_blocks_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"SHA-256 mismatch for {path}")
    with np.load(DEVELOPMENT / "confusion_counts.npz", allow_pickle=False) as low:
        low_counts = low["counts"][0, np.asarray(protocol["rosters"]) - 1]
    with np.load(EXTENSION / "confusion_counts.npz", allow_pickle=False) as high:
        if high["rosters"].tolist() != protocol["rosters"]:
            raise ValueError("Higher-budget roster order does not match the protocol")
        counts = np.stack((low_counts, high["counts"]))
    analyze(counts, protocol, working, lambda: 0.0)
    measured = read_json(working / "summary.json")
    expected = read_json(EXTENSION / "summary.json")
    measured.pop("elapsed_seconds")
    expected.pop("elapsed_seconds")
    verify_value(expected, measured, "Budget extension")


def main():
    plan = read_json(DEVELOPMENT / "plan.json")
    with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
        working = Path(directory)
        verify_bigru(plan, working)
        verify_extension(working)
    print("Verified BiGRU and budget-extension estimates and paired intervals.")


if __name__ == "__main__":
    main()
