"""Recompute the published results from archived subject-level confusion counts."""

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from confirm import summarize as summarize_external
from run import sha256, summarize as summarize_development


HERE = Path(__file__).resolve().parent
DEVELOPMENT = HERE.parent / "week3_budget_replication_20261128"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_value(expected, actual, field):
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"Different keys in {field}")
        for key in expected:
            verify_value(expected[key], actual[key], f"{field}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(f"Different length in {field}")
        for i, (want, got) in enumerate(zip(expected, actual)):
            verify_value(want, got, f"{field}[{i}]")
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{field}: expected {expected}, got {actual}")
    elif expected != actual:
        raise ValueError(f"{field}: expected {expected}, got {actual}")


def main():
    development_results = DEVELOPMENT / "results"
    external_results = HERE / "results"
    plan = read_json(development_results / "plan.json")
    data_lock = read_json(external_results / "data_lock.json")
    locks = {development_results / name: digest for name, digest in plan["locked_files"].items()}
    locks.update({
        DEVELOPMENT / "run.py": plan["source_sha256"]["run.py"],
        HERE / "confirm.py": data_lock["code_sha256"],
        external_results / "blocks.csv": data_lock["blocks_sha256"],
    })
    for path, expected in locks.items():
        if sha256(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")

    with np.load(development_results / "confusion_counts.npz", allow_pickle=False) as archive:
        development_counts = archive["counts"]
    external_counts = np.stack(
        [np.load(external_results / f"fold{fold}_confusions.npy", allow_pickle=False) for fold in range(1, 5)]
    )
    with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
        working = Path(temp)
        summarize_development(development_counts, plan, working)
        development = read_json(working / "summary.json")
        summarize_external(external_counts, working)
        external = read_json(working / "summary.json")
    verify_value(read_json(development_results / "summary.json"), development, "Sleep-EDF")
    verify_value(read_json(external_results / "summary.json"), external, "Ying")
    print("Verified Sleep-EDF and Ying checksums, estimates, intervals, and decisions.")


if __name__ == "__main__":
    main()
