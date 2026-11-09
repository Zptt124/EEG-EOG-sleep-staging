"""Recompute the archived Week 3 C scores using the team evaluator."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pandas as pd


ARCHIVE = Path(__file__).resolve().parents[1]
REPO = ARCHIVE.parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hash(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 differs for {path}")


def check_value(observed, expected, name):
    if not math.isclose(observed, float(expected), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{name} differs: {observed} vs {expected}")


def main():
    spec = importlib.util.spec_from_file_location("team_evaluate", REPO / "src/evaluation/evaluate.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    reference = pd.read_csv(REPO / "manifests/val_reference_v1.csv")
    publication = read_json(ARCHIVE / "publication.json")
    for name, expected in publication["source_and_json_sha256"].items():
        verify_hash(ARCHIVE / name, expected)
    with (ARCHIVE / "run_registry.csv").open(encoding="utf-8", newline="") as stream:
        registry = list(csv.DictReader(stream))
    if len(registry) != 36:
        raise ValueError("Expected 36 prediction tables")
    pairs = {}
    for row in registry:
        path = ARCHIVE / row["prediction_path"]
        verify_hash(path, row["prediction_sha256"])
        frame = evaluator.load_predictions(path)
        evaluator.check_against_reference(frame, reference)
        metrics = evaluator.compute_metrics(frame["y_true"], frame["y_pred"])
        saved = read_json(path.with_name("metrics.json"))
        if metrics["confusion_matrix"] != saved["confusion_matrix"] or row["test_used"] != "False":
            raise ValueError(f"Run evidence differs for {path}")
        for key in ("macro_f1", "n1_f1", "accuracy", "n1_precision", "n1_recall"):
            check_value(metrics[key], saved[key], f"{path} {key}")
            if key in row:
                check_value(metrics[key], row[key], f"{path} registry {key}")
        check_value(metrics["per_class_f1"]["REM"], row["rem_f1"], f"{path} REM F1")
        run = path.parent.parent
        frames = pairs.setdefault(run, {})
        if row["arm"] in frames:
            raise ValueError(f"Duplicate arm for {run}")
        frames[row["arm"]] = frame
    if len(pairs) != 18 or any(set(frames) != {"C0", "C1"} for frames in pairs.values()):
        raise ValueError("Expected 18 C0/C1 pairs")
    for run, frames in pairs.items():
        archived = read_json(run / "paired_results.json")["paired_C1_minus_C0"]
        paired = evaluator.paired_bootstrap_delta(frames["C1"], frames["C0"],
                                                n_boot=archived["n_boot"], seed=archived["seed"])
        for metric in ("macro_f1", "n1_f1", "accuracy"):
            for key in ("point", "lo", "hi"):
                check_value(paired[metric][key], archived[metric][key], f"{run} {metric} {key}")
    print("Verified 36 prediction tables, reference coverage, point metrics, and 18 paired intervals.")


if __name__ == "__main__":
    main()
