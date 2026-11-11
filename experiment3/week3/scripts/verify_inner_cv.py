"""Verify completed C1 five-fold outputs and optionally replay the B comparison."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


WEEK = Path(__file__).resolve().parents[1]
ARCHIVE = WEEK / "inner_cv_v01"
REPO = WEEK.parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, description):
    require(math.isclose(a, b, rel_tol=0, abs_tol=1e-12), f"{description} differs")


def verify_seed(seed, plan, evaluator, train_reference, dev_reference):
    folder = ARCHIVE / f"seed{seed}_v01"
    selected = load(folder / "selection.json")
    complete = load(folder / "complete.json")
    saved = load(folder / "metrics.json")
    curve = pd.read_csv(folder / "inner_cv.csv")
    require(curve["epoch"].tolist() == list(range(1, 51)), f"Seed {seed} candidate epochs differ")
    best = curve.sort_values(["oof_macro_f1", "epoch"], ascending=[False, True]).iloc[0]
    require(selected["folds"] == plan["folds"] and not selected["dev_used"] and not saved["test_used"]
            and best["epoch"] == selected["epoch"] == complete["selected_epoch"] == saved["selected_epoch"],
            f"Seed {seed} epoch selection differs")
    close(best["oof_macro_f1"], selected["oof_macro_f1"], f"Seed {seed} selected OOF Macro-F1")
    for phase, epochs in [(f"fold{i}", 50) for i in range(1, 6)] + [("refit", selected["epoch"])]:
        history = load(folder / phase / "history.json")
        require([row["epoch"] for row in history] == list(range(1, epochs + 1)),
                f"Seed {seed} {phase} training history incomplete")
    oof = evaluator.load_predictions(folder / "oof_train.csv")
    evaluator.check_against_reference(oof, train_reference)
    folds = {subject: i for i, fold in enumerate(plan["folds"]) for subject in fold}
    require(oof["subject_id"].map(folds).equals(oof["fold"]), f"Seed {seed} OOF fold differs")
    oof_metrics = evaluator.compute_metrics(oof["y_true"], oof["y_pred"])
    for metric in ("macro_f1", "n1_f1"):
        close(oof_metrics[metric], selected[f"oof_{metric}"], f"Seed {seed} OOF {metric}")
    path = folder / "predictions_val.csv"
    require(sha(path) == complete["predictions_sha256"], f"Seed {seed} predictions changed")
    dev = evaluator.load_predictions(path)
    require(evaluator.check_against_reference(dev, dev_reference) == saved["sample_set_sha256"],
            f"Seed {seed} dev coverage differs")
    metrics = evaluator.compute_metrics(dev["y_true"], dev["y_pred"])
    require(metrics["confusion_matrix"] == saved["confusion_matrix"], f"Seed {seed} confusion matrix differs")
    for metric in ("macro_f1", "n1_f1", "accuracy"):
        close(metrics[metric], saved[metric], f"Seed {seed} dev {metric}")
    return dev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-archive", type=Path, help="B Release w3_b_runs_v01.zip")
    args = parser.parse_args()
    for name, digest in load(ARCHIVE / "artifact_hashes.json").items():
        require(sha(ARCHIVE / name) == digest, f"Archive hash differs for {name}")
    plan = load(ARCHIVE / "plan.json")
    require(sha(WEEK / "scripts/run_week3_c_inner_cv.py") == plan["cv_runner_sha256"],
            "C1 runner hash differs")
    spec = importlib.util.spec_from_file_location("team_evaluate", REPO / "src/evaluation/evaluate.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    dev_reference = pd.read_csv(REPO / "manifests/val_reference_v1.csv")
    sample_index = WEEK / "deliveries/B_week3_features/eeg_mean_t1_cond14_label25/sample_index.csv"
    samples = pd.read_csv(sample_index)
    train_reference = samples.loc[samples["split"] == "train"].copy()
    subjects = sorted(train_reference["subject_id"].unique())
    order = np.random.default_rng(0).permutation(subjects)
    require(plan["folds"] == [sorted(fold.tolist()) for fold in np.array_split(order, 5)],
            "Fold assignment differs")
    require(len(train_reference) == 32310 and len(dev_reference) == 21186
            and set(subjects).isdisjoint(dev_reference["subject_id"]), "Train/dev scope differs")
    dev = {seed: verify_seed(seed, plan, evaluator, train_reference, dev_reference) for seed in (0, 1, 2)}
    print("Verified 3 C1 seeds, 15 fold histories, selected epochs, OOF coverage, and dev metrics.")
    if args.b_archive:
        provenance = load(ARCHIVE / "b_c0_reference.json")
        require(sha(args.b_archive) == provenance["sha256"], "B Release archive hash differs")
        with ZipFile(args.b_archive) as bundle:
            for seed, c1 in dev.items():
                row = load(ARCHIVE / f"seed{seed}_v01/paired_vs_b_c0.json")
                name = row["b_run"]
                prediction = bundle.read(f"{name}/predictions_val.csv")
                require(hashlib.sha256(prediction).hexdigest() ==
                        provenance["runs"][name]["predictions_sha256"], f"B seed {seed} digest differs")
                with bundle.open(f"{name}/predictions_val.csv") as stream:
                    c0 = evaluator.load_predictions(stream)
                evaluator.check_against_reference(c0, dev_reference)
                paired = evaluator.paired_bootstrap_delta(c1, c0, n_boot=2000, seed=0)
                for metric in ("macro_f1", "n1_f1", "accuracy"):
                    for key in ("point", "lo", "hi"):
                        close(paired[metric][key], row["paired_C1_minus_B_C0"][metric][key],
                              f"Seed {seed} paired {metric} {key}")
        print("Verified 3 paired subject-bootstrap comparisons against the released B C0 predictions.")


if __name__ == "__main__":
    main()
