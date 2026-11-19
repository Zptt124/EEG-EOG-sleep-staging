"""Verify C W4's twelve runs from the repository root, without training."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
DELIVERY = REPO / "experiment3/week4/fourcell_v01"


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-archive", type=Path)
    args = parser.parse_args()
    for relative, expected in load(DELIVERY / "artifact_sha256.json").items():
        if digest(REPO / relative) != expected:
            raise ValueError(f"Published file hash differs: {relative}")
    spec = importlib.util.spec_from_file_location("team_evaluator", REPO / "src/evaluation/evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    reference = pd.read_csv(REPO / "manifests/val_reference_v1.csv")
    table = pd.read_csv(DELIVERY / "metrics_summary.csv").set_index("run")
    registry = pd.read_csv(REPO / "reports/week4/c_eval_registry_w4.csv")
    assert len(registry) == 12 and registry.run_id.is_unique
    assert set(zip(registry.arm, registry.seed)) == {
        (f"{head}_{arm}", seed) for head in ("linear", "lft")
        for arm in ("random_init", "pretrained") for seed in (0, 1, 2)
    }
    lock = load(DELIVERY / "configuration_lock.json")
    assert lock["pooling_seconds"] == 1 and lock["diffusion_t"] == 1
    assert not lock["test_used"]
    alphas = np.logspace(-1, 1.5, 801)
    identity = None
    for record in registry.to_dict("records"):
        prediction_path = REPO / record["predictions"]
        name = prediction_path.parent.name
        directory = prediction_path.parent
        frame = ev.load_predictions(prediction_path)
        ev.check_against_reference(frame, reference)
        assert len(frame) == 21186 and frame.subject_id.nunique() == 8
        metadata = frame[["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]]
        assert identity is None or metadata.equals(identity)
        identity = metadata
        config, metrics = load(directory / "config.json"), load(directory / "metrics.json")
        assert config["run_id"] == record["run_id"] and not config["test_used"]
        assert config["configuration_lock_sha256"] == digest(DELIVERY / "configuration_lock.json")
        probability = frame[ev.PROB_COLS].to_numpy()
        np.testing.assert_allclose(probability.sum(1), 1, atol=2e-6)
        assert np.array_equal(probability.argmax(1), frame.y_pred)
        raw = ev.compute_metrics(frame.y_true, frame.y_pred)
        for key in ("macro_f1", "n1_f1"):
            np.testing.assert_allclose(raw[key], metrics[key], atol=1e-12)
            np.testing.assert_allclose(raw[key], table.loc[name, key], atol=1e-12)
        oof = pd.read_csv(directory / "oof_train.csv")
        assert len(oof) == 11936 and oof.subject_id.nunique() == 5
        assert set(oof.subject_id).isdisjoint(frame.subject_id)
        assert digest(directory / "oof_train.csv") == lock["alpha_selection"][name]["oof_sha256"]
        probabilities = oof[ev.PROB_COLS].to_numpy()
        scores = []
        for alpha in alphas:
            adjusted = probabilities.copy()
            adjusted[:, 1] *= alpha
            scores.append(ev.compute_metrics(oof.y_true, adjusted.argmax(1))["n1_f1"])
        scores = np.array(scores)
        candidates = np.flatnonzero(scores == scores.max())
        selected = candidates[np.argmin(np.abs(np.log(alphas[candidates])))]
        np.testing.assert_allclose(alphas[selected], config["alpha"], rtol=0, atol=1e-12)
        adjusted = probability.copy()
        adjusted[:, 1] *= config["alpha"]
        after = ev.compute_metrics(frame.y_true, adjusted.argmax(1))
        for key in ("macro_f1", "n1_f1"):
            np.testing.assert_allclose(after[key], metrics["adjusted"][key], atol=1e-12)
        curve = pd.read_csv(DELIVERY / "screening" / f"seed{record['seed']}" /
                            record["arm"] / "inner_cv.csv")
        assert len(curve) == 50
        epoch = int(curve.sort_values(["oof_macro_f1", "epoch"], ascending=[False, True]).iloc[0].epoch)
        assert epoch == config["selected_epoch"]
        print(f"Verified {record['run_id']}", flush=True)
    if args.models_archive:
        with zipfile.ZipFile(args.models_archive) as archive:
            assert archive.testzip() is None
            for model in load(DELIVERY / "model_index.json")["models"]:
                assert hashlib.sha256(archive.read(model["archive_path"])).hexdigest() == model["sha256"]
        print("Verified all twelve final models", flush=True)
    print("All twelve W4 C runs passed", flush=True)


if __name__ == "__main__":
    main()
