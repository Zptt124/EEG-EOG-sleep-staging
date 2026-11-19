"""Verify C's six crossmodal predictions and the cited D three-seed tables."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import yaml


REPO = Path(__file__).resolve().parents[3]
DELIVERY = REPO / "experiment3/week4/crossmodal_final_v01"


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for relative, expected in load(DELIVERY / "artifact_sha256.json").items():
        assert sha(REPO / relative) == expected, relative
    spec = importlib.util.spec_from_file_location("team_evaluator", REPO / "src/evaluation/evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    reference = pd.read_csv(REPO / "manifests/val_reference_v1.csv")
    registry = pd.read_csv(REPO / "reports/week4/c_crossmodal_registry_w4.csv")
    assert len(registry) == 6 and registry.run_id.is_unique
    assert set(zip(registry.arm, registry.budget, registry.seed)) == {
        (arm, budget, 0) for arm in ("xm_sync", "xm_shift", "xm_none")
        for budget in ("label10", "label25")
    }
    identity = None
    for record in registry.to_dict("records"):
        path = REPO / record["predictions"]
        frame = ev.load_predictions(path)
        ev.check_against_reference(frame, reference)
        assert len(frame) == 21186 and frame.subject_id.nunique() == 8
        metadata = frame[["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]]
        assert identity is None or metadata.equals(identity)
        identity = metadata
        config = yaml.safe_load((path.parent / "config.yaml").read_text(encoding="utf8"))
        metric = load(path.parent / "metrics.json")
        assert config["run_id"] == record["run_id"]
        assert config["modalities"] == ["eog"] and not config["test_used"]
        assert config["downstream_seed"] == 0 and config["feature_dims"]["eog"] == 768
        assert sha(path) == config["predictions_sha256"]
        raw = ev.compute_metrics(frame.y_true, frame.y_pred)
        for key in ("macro_f1", "n1_f1"):
            np.testing.assert_allclose(raw[key], metric[key], atol=1e-12)
        np.testing.assert_allclose(average_precision_score(frame.y_true == 1, frame.p_N1),
                                   metric["n1_ap"], atol=1e-12)
        probability = frame[ev.PROB_COLS].to_numpy()
        np.testing.assert_allclose(probability.sum(1), 1, atol=2e-6)
        adjusted = pd.read_csv(path.parent / "predictions_val_n1adj.csv")
        probability[:, 1] *= metric["n1_adjust"]["alpha"]
        assert np.array_equal(adjusted.y_pred, probability.argmax(1))
        after = ev.compute_metrics(frame.y_true, adjusted.y_pred)
        for key in ("macro_f1", "n1_f1"):
            np.testing.assert_allclose(after[key], metric["n1_adjust"][key + "_adjusted"], atol=1e-12)
        curve = pd.read_csv(path.parent / "inner_cv.csv")
        selected = int(curve.sort_values(["oof_macro_f1", "epoch"], ascending=[False, True]).iloc[0].epoch)
        assert len(curve) == 50 and selected == config["selected_epoch"]
        oof = pd.read_csv(path.parent / "oof_train.csv")
        assert oof.subject_id.nunique() == config["labeled_subjects"]
        assert len(oof) == config["train_samples"]
        assert set(oof.subject_id).isdisjoint(frame.subject_id)
        probabilities = oof[ev.PROB_COLS].to_numpy()
        alphas = np.logspace(-1, 1.5, 801)
        scores = []
        for alpha in alphas:
            candidate = probabilities.copy()
            candidate[:, 1] *= alpha
            scores.append(ev.compute_metrics(oof.y_true, candidate.argmax(1))["n1_f1"])
        scores = np.array(scores)
        candidates = np.flatnonzero(scores == scores.max())
        chosen = candidates[np.argmin(np.abs(np.log(alphas[candidates])))]
        np.testing.assert_allclose(alphas[chosen], metric["n1_adjust"]["alpha"], atol=1e-12)
        print("Verified", record["run_id"], flush=True)
    d_source = DELIVERY / "external/d_three_head_seed"
    for item in load(d_source / "source_receipt.json"):
        assert sha(d_source / item["file"]) == item["sha256"], item["file"]
    d_pairs = pd.read_csv(d_source / "d_paired.csv")
    assert len(d_pairs) == 6 and (d_pairs.seeds == "0 1 2").all()
    row = d_pairs[(d_pairs.comparison == "a_same030k_eog")
                  & (d_pairs.reference == "d_pre030k_eog") & (d_pairs.budget == "label10")].iloc[0]
    assert row.delta_macro_f1_hi < 0
    assert ((d_pairs.delta_n1_ap_lo <= 0) & (d_pairs.delta_n1_ap_hi >= 0)).all()
    np.testing.assert_allclose(row.delta_macro_f1, -0.00629519, atol=1e-10)
    all_runs = pd.read_csv(REPO / "reports/week4/c_eval_registry_w4_all.csv")
    assert len(all_runs) == 18 and all_runs.run_id.is_unique
    assert all((REPO / name).is_file() for name in all_runs.predictions)
    print("Verified D source tables and all eighteen C prediction paths", flush=True)


if __name__ == "__main__":
    main()
