"""Run paired linear/LFT five-fold screening without opening development data."""

import argparse
import fcntl
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score
import torch

import run_week3_c_readout as c
from run_week3_c_inner_cv import fit_phase, write_selection


def stage_arm(cache, arm, records, pooling, output):
    sources = {modality: output / modality for modality in c.MODALITIES}
    for directory in sources.values():
        directory.mkdir(parents=True, exist_ok=True)
    expected = (12, 30 // pooling, 64)
    rows, pooled = [], []
    for record, subject in records:
        with np.load(cache / arm / f"{record}.npz", allow_pickle=False) as data:
            labels, epochs = data["y_true"], data["epoch_index"]
            c.require(str(data["subject_id"]) == subject and len(set(epochs)) == len(labels),
                      f"Invalid epoch metadata for {record}")
            features = []
            for modality, directory in sources.items():
                array = data[f"{modality.lower()}_{pooling}s_std"]
                c.require(array.shape == (len(labels), *expected) and np.isfinite(array).all(),
                          f"Invalid {modality} features for {record}")
                path = directory / f"{record}.npy"
                if not path.is_file():
                    temporary = path.with_suffix(".partial")
                    with temporary.open("wb") as stream:
                        np.save(stream, array)
                    temporary.replace(path)
                features.append(array.mean(axis=2, dtype=np.float32).reshape(len(labels), -1))
            pooled.append(np.concatenate(features, axis=1))
        for index, (epoch, label) in enumerate(zip(epochs, labels)):
            rows.append(dict(sample_id=f"{record}_{int(epoch):04d}", subject_id=subject,
                             record_id=record, epoch_index=int(epoch), y_true=int(label),
                             row_index=index))
    return rows, np.concatenate(pooled), sources


def run(args):
    cache_plan = json.loads((args.cache / "plan.json").read_text(encoding="utf8"))
    c.require((args.cache / "DONE").is_file() and cache_plan["mode"] == "screening"
              and not cache_plan["dev_used"] and not cache_plan["test_used"],
              "Expected complete training-only feature cache")
    b_path = args.b_checkout.resolve() / "scripts/27_b_w3_linear_probe.py"
    sys.path.insert(0, str(b_path.parent))
    b = runpy.run_path(str(b_path))
    evaluator = b["ev"]
    subjects = sorted({subject for _, subject in cache_plan["records"]})
    roster, _ = b["w"].selected_subjects(args.b_checkout / "manifests/label_subsets_v2.csv",
                                          "label10")
    c.require(len(subjects) == 5 and set(subjects) == set(roster)
              and cache_plan["checkpoint_steps"] == {"random_init": 0, "pretrained": 30000}
              and cache_plan["diffusion_t"] == 1 and b["MAX_EPOCHS"] == 50
              and b["FOLD_SEED"] == 0, "Unexpected screening protocol")
    folds = b["subject_folds"](subjects, len(subjects))
    config = dict(layers=list(range(12)), feature_hidden=64, feature_positions=30 // args.pooling,
                  lft_tokens_per_second=4, batch_size=32, blocks_per_group=8,
                  learning_rate=0.0005, weight_decay=0.0001, gradient_clip_norm=3.0)
    plan = dict(cache=cache_plan, pooling=args.pooling, head_seed=args.seed, folds=folds,
                epochs=b["MAX_EPOCHS"], lft=config, linear="W3 B linear probe",
                linear_probe_sha256=c.sha(b_path), evaluator_sha256=c.sha(
                    args.b_checkout / "src/evaluation/evaluate.py"),
                cache_sha256={
                    f"{arm}/{record}": c.sha(args.cache / arm / f"{record}.npz")
                    for arm in cache_plan["checkpoint_steps"] for record, _ in cache_plan["records"]
                }, dev_used=False, test_used=False)
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        c.require(json.loads(plan_path.read_text(encoding="utf8")) == plan, "Run inputs changed")
    else:
        c.write_json(plan_path, plan)

    device = torch.device("cuda")
    c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    torch.set_num_threads(4)
    arms = ("random_init", "pretrained")
    data = {arm: stage_arm(args.cache, arm, cache_plan["records"], args.pooling,
                           args.output / "staged" / arm) for arm in arms}
    rows = data[arms[0]][0]
    c.require(rows == data[arms[1]][0], "Backbone arms differ in selected samples")
    subject_ids = np.array([row["subject_id"] for row in rows])
    labels = np.array([row["y_true"] for row in rows], dtype=np.int64)
    n_epochs = plan["epochs"]
    selections = {}
    for readout in ("lft", "linear"):
        for arm in arms:
            _, pooled, sources = data[arm]
            output = args.output / f"{readout}_{arm}"
            output.mkdir(exist_ok=True)
            selection_path = output / "screening_metrics.json"
            if selection_path.exists():
                selections[output.name] = json.loads(selection_path.read_text(encoding="utf8"))
                continue
            oof = np.full((n_epochs, len(rows), 5), np.nan, dtype=np.float32)
            fold_index = np.full(len(rows), -1, dtype=np.int8)
            for fold, held_subjects in enumerate(folds):
                held = np.isin(subject_ids, held_subjects)
                fold_index[held] = fold
                fold_output = output / f"fold{fold + 1}"

                def progress(stage, **values):
                    state = dict(readout=readout, backbone=arm, fold=fold + 1, stage=stage,
                                 **values)
                    c.write_json(args.output / "progress.json", state)
                    if not values.get("updates"):
                        print(json.dumps(state), flush=True)

                started = time.perf_counter()
                if readout == "lft":
                    fit_data = c.CachedEpochs(rows, np.flatnonzero(~held), config["layers"], sources)
                    held_data = c.CachedEpochs(rows, np.flatnonzero(held), config["layers"], sources)
                    _, _, _, predictions, _ = fit_phase(
                        config, args.seed, device, fit_data, fold_output / "normalization.pt",
                        fold_output, n_epochs, progress, held_data,
                    )
                    oof[:, held] = predictions.numpy()
                else:
                    progress("training")
                    _, (fit, validation) = b["standardize"](pooled[~held], pooled[held])
                    _, predictions, _ = b["train_head"](fit, labels[~held], args.seed,
                                                        n_epochs, validation)
                    oof[:, held] = np.stack(predictions)
                progress("fold_complete", seconds=time.perf_counter() - started)
            selected = write_selection(output, rows, folds, fold_index, oof, evaluator)
            probabilities = oof[selected["epoch"] - 1]
            selections[output.name] = dict(
                selected, oof_n1_ap=float(average_precision_score(labels == 1, probabilities[:, 1]))
            )
            c.write_json(selection_path, selections[output.name])
    deltas = {
        metric: {
            head: selections[f"{head}_pretrained"][metric] - selections[f"{head}_random_init"][metric]
            for head in ("linear", "lft")
        }
        for metric in ("oof_macro_f1", "oof_n1_f1", "oof_n1_ap")
    }
    for values in deltas.values():
        values["lft_minus_linear_gain"] = values["lft"] - values["linear"]
    c.write_json(args.output / "summary.json", dict(arms=selections, pretrained_gain=deltas,
                                                  dev_used=False, test_used=False))
    c.write_json(args.output / "progress.json", dict(stage="complete"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b-checkout", type=Path, required=True)
    parser.add_argument("--pooling", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
