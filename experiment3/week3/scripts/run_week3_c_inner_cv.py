"""Refit the all20 LFT with B's subject-wise epoch selection protocol."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
import time

import numpy as np
import torch

import run_week3_c_readout as c

B_PROBE_SHA = "c7e8ce018e71d4168664720529fb76bd7f42d254f5e7a6029eff4355cdb8c726"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path, state):
    partial = path.with_suffix(".tmp")
    torch.save(state, partial)
    partial.replace(path)


def stage_training_features(rows, positions, folder):
    """Move repeated reads off the Windows mount without changing feature values."""
    folder.mkdir(parents=True, exist_ok=True)
    sources = {name: folder / name for name in c.MODALITIES}
    paths = sorted({rows[i]["feature_path"] for i in positions})
    receipt = folder / "copies.json"
    copies = load_json(receipt) if receipt.exists() else {}
    for name, output in sources.items():
        output.mkdir(exist_ok=True)
        for filename in paths:
            original = c.SOURCES[name] / filename
            target = output / filename
            stat = original.stat()
            key = f"{name}/{filename}"
            saved = copies.get(key, {})
            if (target.is_file() and saved.get("size") == stat.st_size
                    and saved.get("mtime_ns") == stat.st_mtime_ns
                    and target.stat().st_size == stat.st_size
                    and c.sha(target) == saved.get("sha256")):
                continue
            partial = target.with_suffix(".partial")
            digest = hashlib.sha256()
            with original.open("rb") as incoming, partial.open("wb") as outgoing:
                while block := incoming.read(16 * 2**20):
                    digest.update(block)
                    outgoing.write(block)
            c.require(c.sha(partial) == digest.hexdigest()
                      and original.stat().st_mtime_ns == stat.st_mtime_ns,
                      f"Staged feature differs from its source: {original}")
            partial.replace(target)
            copies[key] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest.hexdigest())
            c.write_json(receipt, copies)
    return sources, copies


def fit_phase(config, seed, device, data, normalization, output, epochs, progress, held=None):
    """Train one fold or the final refit, committing each completed epoch."""
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    models = {"C1": c.build_heads(config, seed, device)["C1"]}
    model = models["C1"]
    optimizer = c.optimizer_for(model, config)
    rng = np.random.default_rng(seed)
    if normalization.exists():
        stats = torch.load(normalization, map_location=device, weights_only=True)
        mean, std = stats["mean"], stats["std"]
    else:
        progress("fitting_train_normalization")
        mean, std = c.train_statistics(data, config, device)
        save_checkpoint(normalization, dict(mean=mean.cpu(), std=std.cpu()))
    history = []
    fold_predictions = torch.empty((epochs, len(held.rows), 5)) if held is not None else None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        c.require(state["seed"] == seed and state["epochs_planned"] == epochs,
                  f"Checkpoint protocol differs in {output}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["shuffle_rng"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device)
        history = state["history"]
        if held is not None:
            fold_predictions[:len(history)] = state["fold_predictions"]
    for epoch in range(len(history) + 1, epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        progress("training", epoch=epoch, epochs_planned=epochs, updates=0)

        def on_batch(updates, samples, losses):
            loss = float(losses.item() / samples)
            c.require(np.isfinite(loss), "Nonfinite training loss")
            progress("training", epoch=epoch, epochs_planned=epochs, updates=updates,
                     samples=samples, train_loss=loss, epoch_seconds=time.perf_counter() - start)

        updates, samples, losses = c.train_epoch(
            models, {"C1": optimizer}, data, config, device, mean, std, rng, on_batch
        )
        train_seconds = time.perf_counter() - start
        loss = float(losses.item() / samples)
        c.require(np.isfinite(loss), "Nonfinite training loss")
        if held is not None:
            progress("evaluating_inner_holdout", epoch=epoch, epochs_planned=epochs)
            model.eval()
            probabilities = c.predict_development(models, held, config, device, mean, std)["C1"]
            c.require(np.isfinite(probabilities).all(), "Incomplete inner holdout predictions")
            fold_predictions[epoch - 1] = torch.from_numpy(probabilities)
        history.append(dict(epoch=epoch, train_samples=samples, updates=updates, train_loss=loss,
                            train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - start,
                            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20))
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), seed=seed,
                     epochs_planned=epochs, history=history, shuffle_rng=rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                     fold_predictions=fold_predictions[:epoch] if held is not None else None)
        save_checkpoint(checkpoint, state)
        c.write_json(output / "history.json", history)
    c.write_json(output / "history.json", history)
    model.eval()
    return model, mean, std, fold_predictions, history


def write_selection(output, train_rows, folds, fold_index, oof, evaluator):
    c.require(np.all(fold_index >= 0) and np.isfinite(oof).all(), "Incomplete training-side predictions")
    truth = np.array([int(row["y_true"]) for row in train_rows])
    curve = []
    for epoch, probabilities in enumerate(oof, 1):
        metrics = evaluator.compute_metrics(truth, probabilities.argmax(axis=1))
        curve.append(dict(epoch=epoch, oof_macro_f1=metrics["macro_f1"], oof_n1_f1=metrics["n1_f1"]))
    best = max(curve, key=lambda row: (row["oof_macro_f1"], -row["epoch"]))
    c.write_csv(output / "inner_cv.csv", curve)
    predictions = []
    for row, fold, probability in zip(train_rows, fold_index, oof[best["epoch"] - 1]):
        predictions.append(dict(
            **{key: row[key] for key in ("sample_id", "subject_id", "record_id", "epoch_index", "y_true")},
            fold=int(fold), y_pred=int(probability.argmax()),
            **dict(zip(evaluator.PROB_COLS, map(float, probability))),
        ))
    c.write_csv(output / "oof_train.csv", predictions)
    c.write_json(output / "selection.json", dict(best, folds=folds, dev_used=False))
    return best


def run(args):
    b_script = args.b_checkout / "scripts/27_b_w3_linear_probe.py"
    c.require(c.sha(b_script) == B_PROBE_SHA, "B selection source differs from the pinned version")
    sys.path.insert(0, str(b_script.parent))
    b = runpy.run_path(str(b_script))
    b_common = b["w"]
    evaluator = b["ev"]
    dev_reference = b_common.VAL_REFERENCE
    config = load_json(c.CONFIG)
    c.require(config["budget"] == "label25" and config["layers"] == list(range(20))
              and b["MAX_EPOCHS"] == 50 and b["FOLD_SEED"] == 0, "Unexpected main experiment protocol")
    config.update(epochs=b["MAX_EPOCHS"],
                  checkpoint_selection="B subject-wise five-fold pooled OOF macro-F1; ties choose earliest",
                  normalization="per modality/layer/hidden shared across time; fitted on each fold's training subjects, then all 14 for the final refit")
    device = torch.device("cuda")
    c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    torch.set_num_threads(4)
    output = args.output
    status = dict(budget="label25", seeds=config["seeds"], completed_seeds=[],
                  folds_per_seed=5, epochs_per_fold=50, test_used=False)
    context = {}

    def progress(stage, **values):
        updated = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
        value = dict(status, stage=stage, updated_at=updated, **context, **values)
        c.write_json(output / "progress.json", value)
        print(json.dumps(value), flush=True)

    try:
        progress("checking_inputs")
        rows, train, dev, audit = c.inspect_inputs(dev_reference)
        subject_ids = audit["labeled_subjects"]
        b_subjects, _ = b_common.selected_subjects(args.b_checkout / "manifests/label_subsets_v2.csv", "label25")
        c.require(set(b_subjects) == set(subject_ids), "B and C training subject lists differ")
        folds = b["subject_folds"](subject_ids, min(5, len(subject_ids)))
        c.require(sorted(s for fold in folds for s in fold) == sorted(subject_ids),
                  "Fold assignment does not cover the labeled subjects once")
        train_rows = [rows[i] for i in train]
        train_subjects = np.array([row["subject_id"] for row in train_rows])
        c.require(not (set(train_subjects) & {rows[i]["subject_id"] for i in dev}),
                  "Outer development subjects overlap training")
        required = sum((folder / name).stat().st_size for folder in c.SOURCES.values()
                       for name in {row["feature_path"] for row in train_rows})
        c.require(shutil.disk_usage(args.staging.parent).free > required + 2**31,
                  "Insufficient space on the WSL training cache volume")
        status.update(train_samples=len(train), dev_samples=len(dev))
        progress("staging_training_features")
        sources, copies = stage_training_features(rows, train, args.staging)
        plan = dict(config=config, folds=folds, input_audit=audit, b_probe_sha256=B_PROBE_SHA,
                    b_evaluator_sha256=c.sha(args.b_checkout / "src/evaluation/evaluate.py"),
                    cv_runner_sha256=c.sha(Path(__file__)), staged_features=copies)
        plan_path = output / "plan.json"
        if plan_path.exists():
            c.require(load_json(plan_path) == json.loads(json.dumps(plan)), "CV inputs changed since launch")
        else:
            c.write_json(plan_path, plan)
        norm = output / "normalization"
        norm.mkdir(exist_ok=True)
        smoke_path = output / "smoke.json"
        if not smoke_path.exists():
            c.write_json(smoke_path, c.smoke(config, rows, train, device, sources))
        for seed in config["seeds"]:
            seed_dir = output / f"seed{seed}_v01"
            seed_dir.mkdir(exist_ok=True)
            if (seed_dir / "complete.json").exists():
                status["completed_seeds"].append(seed)
                continue
            oof = np.full((config["epochs"], len(train), 5), np.nan, dtype=np.float32)
            fold_index = np.full(len(train), -1, dtype=np.int8)
            for k, held_subjects in enumerate(folds):
                context.update(seed=seed, phase="inner_cv", fold=k + 1)
                held_mask = np.isin(train_subjects, held_subjects)
                c.require(held_mask.any() and (~held_mask).any() and np.all(fold_index[held_mask] == -1),
                          "Invalid held-out fold")
                fit_data = c.CachedEpochs(rows, train[~held_mask], config["layers"], sources)
                held_data = c.CachedEpochs(rows, train[held_mask], config["layers"], sources)
                _, _, _, predictions, _ = fit_phase(
                    config, seed, device, fit_data, norm / f"fold{k + 1}.pt",
                    seed_dir / f"fold{k + 1}", config["epochs"], progress, held_data
                )
                oof[:, held_mask] = predictions.numpy()
                fold_index[held_mask] = k
                progress("fold_complete")
            selected = write_selection(seed_dir, train_rows, folds, fold_index, oof, evaluator)
            context.clear()
            context.update(seed=seed, phase="final_refit", selected_epoch=selected["epoch"])
            fit_data = c.CachedEpochs(rows, train, config["layers"], sources)
            model, mean, std, _, history = fit_phase(
                config, seed, device, fit_data, norm / "all14.pt", seed_dir / "refit",
                selected["epoch"], progress
            )
            save_checkpoint(seed_dir / "final_head.pt",
                            dict(model=model.state_dict(), mean=mean.cpu(), std=std.cpu(),
                                 seed=seed, epoch=selected["epoch"], config=config))
            progress("evaluating_outer_development")
            dev_data = c.CachedEpochs(rows, dev, config["layers"])
            probabilities = c.predict_development({"C1": model}, dev_data, config, device, mean, std)["C1"]
            c.require(np.isfinite(probabilities).all(), "Nonfinite outer development predictions")
            path = seed_dir / "predictions_val.csv"
            b_common.write_prediction_table(path, dev_data.rows, probabilities)
            metrics = evaluator.evaluate(path, reference=dev_reference, n_boot=2000, seed=0)
            metrics.update(selected_epoch=selected["epoch"], oof_macro_f1=selected["oof_macro_f1"],
                           final_updates=sum(epoch["updates"] for epoch in history), test_used=False)
            c.write_json(seed_dir / "metrics.json", metrics)
            evaluator.per_subject_table(evaluator.load_predictions(path)).to_csv(
                seed_dir / "per_subject_metrics.csv", index=False)
            c.write_json(seed_dir / "complete.json", dict(seed=seed, selected_epoch=selected["epoch"],
                                                         predictions_sha256=c.sha(path)))
            status["completed_seeds"].append(seed)
            progress("seed_complete")
        context.clear()
        progress("complete")
    except Exception as error:
        progress("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
