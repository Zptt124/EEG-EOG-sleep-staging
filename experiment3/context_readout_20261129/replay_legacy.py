"""Reproduce six archived K1 models for calibration before the new run."""

import argparse
import csv
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from prepare import (BUNDLE, DEST, OUT, RECIPE, RESULT, ROOT, STAGES, hash_file,
                     inputs, read_json, read_rows, require, save_json, verify_selection)
from run_screening_bigru import previous_runner, training_context


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-context", type=Path)
    args = parser.parse_args()
    RESULT.mkdir(parents=True, exist_ok=True)
    calibration = read_json(RECIPE)
    runner = previous_runner()
    sources = (ROOT / "experiment3/window_budget_c_20261128/run_screening_bigru.py",
               Path(runner.__file__), runner.CONTEXT_SOURCE / "13_context_head.py")
    for source in sources:
        require(hash_file(source) == calibration["source_sha256"][source.name],
                f"Historical source altered: {source}")
    anchors, chosen, _ = verify_selection()
    _, _, info, _, shared, evaluation, eval_keys, cached, cache_x, cache_y = inputs()
    all_centers = sorted(set().union(*anchors.values(), *(keys for (_, _, c, s), keys in chosen.items()
                                                        if c in ("A", "B") and s in STAGES)))
    positions = {key: i for i, key in enumerate(all_centers)}
    cache_idx = [cached[k] for k in all_centers]
    arrays = []
    for name in ("eeg_std", "eog_std"):
        path = BUNDLE / f"{name}.npy"
        require(hash_file(path) == info["files"][name]["sha256"], f"Changed feature file: {name}")
        arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
    windows = np.asarray([[shared.get((record, e + shift), shared[record, e]) for shift in (-1, 0, 1)]
                          for record, e in eval_keys])
    eval_x = np.concatenate([arr[windows] for arr in arrays], axis=-1)
    torch.set_num_threads(4)
    train_x, supplemental = training_context(positions, cache_x[cache_idx], shared, arrays,
        {r["record_id"]: r for r in read_rows(OUT / "recording_metadata.csv")})
    if args.audit_context:
        with np.load(args.audit_context, allow_pickle=False) as data:
            near = data["train_windows"][:, 9:12]
            records, epochs, new = data["feature_record"], data["feature_epoch"], data["x"]
            keys = list(zip(records[near[:, 1]], epochs[near[:, 1]]))
            diffs = np.max(np.abs(train_x[[positions[k] for k in keys]] - new[near]), axis=2)
            old_eval = data["old_eval_windows"]
            require(np.array_equal(eval_x, new[old_eval]), "Reconstructed old evaluation features differ")
        audit = RESULT / "k1_train_feature_audit.csv"
        with audit.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("record_id", "center_epoch", "prev_max_abs", "center_max_abs", "next_max_abs"))
            writer.writerows((*key, *map(float, diff)) for key, diff in zip(keys, diffs))
        require(np.all(diffs[:, 1] == 0), "Supervised center features changed")
        with np.load(args.audit_context, allow_pickle=False) as data:
            for roster in np.unique(data["roster"]):
                with np.load(DEST / "bigru_ab" / f"roster_{roster:02d}.npz", allow_pickle=False) as prior:
                    unit_ids = np.where(data["roster"] == roster)[0]
                    for i in unit_ids:
                        require(prior["scored_additions"][data["draw"][i] - 1, 0,
                                STAGES.index(data["strategy"][i])] == data["effective_additions"][i],
                                f"Historical supervised count differs for unit {i}")
        save_json(RESULT / "legacy_feature_check.json", dict(
            input_sha256=hash_file(args.audit_context), historical_train_extra_epochs=supplemental,
            audited_train_centers=len(keys), audited_unit_masks=260, old_evaluation_centers=len(eval_keys),
            changed_train_centers=int(np.count_nonzero(diffs.any(axis=1))),
            largest_train_abs_difference=float(diffs.max()), audit_sha256=hash_file(audit)))
        from train_bigru import configure, fit, infer
        roster = min(anchors)
        selected = sorted(set(anchors[roster]) | {k for k in chosen[roster, 1, "A", STAGES[0]]
                                                if cache_y[cached[k]] >= 0})
        y = cache_y[[cached[k] for k in selected]]
        device = configure("cuda")
        model, mean, std = fit(train_x[[positions[k] for k in selected]], y, 0, False, device)
        identity = np.arange(len(eval_x) * 3).reshape(-1, 3)
        pred, _ = infer(model, mean, std, eval_x.reshape(-1, 384), identity, device)
        refs = DEST / "bigru_ab" / f"roster_{roster:02d}.npz"
        with np.load(refs, allow_pickle=False) as old:
            mismatches = int(np.count_nonzero(pred != old["predictions"][0, 0, 0]))
        save_json(RESULT / "shared_trainer_legacy_check.json", dict(
            roster=roster, draw=1, strategy=STAGES[0], disagreements=mismatches,
            evaluation_centers=len(pred), trainer_sha256=hash_file(Path(__file__).with_name("train_bigru.py"))))
        require(mismatches <= len(pred) * .001, "Shared trainer failed legacy calibration")
        print(f"Shared trainer on archived K1 inputs: {mismatches} mismatches", flush=True)
        return
    torch.use_deterministic_algorithms(True)
    roster = min(anchors)
    references = DEST / "bigru_ab" / f"roster_{roster:02d}.npz"
    with np.load(references, allow_pickle=False) as archive:
        require(archive["protocol_sha256"].item() == hash_file(RECIPE), "Historical archive changed")
        results = []
        for draw in (1, 2, 3):
            for s, strategy in enumerate(STAGES):
                idx = [cached[k] for k in chosen[roster, draw, "A", strategy]]
                selected = sorted(set(anchors[roster]).union(k for k, j in zip(
                    chosen[roster, draw, "A", strategy], idx) if cache_y[j] >= 0))
                y = cache_y[[cached[k] for k in selected]].astype(np.int64)
                require(len(y) - 1200 == archive["scored_additions"][draw - 1, 0, s],
                        "The legacy supervision mask disagrees with archived effective budget")
                start = perf_counter()
                pred = runner.fit(train_x[[positions[k] for k in selected]], y, eval_x, 0)
                ref = archive["predictions"][draw - 1, 0, s]
                counts = np.bincount(evaluation["subject_index"] * 25 + evaluation["truth"] * 5 + pred,
                                     minlength=400).reshape(16, 25)
                disagreements = int(np.count_nonzero(pred != ref))
                results.append(dict(roster=roster, draw=draw, strategy=strategy, supervised=len(y),
                                    eval_centers=len(ref), disagreements=disagreements,
                                    agreement=1 - disagreements / len(ref),
                                    counts_equal=bool(np.array_equal(counts, archive["counts"][draw - 1, 0, s])),
                                    seconds=perf_counter() - start))
                print(f"Archived K1 roster {roster} draw {draw} {strategy}: {disagreements} mismatches", flush=True)
    save_json(RESULT / "legacy_replay.json", dict(archive_sha256=hash_file(references),
              protocol_sha256=hash_file(RECIPE), supplemental_train_epochs=supplemental,
              python=platform.python_version(), torch=str(torch.__version__),
              gpu=torch.cuda.get_device_name(), results=results))
    require(all(row["agreement"] >= .999 for row in results), "Archived K1 replay below 99.9%")


if __name__ == "__main__":
    main()
