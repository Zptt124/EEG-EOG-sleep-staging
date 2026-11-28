"""Compare nested 600- and 1,200-epoch additions using the frozen Sleep-EDF line."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment3/week3_budget_replication_20261128"))
from run import ARMS, BLOCK_SIZE, FEATURE_HASH, N_BOOT, N_DRAWS
from run import features, fit, labels, macro, paired, read_csv, require, sha256, write_csv, write_json

EXTRA_SEED = 20261129
CONTRASTS = ("other_minus_deepen", "new_minus_other", "new_minus_deepen")
PRIMARY = "new_minus_deepen"


def block_key(row):
    return (
        int(row["roster"]), row["arm"], row["subject_id"], row["record_id"], int(row["night_third"])
    )


def make_plan(source, output, old_plan):
    for filename, digest in old_plan["locked_files"].items():
        require(sha256(source / filename) == digest, f"Frozen input changed: {filename}")
    candidates = defaultdict(set)
    for row in read_csv(source / "candidate_blocks.csv"):
        candidates[block_key(row)].add(int(row["first_epoch"]))
    by_roster = defaultdict(list)
    for key, starts in candidates.items():
        by_roster[key[0]].append(len(starts))
    require(set(by_roster) == set(range(1, 31)) and all(len(v) == 30 for v in by_roster.values()),
            "Incomplete frozen candidate coverage")
    eligible = sorted(i for i, sizes in by_roster.items() if min(sizes) >= 2)
    require(set(range(1, 31)) - set(eligible) == {25, 27}, "Unexpected eligibility")
    training_blocks = read_csv(source / "blocks.csv")
    additions = sorted(
        (r for r in training_blocks if int(r["draw"]) > 0 and int(r["roster"]) in eligible),
        key=lambda row: (int(row["draw"]), block_key(row)),
    )
    seen = set()
    extended = []
    rng = random.Random(EXTRA_SEED)
    for row in additions:
        key = block_key(row)
        draw_key = (int(row["draw"]), key)
        require(draw_key not in seen, "More than one old block in a draw cell")
        seen.add(draw_key)
        start = int(row["first_epoch"])
        require(start in candidates[key], "Published low-budget block outside frozen candidates")
        second = rng.choice(sorted(candidates[key] - {start}))
        extended.extend([{**row, "block_number": 1}, {**row, "first_epoch": second, "block_number": 2}])
    require(len(seen) == len(eligible) * N_DRAWS * 30, "Incomplete matched design")
    archive_names = ["plan.json", "blocks.csv", "candidate_blocks.csv", "evaluation_index.npz"]
    archive_names.extend(f"roster_{i:02d}.npz" for i in eligible)
    protocol = {
        "rosters": eligible,
        "excluded_for_candidate_count": [25, 27],
        "base_epochs": 1200,
        "low_added_epochs": 600,
        "high_added_epochs": 1200,
        "draws_per_roster": N_DRAWS,
        "high_new_fits": len(eligible) * N_DRAWS * len(ARMS),
        "additional_block_seed": EXTRA_SEED,
        "block_rule": "Uniform draw from other locked candidates in the same subject, night and third. "
                      "Retain original first block separately per draw.",
        "head_seed": old_plan["primary_head_seed"],
        "head_epochs": old_plan["head_epochs"],
        "head_note": "Fixed epochs for both budgets. Higher budget entails more optimizer steps per epoch.",
        "primary": PRIMARY,
        "analysis": "Report the contrast at each budget and high minus low. "
                    "Bootstrap 10000 times, pairing budgets/arms over rosters, draws, and subjects. "
                    "Resample evaluation subjects within the two original groups of 8. "
                    "Conditional on the observed training pool and 28 eligible rosters.",
        "interpretation": "Positive at high budget if its interval is above zero. "
                          "Note separately whether the point estimate reaches 1 pp. "
                          "Attenuation or growth requires the high-minus-low interval below or above zero. "
                          "An interval spanning zero leaves the budget interaction unresolved.",
        "locked_inputs_sha256": {name: sha256(source / name) for name in archive_names},
        "features_info_sha256": FEATURE_HASH,
        "code_sha256": {Path(path).name: sha256(Path(path))
                        for path in (__file__, macro.__code__.co_filename, fit.__code__.co_filename)},
    }
    protocol_path = output / "protocol.json"
    frozen_blocks = output / "extended_blocks.csv"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        require(sha256(frozen_blocks) == saved.pop("extended_blocks_sha256"), "Extended blocks changed")
        require(protocol == saved, "Run protocol changed")
    else:
        write_csv(frozen_blocks, extended)
        write_json(protocol_path, {**protocol, "extended_blocks_sha256": sha256(frozen_blocks)})
    baseline = [r for r in training_blocks if r["arm"] == "base" and int(r["roster"]) in eligible]
    return protocol, baseline + extended


def load_cache(bundle, source, old_plan):
    metadata = bundle / "features_info.json"
    require(sha256(metadata) == FEATURE_HASH, "Unexpected feature receipt")
    receipt = json.loads(metadata.read_text())
    index_path = bundle / "index.csv"
    require(sha256(index_path) == receipt["index_sha256"], "Feature index differs")
    rows = read_csv(index_path)
    arrays = []
    for name in ("eeg_std", "eog_std"):
        path = bundle / f"{name}.npy"
        require(sha256(path) == receipt["files"][name]["sha256"], f"Feature checksum mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(array.shape == (len(rows), 192) and array.dtype == np.float32, f"Feature shape: {name}")
        arrays.append(array)
    people = old_plan["dev8"] + old_plan["pretrain_eval8"]
    person_index = {s: i for i, s in enumerate(people)}
    ids = sorted((i for i, r in enumerate(rows) if r["subject_id"] in person_index),
                 key=lambda i: (person_index[rows[i]["subject_id"]], i))
    truth = labels(rows, ids)
    groups = np.array([person_index[rows[i]["subject_id"]] for i in ids])
    with np.load(source / "evaluation_index.npz", allow_pickle=False) as saved:
        comparisons = ((saved["sample_id"], [rows[i]["sample_id"] for i in ids]),
                       (saved["truth"], truth), (saved["subject_index"], groups), (saved["subjects"], people))
        require(all(np.array_equal(a, b) for a, b in comparisons), "Evaluation differs from low budget")
    train_index = {
        (r["subject_id"], r["record_id"], int(r["epoch_index"])): i
        for i, r in enumerate(rows) if r["role"] == "pretrain_train" and r["subject_id"] not in person_index
    }
    return rows, arrays, train_index, features(arrays, ids), groups * 25 + truth * 5


def make_masks(blocks, train_index, rosters):
    masks = defaultdict(list)
    for row in blocks:
        key = int(row["roster"]), int(row["draw"]), row["arm"]
        start = int(row["first_epoch"])
        masks[key].extend(train_index[row["subject_id"], row["record_id"], e]
                          for e in range(start, start + BLOCK_SIZE))
    for r in rosters:
        base = masks[r, 0, "base"]
        require(len(base) == len(set(base)) == 1200, "Malformed original mask")
        for d in range(1, N_DRAWS + 1):
            added = [masks[r, d, arm] for arm in ARMS]
            require(all(len(seq) == 1200 for seq in added)
                    and len(set(base + sum(added, []))) == 4800, "Malformed higher budget or pairing")
    return masks


def load_low_counts(source, roster, truth_keys, seed):
    filename = source / f"roster_{roster:02d}.npz"
    with np.load(filename, allow_pickle=False) as saved:
        counts = saved["counts"][seed]
        for draw in range(1, N_DRAWS + 1):
            for i, arm in enumerate(ARMS):
                predicted = saved[f"{draw}_{arm}_{seed}"]
                measured = np.bincount(truth_keys + predicted, minlength=400).reshape(16, 25)
                require(np.array_equal(measured, counts[draw - 1, i]), f"Previous counts mismatch: {filename}")
    return counts


def train_all(source, output, protocol, cache, masks):
    rows, arrays, _, eval_x, truth_keys = cache
    low, high, scores = [], [], []
    seed = protocol["head_seed"]
    rosters = protocol["rosters"]
    start = time.perf_counter()
    fresh = 0
    for j, r in enumerate(rosters, start=1):
        prior_counts = load_low_counts(source, r, truth_keys, seed)
        low.append(prior_counts)
        finished = output / f"roster_{r:02d}.npz"
        if finished.exists():
            with np.load(finished, allow_pickle=False) as archive:
                high.append(archive["counts"])
                scores.extend(json.loads(str(archive["scores"])))
            continue
        base = masks[r, 0, "base"]
        counts = np.zeros((N_DRAWS, len(ARMS), 16, 25), dtype=np.int64)
        measurements, predictions = [], {}
        for d in range(1, N_DRAWS + 1):
            for i, arm in enumerate(ARMS):
                ids = sorted(base + masks[r, d, arm])
                model, mean, std, curve = fit(features(arrays, ids), labels(rows, ids), seed, protocol["head_epochs"])
                with torch.inference_mode():
                    pred = model(torch.from_numpy((eval_x - mean) / std)).argmax(dim=1).numpy()
                conf = np.bincount(truth_keys + pred, minlength=400).reshape(16, 25)
                counts[d - 1, i] = conf
                predictions[f"{d}_{arm}"] = pred.astype(np.uint8)
                measurements.append(dict(roster=r, draw=d, arm=arm, head_seed=seed,
                                         macro_f1_low=macro(prior_counts[d - 1, i].sum(axis=0)).item(),
                                         macro_f1_high=macro(conf.sum(axis=0)).item(),
                                         updates_high=curve[-1]["cumulative_updates"]))
        pending = finished.with_suffix(".tmp")
        with pending.open("wb") as handle:
            np.savez_compressed(handle, counts=counts, scores=json.dumps(measurements), **predictions)
        pending.replace(finished)
        high.append(counts)
        scores.extend(measurements)
        fresh += len(measurements)
        completed = j * N_DRAWS * len(ARMS)
        left = protocol["high_new_fits"] - completed
        eta = (time.perf_counter() - start) * left / fresh
        print(f"Roster {r:02d} done | {completed}/{protocol['high_new_fits']} | fit ETA {eta:.0f}s", flush=True)
    write_csv(output / "scores.csv", scores)
    return np.stack((low, high))


def analyze(counts, protocol, output, elapsed):
    n_roster = len(protocol["rosters"])
    draws = np.empty((N_BOOT, 2, len(CONTRASTS)))
    flat = np.moveaxis(counts, -2, 0).reshape(16, -1)
    rng = np.random.default_rng(0)
    for start in range(0, N_BOOT, 100):
        batch = min(100, N_BOOT - start)
        r = rng.integers(n_roster, size=(batch, n_roster, 1))
        d = rng.integers(N_DRAWS, size=(batch, n_roster, N_DRAWS))
        weights = rng.multinomial(8, [1 / 8] * 8, size=(batch, 2)).reshape(batch, 16)
        resampled = (weights @ flat).reshape(batch, 2, n_roster, N_DRAWS, 3, 25)
        effects = paired(macro(resampled))
        picked = effects[np.arange(batch)[:, None, None], :, r, d]
        draws[start:start + batch] = picked.mean(axis=(1, 2))
    arm_scores = macro(counts.sum(axis=-2))
    effects = paired(arm_scores)
    comparisons = {}
    for i, name in enumerate(CONTRASTS):
        lo, hi = effects[..., i]
        measures = (lo, hi, hi - lo)
        simulations = (draws[:, 0, i], draws[:, 1, i], draws[:, 1, i] - draws[:, 0, i])
        comparisons[name] = {
            key: {"mean_pp": float(value.mean()), "interval_95_pp": np.percentile(sample, [2.5, 97.5]).tolist(),
                  "positive_rosters": int((value.mean(axis=1) > 0).sum()),
                  "roster_means_pp": value.mean(axis=1).tolist()}
            for key, value, sample in zip(("low", "high", "high_minus_low"), measures, simulations)
        }
    result = dict(primary=PRIMARY, high_new_fits=protocol["high_new_fits"], elapsed_seconds=elapsed(),
                  rosters=protocol["rosters"], contrasts=comparisons,
                  arm_f1={key: dict(zip(ARMS, values.tolist()))
                          for key, values in zip(("low", "high"), arm_scores.mean(axis=(1, 2)))})
    write_json(output / "summary.json", result)
    main = result["contrasts"][PRIMARY]
    for key, estimates in main.items():
        lo, hi = estimates["interval_95_pp"]
        print(f"{key}: {estimates['mean_pp']:+.2f} pp [{lo:+.2f}, {hi:+.2f}]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "output/budget_replication_20261128")
    parser.add_argument("--bundle", type=Path, default=ROOT / "experiment3/deliveries/d-w6-shared-cache-v01-20261126/full")
    parser.add_argument("--output", type=Path, default=ROOT / "output/budget_extension_20261128")
    args = parser.parse_args()
    start = time.perf_counter()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    old_plan = json.loads((args.source / "plan.json").read_text(encoding="utf-8"))
    protocol, blocks = make_plan(args.source, args.output, old_plan)
    cache = load_cache(args.bundle, args.source, old_plan)
    masks = make_masks(blocks, cache[2], protocol["rosters"])
    print(f"Locked {len(protocol['rosters'])} rosters | {protocol['high_new_fits']} CPU fits", flush=True)
    counts = train_all(args.source, args.output, protocol, cache, masks)
    print("Fits done | paired bootstrap", flush=True)
    analyze(counts, protocol, args.output, lambda: time.perf_counter() - start)


if __name__ == "__main__":
    main()
