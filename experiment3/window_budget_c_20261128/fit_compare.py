"""Fit the matched label-blind window arms and compare with archived old-window fits."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from download import ROOT, OLD, OUT, hash_file, read_rows
from common_run import BUNDLE, BLOCK, load_lock

sys.path.insert(0, str(ROOT / "experiment3/week3_budget_replication_20261128"))
from run import ARMS, features, fit, labels, macro, require, write_json


def group_blocks(path, paired):
    grouped = defaultdict(list)
    for row in read_rows(path):
        roster = int(row["roster"])
        if roster not in paired:
            continue
        key = (roster, int(row["draw"]), row["arm"])
        first = int(row["first_epoch"])
        grouped[key].extend((row["record_id"], i) for i in range(first, first + BLOCK))
    return grouped


def supplemental_data(record, expected, positions):
    path = OUT / "supplement" / f"{record}.npz"
    with np.load(path, allow_pickle=False) as saved:
        epoch, target, extra_epoch, x = (saved[k] for k in ("epoch", "y", "missing_epoch", "x"))
        require(np.array_equal(epoch, sorted(expected)), f"Changed selected epochs: {record}")
        require(np.array_equal(extra_epoch, [i for i in epoch if (record, i) not in positions]),
                f"Wrong supplemental epochs: {record}")
        require(x.shape == (len(extra_epoch), 384) and np.isfinite(x).all(), f"Bad features: {record}")
        require(np.isin(target, [-1, 0, 1, 2, 3, 4]).all(), f"Bad labels: {record}")
        return (dict(zip(epoch, target)), dict(zip(extra_epoch, x)),
                int(saved["checked_overlap"]), float(saved["max_abs_overlap_error"]))


def bootstrap(counts, good, n_resamples, seed):
    # Axis order is roster, draw, window, arm, original evaluation subject, confusion cell.
    roster_groups = np.flatnonzero(good.any(axis=1))
    if not len(roster_groups):
        return None
    counts = counts[roster_groups]
    good = good[roster_groups]
    samples = [np.flatnonzero(mask) for mask in good]
    n_rosters = len(samples)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_resamples, 3))

    def contrasts(conf):
        score = macro(conf)
        effect = score[..., 2] - score[..., 0]
        return np.concatenate([effect, (effect[..., 1] - effect[..., 0])[..., None]], axis=-1)

    point = contrasts(counts.sum(axis=-2))
    estimate = np.array([value[mask].mean(axis=0) for value, mask in zip(point, good)]).mean(axis=0)
    flat = counts.reshape(-1, 16, 25).transpose(1, 0, 2).reshape(16, -1)
    max_draws = counts.shape[1]
    padded = np.stack([np.pad(d, (0, max_draws - len(d))) for d in samples])
    sizes = np.array([len(d) for d in samples])
    for lo in range(0, n_resamples, 100):
        n = min(100, n_resamples - lo)
        rosters = rng.integers(n_rosters, size=(n, n_rosters))
        resample = rng.random((n, n_rosters, max_draws))
        nr = sizes[rosters, None]
        draw_slots = padded[rosters[..., None], (resample * nr).astype(int)]
        epoch_weights = rng.multinomial(8, [0.125] * 8, size=(n, 2)).reshape(n, 16)
        weighted = (epoch_weights @ flat).reshape(n, *counts.shape[:4], 25)
        cs = contrasts(weighted)
        values = cs[np.arange(n)[:, None, None], rosters[..., None], draw_slots]
        valid_draws = np.arange(max_draws) < nr
        boot[lo:lo+n] = ((values * valid_draws[..., None]).sum(axis=2) / nr).mean(axis=1)
    names = ("old_newcomer_minus_original_night", "lights_off_newcomer_minus_original_night",
             "paired_difference_of_differences")
    return dict(rosters=n_rosters, draw_pairs=int(good.sum()),
                **{name: dict(mean_pp=float(estimate[k]),
                              interval_95_pp=np.percentile(boot[:, k], [2.5, 97.5]).tolist())
                   for k, name in enumerate(names)})


def main():
    lock = load_lock()
    plan = json.loads((OLD / "plan.json").read_text())
    for name, digest in plan["locked_files"].items():
        require(hash_file(OLD / name) == digest, f"Changed historical file: {name}")
    paired = lock["paired_rosters"]
    additional = group_blocks(OUT / "selected_blocks.csv", paired)
    anchors = group_blocks(OUT / "anchor_blocks.csv", paired)
    needed = defaultdict(set)
    for keys in additional.values():
        for record, epoch in keys:
            needed[record].add(epoch)
    dest = OUT / "supplement"
    start = time.monotonic()
    while pending := sorted(r for r in needed if not (dest / f"{r}.npz").exists()):
        if time.monotonic() - start > 7200:
            raise TimeoutError(f"Missing supplemental outputs: {pending}")
        time.sleep(5)

    rows = read_rows(BUNDLE / "index.csv")
    positions = {(r["record_id"], int(r["epoch_index"])): i for i, r in enumerate(rows)}
    sample_position = {r["sample_id"]: i for i, r in enumerate(rows)}
    target, extras = {}, {}
    replay_count, max_delta = 0, 0.0
    for record, expected in sorted(needed.items()):
        y, x, checked, delta = supplemental_data(record, expected, positions)
        target.update({(record, k): v for k, v in y.items()})
        extras.update({(record, k): v for k, v in x.items()})
        replay_count += checked
        max_delta = max(max_delta, delta)

    arrays = [np.load(BUNDLE / f"{m}_std.npy", mmap_mode="r") for m in ("eeg", "eog")]
    with np.load(OLD / "evaluation_index.npz", allow_pickle=False) as eval_index:
        ids = [sample_position[k] for k in eval_index["sample_id"]]
        eval_people = eval_index["subjects"].tolist()
        eval_subjects = eval_index["subject_index"].copy()
        eval_y = eval_index["truth"].copy()
    require(eval_people == plan["dev8"] + plan["pretrain_eval8"] and len(set(ids)) == len(ids)
            and np.array_equal(eval_y, labels(rows, ids))
            and all(rows[i]["subject_id"] == eval_people[s] for i, s in zip(ids, eval_subjects)),
            "Changed evaluation data")
    eval_x = features(arrays, ids)
    old_meta = json.loads((OLD / "summary.json").read_text())
    require(old_meta["fits"] == 540, "Historical run is incomplete")
    n_draws = plan["draws_per_roster"]
    counts = np.zeros((len(paired), n_draws, 2, 3, 16, 25), dtype=np.int64)
    good = np.ones((len(paired), n_draws), dtype=bool)
    scores = []
    torch.set_num_threads(4)
    expected_rows = len(paired) * n_draws * 3
    print(f"Verified {len(needed)} recordings, {len(extras)} supplemental epochs, "
          f"{replay_count} replayed overlap epochs. Starting {expected_rows} fits.", flush=True)
    for j, roster in enumerate(paired):
        base = anchors[roster, 0, "base"]
        require(len(set(base)) == len(base) == 1200, f"Bad anchor for roster {roster}")
        historical = OLD / f"roster_{roster:02}.npz"
        with np.load(historical, allow_pickle=False) as old:
            counts[j, :, 0] = old["counts"][0]
            for d in range(1, n_draws + 1):
                for a, arm in enumerate(ARMS):
                    pred = old[f"{d}_{arm}_0"]
                    replay = np.bincount(eval_subjects * 25 + eval_y * 5 + pred,
                                         minlength=400).reshape(16, 25)
                    require(np.array_equal(replay, counts[j, d-1, 0, a]),
                            f"Historical predictions do not reproduce counts: {roster=} {d=} {arm=}")
        base_x = {key: value for key, value in zip(base, features(arrays, [positions[k] for k in base]))}
        for d in range(1, n_draws + 1):
            added = [additional[roster, d, arm] for arm in ARMS]
            require(all(len(a) == 600 for a in added) and len(set(base + sum(added, []))) == 3000,
                    f"Overlapping or malformed allocations: {roster=} {d=}")
            for a, (arm, keys) in enumerate(zip(ARMS, added)):
                observed = [key for key in keys if target[key] >= 0]
                good[j, d-1] &= len(observed) == 600
                train_keys = sorted(base + observed)
                y = np.array([int(rows[positions[k]]["y_true"]) if k in base_x else target[k]
                              for k in train_keys])
                x = np.array([base_x[k] if k in base_x else
                              (np.concatenate([v[positions[k]] for v in arrays])
                               if k in positions else extras[k]) for k in train_keys])
                require(np.isfinite(x).all() and np.isin(y, range(5)).all(), "Invalid training inputs")
                model, mean, sd, curve = fit(x, y, lock["head_seed"], lock["head_epochs"])
                with torch.inference_mode():
                    pred = model(torch.from_numpy((eval_x - mean) / sd)).argmax(dim=1).numpy()
                conf = np.bincount(eval_subjects * 25 + eval_y * 5 + pred, minlength=400).reshape(16, 25)
                counts[j, d-1, 1, a] = conf
                scores.append(dict(roster=roster, draw=d, arm=arm, attempted_additions=600,
                                   scorable_additions=len(observed),
                                   old_macro_f1=macro(counts[j, d-1, 0, a].sum(axis=0)).item(),
                                   lights_off_macro_f1=macro(conf.sum(axis=0)).item(),
                                   updates=curve[-1]["cumulative_updates"]))
        print(f"Roster {roster:02} complete | {len(scores)}/{expected_rows} fits | "
              f"{int(good[j].sum())}/{n_draws} exact-budget draws", flush=True)

    with (OUT / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    with (OUT / "paired_deltas.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["roster", "draw", "exact_600_all_arms", "old_delta_pp", "lights_off_delta_pp",
                         "difference_of_differences_pp"])
        f1 = macro(counts.sum(axis=-2))
        differences = f1[..., 2] - f1[..., 0]
        for j, roster in enumerate(paired):
            for d, (old, new) in enumerate(differences[j], 1):
                writer.writerow([roster, d, int(good[j, d-1]), old, new, new - old])
    np.savez_compressed(OUT / "paired_confusions.npz", counts=counts, rosters=paired, exact_600=good,
                        windows=["old", "lights_off"], arms=ARMS, evaluation_subjects=eval_people)
    print("All fits complete. Computing paired intervals.", flush=True)
    boot = (lock["bootstrap_draws"], lock["bootstrap_seed"])
    summary = dict(protocol_sha256=hash_file(OUT / "protocol.json"), evaluation_epochs=len(ids),
                   evaluation_people=len(eval_people), fits=len(scores),
                   supplemental_epochs=len(extras), replayed_features=replay_count,
                   max_abs_replay_error=max_delta,
                   exact_600=bootstrap(counts, good, *boot),
                   fixed_600_attempts=bootstrap(counts, np.ones_like(good), *boot))
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
