"""Paired external confirmation of how to spend an additional 600 labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment3/week3_budget_replication_20261128"))
from run import ARMS, BLOCK_SIZE, N_DRAWS, fit, macro, paired, write_json
from protocol import write_csv, sha256

N_FOLDS = 4
N_ROSTERS = 8
N_BOOT = 10000
SEED = 20261128
PRESET = {
    "primary": "add_people_minus_deepen",
    "practical_pp": 1,
    "folds": N_FOLDS, "rosters_per_fold": N_ROSTERS, "draws": N_DRAWS,
    "block_epochs": BLOCK_SIZE, "base_epochs": 1200, "added_epochs": 600,
    "head_seed": 0, "head_epochs": 30, "fold_and_roster_seed": SEED, "bootstrap_seed": 0,
    "n_bootstrap": N_BOOT, "metric": "per-fold pooled five-class Macro-F1, fold-equal paired average",
    "input": "Ying OSF gx36m; same-amplifier Days 1 and 2; documented missing Day 2 for 004 and 020",
    "encoding": "F4-A1 and EOG-R minus EOG-L; 500 -> 100 -> 200 Hz; frozen W4 30k EMA",
    "target": "3-expert majority; Expert 1 on ties; complete annotation-aligned 30 s epochs; "
              "omit author NotScored epochs and use common annotation prefix on SLP010 Day 1",
    "time_matching": "same pair of night thirds across all arms; newcomer matched to other_night day",
    "training_eligibility": "both Days 1-2, each with at least six usable blocks across at least two thirds",
    "boot": "resample held-out people, rosters and paired block draws within fixed folds; conditional on observed cohort",
    "decision": "external_support if mean >= 1 pp, lower bound > 0 and >= 3 of 4 fold means positive; "
                "below_practical_target if upper bound < 1; otherwise inconclusive",
}


def expected_folds():
    subjects = [f"SLP{i:03d}" for i in range(1, 21)]
    random.Random(SEED).shuffle(subjects)
    return [sorted(subjects[i::N_FOLDS]) for i in range(N_FOLDS)]


def nights_and_epochs(directory):
    files = sorted((directory / "features").glob("*.npz"))
    expected = {f"SLP{i:03d}_Day{d}" for i in range(1, 21) for d in (1, 2)
                if (i,d) not in {(4,2),(20,2)}}
    if {p.stem for p in files} != expected:
        raise ValueError("Incomplete external feature cache")
    nights, data = {}, []
    cursor = 0
    for p in files:
        subject, day = p.stem.split("_Day")
        with np.load(p, allow_pickle=False) as r:
            x = r["x"]
            y = r["y"]
        if x.shape != (len(y), 384) or not np.isin(y, np.arange(-1, 5)).all():
            raise ValueError(f"Malformed features: {p.name}")
        data.append((p.stem, x, y))
        thirds = [[], [], []]
        for start in range(0, len(y) - BLOCK_SIZE + 1, BLOCK_SIZE):
            third = min(2, (3 * (2 * start + BLOCK_SIZE - 1)) // (2 * len(y)))
            if (y[start:start+BLOCK_SIZE] >= 0).all():
                thirds[third].append(cursor + start)
        nights.setdefault(subject, {})[int(day)] = (thirds, cursor, len(y), p.stem)
        cursor += len(y)
    return nights, data


def design(nights, folds):
    rng = random.Random(SEED + 1)
    people = set(nights)
    plans, audit = {}, []
    for f, evaluation in enumerate(folds):
        eligible = sorted(
            s for s in people - set(evaluation)
            if len(nights[s]) == 2 and all(
                sum(map(len, thirds)) >= 6 and sum(bool(options) for options in thirds) >= 2
                for thirds, _, _, _ in nights[s].values()
            )
        )
        for r in range(N_ROSTERS):
            chosen = rng.sample(eligible, 10)
            base, added = [], {(d, a): [] for d in range(N_DRAWS) for a in ARMS}
            start_days = [1, 1, 1, 2, 2] if (f + r) % 2 == 0 else [1, 1, 2, 2, 2]
            rng.shuffle(start_days)
            for old, new, day in zip(chosen[:5], chosen[5:], start_days):
                source, source_offset, _, source_record = nights[old][day]
                other, other_offset, _, other_record = nights[old][3 - day]
                newcomer, newcomer_offset, _, newcomer_record = nights[new][3 - day]
                for _ in range(500):
                    start = rng.sample([(k, e) for k, opts in enumerate(source) for e in opts], 4)
                    unused = [[e for e in opts if (k, e) not in start] for k, opts in enumerate(source)]
                    matched = [k for k in range(3) if unused[k] and other[k] and newcomer[k]]
                    if len(matched) >= 2:
                        break
                else:
                    raise ValueError(f"Unable to sample identical fixed budgets for {old} and {new}")
                for k, e in start:
                    base.append(e)
                    audit.append(dict(fold=f+1, roster=r+1, draw=0, arm="base", subject=old,
                                      record=source_record, start=e-source_offset, third=k+1))
                for k in rng.sample(matched, 2):
                    for arm, opts, subject, record, offset in zip(
                        ARMS, (unused[k], other[k], newcomer[k]), (old, old, new),
                        (source_record, other_record, newcomer_record), (source_offset, other_offset, newcomer_offset)
                    ):
                        for d in range(N_DRAWS):
                            e = rng.choice(opts)
                            added[d, arm].append(e)
                            audit.append(dict(fold=f+1, roster=r+1, draw=d+1, arm=arm, subject=subject,
                                              record=record, start=e-offset, third=k+1))
            plans[f, r] = (base, added)
    return plans, audit


def epochs(starts):
    return np.sort(np.array(starts)[..., None] + np.arange(BLOCK_SIZE), axis=None)


def run(directory):
    start = time.perf_counter()
    folds = expected_folds()
    preset_path = directory / "protocol.json"
    locked = dict(PRESET, evaluation_folds=folds)
    if preset_path.exists():
        if json.loads(preset_path.read_text(encoding="utf-8")) != locked:
            raise ValueError("Frozen protocol differs from the runtime configuration")
    else:
        write_json(preset_path, locked)
    nights, data = nights_and_epochs(directory)
    x = np.concatenate([r[1] for r in data])
    y = np.concatenate([r[2] for r in data])
    plans, audit = design(nights, folds)
    for f, testing in enumerate(folds, 1):
        if any(row["subject"] in testing for row in audit if row["fold"] == f):
            raise ValueError("Training uses an evaluation subject")
    write_csv(directory / "blocks.csv", audit)
    write_json(directory / "data_lock.json",
               {"blocks_sha256": sha256(directory / "blocks.csv"),
                "code_sha256": sha256(Path(__file__)),
                "features": {p.name: sha256(p) for p in sorted((directory/"features").glob("*.npz"))}})
    counts = np.empty((N_FOLDS, N_ROSTERS, N_DRAWS, 3, 5, 25), dtype=np.int64)
    for f, subjects in enumerate(folds):
        ev, group = [], []
        for j, s in enumerate(subjects):
            for _, offset, n, _ in nights[s].values():
                valid = np.flatnonzero(y[offset:offset+n] >= 0)
                ev.extend(offset + valid)
                group.extend([j] * len(valid))
        ev = np.asarray(ev)
        xe, ye = x[ev], y[ev]
        by_subject = np.asarray(group) * 25 + ye.astype(np.int64) * 5
        for r in range(N_ROSTERS):
            base, added = plans[f, r]
            base = epochs(base)
            for d in range(N_DRAWS):
                extra = [epochs(added[d, a]) for a in ARMS]
                if len(base) != 1200 or any(len(e) != 600 for e in extra) or len(np.unique(
                    np.concatenate([base]+extra))) != 3000:
                    raise ValueError("Overlapping or unequal labels")
                for a, e in enumerate(extra):
                    ids = np.sort(np.r_[base, e])
                    model, center, scale, _ = fit(x[ids], y[ids], 0, 30)
                    with torch.inference_mode():
                        pred = model(torch.from_numpy((xe-center)/scale)).argmax(dim=1).numpy()
                    counts[f,r,d,a] = np.bincount(by_subject+pred, minlength=125).reshape(5,25)
            print(f"Fold {f+1}/{N_FOLDS} roster {r+1}/{N_ROSTERS} complete, {time.perf_counter()-start:.0f}s", flush=True)
        np.save(directory / f"fold{f+1}_confusions.npy", counts[f], allow_pickle=False)
    summarize(counts, directory)


def summarize(counts, directory):
    measured = macro(counts.sum(axis=-2))
    deltas = paired(measured)
    rng = np.random.default_rng(0)
    total = np.zeros((N_BOOT, 3))
    for f in range(N_FOLDS):
        flat = counts[f].reshape(-1,5,25).transpose(1,0,2).reshape(5,-1)
        for start in range(0, N_BOOT, 200):
            weights = rng.multinomial(5, [0.2]*5, size=200)
            draw = (weights@flat).reshape(200,N_ROSTERS,N_DRAWS,3,25)
            rs = rng.integers(N_ROSTERS, size=(200,N_ROSTERS,1))
            ds = rng.integers(N_DRAWS, size=(200,N_ROSTERS,N_DRAWS))
            selected = paired(macro(draw))[np.arange(200)[:,None,None], rs, ds]
            total[start:start+200] += selected.mean(axis=(1,2)) / N_FOLDS
    results = {}
    for k, name in enumerate(("other_night_minus_deepen","add_people_minus_other_night","add_people_minus_deepen")):
        d = deltas[...,k]
        results[name] = {"mean_pp": float(d.mean()),
                         "95_interval_pp": np.percentile(total[:,k],[2.5,97.5]).tolist(),
                         "fold_means_pp": d.mean(axis=(1,2)).tolist(),
                         "positive_rosters": int((d.mean(axis=2)>0).sum())}
    primary = results[PRESET["primary"]]
    low, high = primary["95_interval_pp"]
    if primary["mean_pp"]>=1 and low>0 and sum(v>0 for v in primary["fold_means_pp"])>=3:
        verdict = "external_support"
    elif high < 1:
        verdict = "below_practical_target"
    else:
        verdict = "inconclusive"
    info = {"decision": verdict, "n_fit": N_FOLDS*N_ROSTERS*N_DRAWS*3,
            "cohort_epoch_count": int(counts[:,0,0,0].sum()),
            "arm_f1": dict(zip(ARMS,measured.mean(axis=(0,1,2)).tolist())), "contrasts": results}
    write_json(directory/"summary.json", info)
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--freeze", action="store_true")
    a = p.parse_args()
    if a.freeze:
        path = a.directory / "protocol.json"
        if path.exists():
            p.error("Protocol already frozen")
        write_json(path, dict(PRESET, evaluation_folds=expected_folds()))
    else:
        torch.set_num_threads(4)
        run(a.directory)

