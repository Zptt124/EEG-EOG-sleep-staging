"""Run the paired development replication using the frozen Week 1 CPU head."""
import argparse
from collections import Counter
import json
from itertools import combinations
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "week1_budget_20261126"))
from plan import candidate_nights
from train import fit, features, labels
from protocol import BLOCK_SIZE, read_csv, require, sha256, write_csv

ARMS = ("deepen", "other_night", "add_people")
FEATURE_HASH = "392122c602030d8dcfa903a8d5a5f55fd582f0e09dc6d086248c86bac4267bdf"
SEED = 20261127
N_ROSTERS = 30
N_DRAWS = 5
N_BOOT = 10000


def write_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def choose_matched_blocks(rng, original, newcomer):
    # First commit four blocks from a starting night. Only then draw new blocks.
    old_person, old_nights = original
    new_person, new_nights = newcomer
    candidates = [
        (start, other, new)
        for start, other in ((a, b) for a in old_nights for b in old_nights if a[0] != b[0])
        for new in new_nights
    ]
    for _ in range(500):
        start, other, new = rng.choice(candidates)
        base = rng.sample([(k, e) for k, epochs in enumerate(start[1]) for e in epochs], 4)
        remaining = [[e for e in epochs if (k, e) not in base] for k, epochs in enumerate(start[1])]
        thirds = [k for k in range(3) if remaining[k] and other[1][k] and new[1][k]]
        if len(thirds) >= 2:
            return old_person, new_person, start, other, new, base, remaining, rng.sample(thirds, 2)
    raise RuntimeError(f"No feasible matched starting mask for {old_person} and {new_person}")


def plan_run(index, rows, out):
    pool = sorted({r["subject_id"] for r in rows if r["role"] == "pretrain_train"})
    dev = sorted({r["subject_id"] for r in rows if r["role"] == "dev"})
    require(len(pool) == 46 and len(dev) == 8 and not (set(pool) & set(dev)), "Unexpected pool or dev membership")
    reserved = sorted(random.Random(SEED).sample(pool, 8))
    eligible = set(pool) - set(reserved)
    nights = candidate_nights(index, eligible)
    dual_night = sorted(s for s, n in nights.items() if len(n) >= 2)
    require(len(eligible) == 38 and len(dual_night) >= 5 and len(nights) >= 10, "Insufficient eligible candidates")
    rng = random.Random(SEED + 1)
    blocks, candidates, roster_people = [], [], []
    for roster in range(1, N_ROSTERS + 1):
        old = rng.sample(dual_night, 5)
        new = rng.sample(sorted(set(nights) - set(old)), 5)
        roster_people.append({"roster": roster, "base": old, "new": new})
        for old_s, new_s in zip(old, new):
            old_s, new_s, start, other, new_night, base, remaining, thirds = choose_matched_blocks(
                rng, (old_s, nights[old_s]), (new_s, nights[new_s])
            )
            for k, epoch in base:
                blocks.append(dict(roster=roster, draw=0, arm="base", subject_id=old_s,
                                   record_id=start[0], first_epoch=epoch, night_third=k + 1))
            for k in thirds:
                lists = (remaining[k], other[1][k], new_night[1][k])
                refs = ((old_s, start[0]), (old_s, other[0]), (new_s, new_night[0]))
                for arm, options, (subject, record) in zip(ARMS, lists, refs):
                    candidates.extend(dict(roster=roster, arm=arm, subject_id=subject,
                                           record_id=record, first_epoch=e, night_third=k + 1) for e in options)
                    for draw in range(1, N_DRAWS + 1):
                        blocks.append(dict(roster=roster, draw=draw, arm=arm, subject_id=subject,
                                           record_id=record, first_epoch=rng.choice(options), night_third=k + 1))
    sets = [set(r["base"]) for r in roster_people]
    unions = [set(r["base"] + r["new"]) for r in roster_people]
    low_overlap = [0]
    while len(low_overlap) < 6:
        low_overlap.append(min((i for i in range(N_ROSTERS) if i not in low_overlap),
                               key=lambda i: (sum(len(sets[i] & sets[j]) for j in low_overlap), i)))

    def overlaps(items):
        pairs = [(len(a & b), len(a & b) / len(a | b)) for a, b in combinations(items, 2)]
        return dict(shared_mean=float(np.mean([x[0] for x in pairs])), shared_max=max(x[0] for x in pairs),
                    jaccard_mean=float(np.mean([x[1] for x in pairs])))

    for name, records in (("blocks.csv", blocks), ("candidate_blocks.csv", candidates)):
        write_csv(out / name, records)
    lock = dict(n_rosters=N_ROSTERS, draws_per_roster=N_DRAWS, block_epochs=BLOCK_SIZE, head_epochs=30,
                primary_head_seed=0, extra_head_seeds=[1, 2], extra_seed_rosters=[1, 2, 3],
                bootstrap_draws=N_BOOT, bootstrap_seed=0, thresholds_pp=dict(positive=0, practical=1),
                selection_seed=SEED, roster_seed=SEED + 1, feature_info_sha256=FEATURE_HASH,
                source_sha256={str(p.name): sha256(p) for p in
                               [Path(__file__), Path(fit.__code__.co_filename),
                                Path(candidate_nights.__code__.co_filename)]},
                python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                version_notes="Sleep-EDF historical label-trimmed cache; targets unseen by block sampler. "
                              "Independent block draws with replacement between replicates.",
                dev8=dev, pretrain_eval8=reserved, roster_people=roster_people,
                low_overlap=[i + 1 for i in low_overlap],
                overlap=dict(base=overlaps(sets), all_training=overlaps(unions),
                             low_base=overlaps([sets[i] for i in low_overlap]),
                             low_all=overlaps([unions[i] for i in low_overlap]),
                             base_frequency=dict(Counter(s for r in roster_people for s in r["base"])),
                             newcomer_frequency=dict(Counter(s for r in roster_people for s in r["new"]))),
                eligible_n=len(eligible), eligible_two_nights=len(dual_night),
                locked_files={name: sha256(out / name) for name in ("blocks.csv", "candidate_blocks.csv")})
    write_json(out / "plan.json", lock)
    return blocks, lock


def macro(confusion):
    matrix = confusion.reshape(*confusion.shape[:-1], 5, 5)
    den = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    diag = np.diagonal(matrix, axis1=-2, axis2=-1)
    class_f1 = np.divide(200.0 * diag, den, out=np.zeros_like(den, dtype=float), where=den > 0)
    return class_f1.mean(axis=-1)


def fit_all(rows, arrays, blocks, lock, out):
    # Evaluate in the same subject/epoch order for every arm, draw and seed.
    eval_people = lock["dev8"] + lock["pretrain_eval8"]
    order = {s: i for i, s in enumerate(eval_people)}
    eval_ids = sorted((i for i, r in enumerate(rows) if r["subject_id"] in order),
                      key=lambda i: (order[rows[i]["subject_id"]], i))
    eval_x = features(arrays, eval_ids)
    eval_y = labels(rows, eval_ids)
    eval_subjects = np.array([order[rows[i]["subject_id"]] for i in eval_ids])
    np.savez_compressed(out / "evaluation_index.npz", sample_id=[rows[i]["sample_id"] for i in eval_ids],
                        truth=eval_y, subject_index=eval_subjects, subjects=eval_people)
    index = {(r["subject_id"], r["record_id"], int(r["epoch_index"])): i
             for i, r in enumerate(rows) if r["role"] == "pretrain_train" and r["subject_id"] not in order}
    grouped = {}
    for b in blocks:
        key = (b["roster"], b["draw"], b["arm"])
        first = b["first_epoch"]
        ids = [index[b["subject_id"], b["record_id"], e] for e in range(first, first + BLOCK_SIZE)]
        grouped.setdefault(key, []).extend(ids)

    counts = np.zeros((3, N_ROSTERS, N_DRAWS, 3, 16, 25), dtype=np.int64)
    scores = []
    for roster in range(1, N_ROSTERS + 1):
        staged = out / f"roster_{roster:02d}.tmp"
        finished = staged.with_suffix(".npz")
        if finished.exists():
            with np.load(finished, allow_pickle=False) as saved:
                counts[:, roster - 1] = saved["counts"]
                scores.extend(json.loads(str(saved["scores"])))
            continue
        base = grouped[roster, 0, "base"]
        predictions, roster_scores = {}, []
        expected = lock["roster_people"][roster - 1]
        require(len(base) == len(set(base)) == 1200, "Malformed common anchor")
        for draw in range(1, N_DRAWS + 1):
            extra = [grouped[roster, draw, arm] for arm in ARMS]
            require(len(set(base + sum(extra, []))) == 3000, "Overlapping or incomplete arm budgets")
            for arm_ix, (arm, added) in enumerate(zip(ARMS, extra)):
                require(len(added) == 600 and len(set(added)) == 600, "Incorrect added labels")
                expected_subjects = expected["new" if arm == "add_people" else "base"]
                require({rows[i]["subject_id"] for i in added} == set(expected_subjects), "Wrong arm subjects")
                ids = sorted(base + added)
                x, y = features(arrays, ids), labels(rows, ids)
                for seed in range(3 if roster <= 3 else 1):
                    model, mean, std, curve = fit(x, y, seed, 30)
                    with torch.inference_mode():
                        y_pred = model(torch.from_numpy((eval_x - mean) / std)).argmax(dim=1).numpy()
                    conf = np.bincount(eval_subjects * 25 + eval_y * 5 + y_pred, minlength=400).reshape(16, 25)
                    counts[seed, roster - 1, draw - 1, arm_ix] = conf
                    predictions[f"{draw}_{arm}_{seed}"] = y_pred.astype(np.uint8)
                    roster_scores.append(dict(roster=roster, draw=draw, arm=arm, seed=seed,
                                              macro_f1=macro(conf.sum(axis=0)).item(),
                                              dev8=macro(conf[:8].sum(axis=0)).item(),
                                              pretrain_eval8=macro(conf[8:].sum(axis=0)).item(),
                                              updates=curve[-1]["cumulative_updates"],
                                              added_stages=np.bincount(labels(rows, added), minlength=5).tolist()))
        with staged.open("wb") as file:
            np.savez_compressed(file, counts=counts[:, roster - 1], scores=json.dumps(roster_scores), **predictions)
        staged.replace(finished)
        scores.extend(roster_scores)
        print(f"Roster {roster:02d}/{N_ROSTERS} complete | {len(scores)}/540 fits", flush=True)
    write_csv(out / "scores.csv", scores)
    return counts


def paired(values):
    return np.stack([values[..., 1] - values[..., 0], values[..., 2] - values[..., 1],
                     values[..., 2] - values[..., 0]], axis=-1)


def summarize(counts, lock, out):
    primary = counts[0]
    initial = paired(macro(primary.sum(axis=-2)))
    rng = np.random.default_rng(0)
    boot = np.empty((N_BOOT, 3, 3))
    flat = primary.reshape(-1, 16, 25).transpose(1, 0, 2).reshape(16, -1)
    for lo in range(0, N_BOOT, 100):
        n = min(100, N_BOOT - lo)
        rs = rng.integers(N_ROSTERS, size=(n, N_ROSTERS, 1))
        bs = rng.integers(N_DRAWS, size=(n, N_ROSTERS, N_DRAWS))
        eval_w = rng.multinomial(8, [1 / 8] * 8, size=(n, 2)).reshape(n, 16)
        for j, sl in enumerate((slice(None), slice(0, 8), slice(8, 16))):
            weighted = (eval_w[:, sl] @ flat[sl]).reshape(n, N_ROSTERS, N_DRAWS, 3, 25)
            cells = paired(macro(weighted))
            boot[lo:lo + n, j] = cells[np.arange(n)[:, None, None], rs, bs].mean(axis=(1, 2))

    result = {}
    for j, (cohort, sl) in enumerate((("combined", slice(None)), ("dev8", slice(0, 8)),
                                      ("pretrain_eval8", slice(8, 16)))):
        cells = paired(macro(primary[..., sl, :].sum(axis=-2)))
        roster_values = cells.mean(axis=1)
        low_ix = np.asarray(lock["low_overlap"]) - 1
        result[cohort] = {}
        for k, name in enumerate(("other_minus_deepen", "new_minus_other", "new_minus_deepen")):
            result[cohort][name] = dict(mean_pp=float(cells[..., k].mean()),
                                        interval_95_pp=np.percentile(boot[:, j, k], [2.5, 97.5]).tolist(),
                                        positive_rosters=int((roster_values[:, k] > 0).sum()),
                                        roster_sd_pp=float(roster_values[:, k].std(ddof=1)),
                                        within_roster_block_rms_sd_pp=float(np.sqrt(
                                            cells[..., k].var(axis=1, ddof=1).mean())),
                                        low_overlap_mean_pp=float(roster_values[low_ix, k].mean()),
                                        roster_deltas_pp=roster_values[:, k].tolist())
    p = result["combined"]["other_minus_deepen"]
    low, high = p["interval_95_pp"]
    if high < 1:
        verdict = "stop_practical_average_gain"
    elif low > 0 and p["mean_pp"] >= 1 and p["low_overlap_mean_pp"] > 0:
        verdict = "advance_to_confirmation"
    else:
        verdict = "inconclusive"
    seeded = paired(macro(counts[:, :3].sum(axis=-2)))[..., 0]
    result["seed_check"] = dict(per_seed_mean_pp=seeded.mean(axis=(1, 2)).tolist(),
                                max_absolute_change_vs_primary_pp=float(np.abs(seeded[1:] - seeded[:1]).max()))
    result["decision"] = verdict
    result["fits"] = 540
    require(np.allclose(initial.mean(axis=(0, 1)), [d["mean_pp"] for d in result["combined"].values()]),
            "Pooled contrasts inconsistent")
    write_json(out / "summary.json", result)
    print(json.dumps({"primary": p, "decision": verdict}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    index = args.bundle / "index.csv"
    info_path = args.bundle / "features_info.json"
    require(sha256(info_path) == FEATURE_HASH, "Unexpected published feature receipt")
    info = json.loads(info_path.read_text())
    require(sha256(index) == info["index_sha256"], "Unexpected cached index")
    torch.set_num_threads(4)
    arrays = []
    for name in ("eeg_std", "eog_std"):
        path = args.bundle / f"{name}.npy"
        require(sha256(path) == info["files"][name]["sha256"], f"Feature checksum failure: {name}")
        arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
    rows = read_csv(index)
    require(all(a.shape == (len(rows), 192) and a.dtype == np.float32 for a in arrays), "Feature shape mismatch")
    if (out / "plan.json").exists():
        lock = json.loads((out / "plan.json").read_text())
        for name, digest in lock["locked_files"].items():
            require(sha256(out / name) == digest, f"Changed frozen plan: {name}")
        for p in (Path(__file__), Path(fit.__code__.co_filename), Path(candidate_nights.__code__.co_filename)):
            require(sha256(p) == lock["source_sha256"][p.name], f"Changed execution source: {p.name}")
        blocks = read_csv(out / "blocks.csv")
        blocks = [{**b, **{k: int(b[k]) for k in ("roster", "draw", "first_epoch", "night_third")}} for b in blocks]
    else:
        blocks, lock = plan_run(index, rows, out)
    print(f"Plan locked in {time.perf_counter() - start:.1f}s | 16 fixed evaluators | 540 planned fits", flush=True)
    counts = fit_all(rows, arrays, blocks, lock, out)
    print(f"All fits complete in {time.perf_counter() - start:.1f}s | starting paired bootstrap", flush=True)
    summarize(counts, lock, out)
    print(f"Total elapsed {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

