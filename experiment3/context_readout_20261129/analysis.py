"""Predeclared roster and development-subcohort bootstrap for the CD primary contrasts."""

import argparse
import json
from pathlib import Path

import numpy as np


def macro(conf):
    denominator = conf.sum(axis=-2) + conf.sum(axis=-1)
    numerator = 200.0 * np.diagonal(conf, axis1=-2, axis2=-1)
    return np.divide(numerator, denominator, where=denominator > 0,
                     out=np.zeros_like(numerator)).mean(axis=-1)


def decision(point, interval):
    low, high = interval
    if low >= 1:
        return "at_least_1_pp"
    if point >= 1 and low > 0:
        return "follow_up"
    if low >= -1 and high <= 1:
        return "inside_practical_band"
    if high < 1:
        return "positive_1_pp_excluded_decline" if high < 0 else "positive_1_pp_excluded"
    return "magnitude_unresolved"


def evaluate(conf, members, paired, seed=0, repetitions=10000):
    """Input axes: roster, policy (C: P/R; D: singleton), draw, head/arm, subject, truth, predicted."""
    if conf.ndim != 7 or conf.shape[3:] != (2, 16, 5, 5) or np.any(conf < 0):
        raise ValueError("Invalid confusion count axes")
    n, policies, repeats = conf.shape[:3]
    if (policies, repeats) != ((2, 5) if paired else (1, 10)):
        raise ValueError("Unexpected policy or draw axes")
    if len(members) != n or any(len(set(row)) != 5 for row in members):
        raise ValueError("Each roster must have five distinct anchor participants")
    people = sorted(set().union(*map(set, members)))
    membership = np.asarray([[p in row for p in people] for row in members], dtype=float) / 5
    rng = np.random.default_rng(seed)
    base = macro(conf.sum(axis=-3))
    roster_effects = (base[..., 1] - base[..., 0]).mean(axis=(1, 2))
    point = float(roster_effects.mean())
    flat = np.moveaxis(conf, -3, 0).reshape(16, -1).astype(float)
    ordinary = np.empty(repetitions)
    clustered = np.empty(repetitions)
    for start in range(0, repetitions, 64):
        m = min(64, repetitions - start)
        indices = np.arange(m)[:, None, None, None]
        subject_weights = rng.multinomial(8, [1 / 8] * 8, size=(m, 2)).reshape(m, 16)
        scores = macro((subject_weights @ flat).reshape(m, n, policies, repeats, 2, 5, 5))
        sampled_rosters = rng.integers(n, size=(m, n))
        draws0 = rng.integers(repeats, size=(m, n, 1 if paired else policies, repeats))
        draws1 = draws0 if paired else rng.integers(repeats, size=draws0.shape)
        policies_ix = np.arange(policies)[None, None, :, None]
        sampled = sampled_rosters[..., None, None]
        effects = scores[indices, sampled, policies_ix, draws1, 1] - scores[indices, sampled, policies_ix, draws0, 0]
        ordinary[start:start + m] = effects.mean(axis=(1, 2, 3))
        # Cross-membership sensitivity: each roster gets the mean of independent Exp(1)
        # multipliers for its five anchor participants. Keep each roster once, with a
        # shared normalized weight across its two arms and existing subject/draw resampling.
        weights = rng.exponential(size=(m, len(people))) @ membership.T
        weights /= weights.sum(axis=1, keepdims=True)
        fixed = np.arange(n)[None, :, None, None]
        effects = scores[indices, fixed, policies_ix, draws1, 1] - scores[indices, fixed, policies_ix, draws0, 0]
        clustered[start:start + m] = (weights * effects.mean(axis=(2, 3))).sum(axis=1)
    interval = np.percentile(ordinary, [2.5, 97.5]).tolist()
    leave_one = []
    for i, person in enumerate(people):
        keep = membership[:, i] == 0
        leave_one.append(dict(subject_id=person, rosters=int(keep.sum()),
                              mean_pp=float(roster_effects[keep].mean()) if keep.any() else None))
    leave_values = [row["mean_pp"] for row in leave_one if row["mean_pp"] is not None]
    return dict(mean_pp=point, ci95_pp=interval, decision=decision(point, interval),
                sensitivity_multi_membership_ci95_pp=np.percentile(clustered, [2.5, 97.5]).tolist(),
                leave_one_anchor_subject=leave_one, repetitions=repetitions, seed=seed,
                leave_one_range_pp=[min(leave_values), max(leave_values)] if leave_values else None,
                roster_effects_pp=roster_effects.tolist(),
                head_means_pp=base.mean(axis=(0, 2)).tolist(),
                subcohort_effects_pp=[float((s[..., 1] - s[..., 0]).mean())
                                      for cohort in (slice(0, 8), slice(8, 16))
                                      for s in (macro(conf[..., cohort, :, :].sum(axis=-3)),)],
                policy_means_pp=(base[..., 1] - base[..., 0]).mean(axis=(0, 2)).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with counts and anchor_members")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=("C", "D"), required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        result = evaluate(data["counts"], data["anchor_members"], args.experiment == "C")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
