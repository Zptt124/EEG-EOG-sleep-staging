"""Freeze paired label budgets using only subject, record and epoch keys."""

import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

from protocol import ARMS, BLOCK_SIZE, SOURCES, read_csv, require, sha256, write_csv


def candidate_nights(path, pool):
    epochs = defaultdict(list)
    for row in read_csv(path):
        if row["subject_id"] in pool:
            epochs[row["subject_id"], row["record_id"]].append(int(row["epoch_index"]))
    nights = defaultdict(list)
    for (subject, record), values in sorted(epochs.items()):
        values.sort()
        require(len(set(values)) == len(values), f"Duplicate epochs in {record}")
        first, last = values[0], values[-1]
        thirds = [[], [], []]
        for _, run in groupby(enumerate(values), key=lambda item: item[1] - item[0]):
            consecutive = [epoch for _, epoch in run]
            for start in consecutive[::BLOCK_SIZE][:len(consecutive) // BLOCK_SIZE]:
                third = min(2, 3 * (2 * (start - first) + BLOCK_SIZE - 1) // (2 * (last - first + 1)))
                thirds[third].append(start)
        if all(thirds) and sum(map(len, thirds)) >= 6:
            nights[subject].append((record, thirds))
    return nights


def make_plan(data, output):
    epoch_path, split_path, pretrain_path, dev_path = [data / SOURCES[key] for key in SOURCES]
    split = read_csv(split_path)
    groups = {kind: {r["subject_id"] for r in split if r["split"] == kind}
              for kind in ("train", "val", "test")}
    roles = defaultdict(set)
    for row in read_csv(pretrain_path):
        roles[row["pretrain_role"]].add(row["subject_id"])
    pool, held = roles["pretrain_train"], roles["heldout_val"]
    require(tuple(map(len, groups.values())) == (54, 8, 16), "Unexpected team split")
    require(len(set.union(*groups.values())) == 78 and len(split) == 78, "Overlapping team split")
    require(len(pool) == 46 and len(held) == 8 and pool | held == groups["train"], "Unexpected W4 split")
    require({r["subject_id"] for r in read_csv(dev_path)} == groups["val"], "Dev reference mismatch")
    nights = candidate_nights(epoch_path, pool)
    require(len(nights) >= 10, "Too few eligible subjects")
    output.mkdir(parents=True, exist_ok=False)
    for number in [6, 1, 2, 3, 4, 5, 7, 8, 9, 10]:
        rng = random.Random(20261126 + number)
        order = rng.sample(sorted(nights), 10)
        blocks = []
        positions = rng.choices([(0, 1), (0, 2), (1, 2)], k=5)
        for index, subject in enumerate(order):
            record, thirds = rng.choice(nights[subject])
            added = [(third, rng.choice(thirds[third])) for third in positions[index % 5]]
            kind = "deepen" if index < 5 else "add_people"
            choices = [(kind, *item) for item in added]
            if index < 5:
                remaining = [(third, start) for third, starts in enumerate(thirds) for start in starts
                             if (third, start) not in added]
                choices.extend(("base", *item) for item in rng.sample(remaining, 4))
            for arm, third, start in choices:
                blocks.append(dict(arm=arm, subject_id=subject, record_id=record, first_epoch=start,
                                   last_epoch=start + BLOCK_SIZE - 1, night_third=third + 1))
        require(Counter(r["arm"] for r in blocks) == dict(zip(ARMS, (20, 10, 10))), "Block count mismatch")
        out = output / f"{number:02d}"
        out.mkdir()
        write_csv(out / "blocks.csv", sorted(blocks, key=lambda r: (r["arm"], r["record_id"], r["first_epoch"])))
        write_csv(out / "evaluation_subjects.csv",
                  [dict(cohort=kind, subject_id=s) for kind, subjects in
                   (("pool36", pool - set(order)), ("dev8", groups["val"])) for s in sorted(subjects)])
    write_csv(output / "subject_roles.csv",
              [dict(role=kind, subject_id=subject) for kind, subjects in
               (("eligible_pool", pool), ("held_out", held), ("dev", groups["val"]), ("sealed_test", groups["test"]))
               for subject in sorted(subjects)])
    manifest = dict(seed_rule="20261126 + roster number", head_seed=0, epochs=30,
                    eligible_subjects=len(nights), ineligible_subjects=sorted(pool - nights.keys()),
                    eligible_nights=sum(map(len, nights.values())),
                    sources={key: sha256(data / name) for key, name in SOURCES.items()},
                    locks={p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*.csv"))})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_plan(args.data, args.output)
