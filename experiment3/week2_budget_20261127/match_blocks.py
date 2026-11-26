"""Freeze label-blind cross-night blocks using the week-one candidate grid."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
WEEK1 = HERE.parent / "week1_budget_20261126"
sys.path.insert(0, str(WEEK1))
from plan import candidate_nights
from protocol import BLOCK_SIZE, KEY, SOURCES, read_csv, require, sha256, write_csv

ORDER = ("01", "04", "02", "03", "06", "07", "08", "09", "10")


def span(block):
    return range(int(block["first_epoch"]), int(block["last_epoch"]) + 1)


def keyed_epochs(block, positions):
    return [positions[block["subject_id"], block["record_id"], epoch] for epoch in span(block)]


def match_roster(roster, old, nights, positions):
    base = [r for r in old if r["arm"] == "base"]
    deepen = [r for r in old if r["arm"] == "deepen"]
    matches, added = [], []
    rng = random.Random(20261127 + int(roster))
    for subject in sorted({r["subject_id"] for r in base}):
        start = {r["record_id"] for r in base if r["subject_id"] == subject}
        require(len(start) == 1, f"Multiple starting nights for {subject}")
        starting_night, = start
        targets = sorted((r for r in deepen if r["subject_id"] == subject),
                         key=lambda r: (int(r["night_third"]), int(r["first_epoch"])))
        require(len(targets) == 2, f"Incorrect old addition count for {subject}")
        need = Counter(int(r["night_third"]) for r in targets)
        eligible = sorted((record, thirds) for record, thirds in nights[subject]
                          if record != starting_night and all(len(thirds[k - 1]) >= n for k, n in need.items()))
        require(len(eligible) == 1, f"Expected one matching other night: {roster} / {subject}")
        record, thirds = eligible[0]
        available = {i: sorted(starts) for i, starts in enumerate(thirds, 1)}
        for target in targets:
            third = int(target["night_third"])
            first = rng.choice(available[third])
            available[third].remove(first)
            last = first + BLOCK_SIZE - 1
            added.append(dict(arm="other_night",subject_id=subject,record_id=record,first_epoch=first,
                              last_epoch=last,night_third=third))
            matches.append(dict(subject_id=subject,starting_record_id=starting_night,
                                original_first_epoch=int(target["first_epoch"]),original_last_epoch=int(target["last_epoch"]),
                                other_record_id=record,other_first_epoch=first,other_last_epoch=last,night_third=third,
                                direction="later" if record > starting_night else "earlier"))
    rows = [r for block in base + added for r in keyed_epochs(block, positions)]
    require(len(base) == 20 and len(matches) == len(added) == 10, f"Wrong roster size: {roster}")
    require(len(rows) == len({r["sample_id"] for r in rows}) == 1800, f"Wrong or overlapping keys: {roster}")
    rows.sort(key=lambda row: int(row["row"]))
    return matches, base + added, [{key: row[key] for key in KEY} for row in rows]


def freeze(data, old_plan, target):
    old_lock = json.loads((old_plan / "manifest.json").read_text())
    for key,digest in old_lock["sources"].items():
        require(sha256(data / SOURCES[key]) == digest, f"Source changed: {key}")
    for name,digest in old_lock["locks"].items():
        require(sha256(old_plan/name) == digest, f"Week-one mask changed: {name}")
    roles = {r["subject_id"]: r["role"] for r in read_csv(old_plan / "subject_roles.csv")}
    pool = {s for s,role in roles.items() if role == "eligible_pool"}
    epoch_path = data / SOURCES["epoch"]
    epochs = read_csv(epoch_path)
    positions = {(r["subject_id"],r["record_id"],int(r["epoch_index"])): r for r in epochs if r["subject_id"] in pool}
    nights = candidate_nights(epoch_path,pool)
    roster05 = read_csv(old_plan/"05/blocks.csv")
    frozen05 = {r["record_id"] for r in roster05 if r["arm"]=="base" and r["subject_id"]=="SC452"}
    actual05 = {r["record_id"] for r in epochs if r["subject_id"]=="SC452"}
    require(actual05 == frozen05 == {"SC4522"}, "Roster 05 exclusion changed")
    target.mkdir(parents=True,exist_ok=False)
    for roster in ORDER:
        old = read_csv(old_plan/roster/"blocks.csv")
        matches, blocks, keys = match_roster(roster,old,nights,positions)
        members = read_csv(old_plan/roster/"evaluation_subjects.csv")
        require({r["subject_id"] for r in keys}.isdisjoint(r["subject_id"] for r in members), f"Fitting overlap: {roster}")
        require(all(roles[r["subject_id"]]=="eligible_pool" for r in keys), f"Frozen subject overlap: {roster}")
        dest=target/roster
        dest.mkdir()
        write_csv(dest/"matched_blocks.csv",matches)
        write_csv(dest/"training_blocks.csv",blocks)
        write_csv(dest/"training_epochs.csv",keys)
        write_csv(dest/"evaluation_subjects.csv",members)
        print(f"{roster}: locked 10 matched blocks, 1,800 epoch keys",flush=True)
    manifest=dict(generation_order=list(ORDER), roster_seed_rule="20261127 + roster number",
                  excluded={"05":{"subject_id":"SC452","available_record_ids":["SC4522"],"retained_in_table_one":True}},
                  first_week_plan_sha256=sha256(old_plan/"manifest.json"), epoch_index_sha256=sha256(epoch_path),
                  planner_sha256=sha256(Path(__file__)), first_week_candidate_algorithm_sha256=sha256(WEEK1/"plan.py"),
                  head_seed=old_lock["head_seed"],epochs=old_lock["epochs"],
                  locks={p.relative_to(target).as_posix():sha256(p) for p in sorted(target.rglob("*.csv"))})
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8",newline="\n")

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data",type=Path,required=True)
    cli.add_argument("--old-plan",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    freeze(args.data,args.old_plan,args.output)
