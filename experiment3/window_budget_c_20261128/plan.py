"""Lock Experiment C's label-blind candidates before reading any hypnograms."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from math import ceil, floor
from pathlib import Path
import random
from urllib.request import Request, urlopen

import xlrd

from download import BASES, OLD, OUT, MANIFEST, hash_file, read_rows, records

ARMS = ("deepen", "other_night", "add_people")
BLOCK = 60
WINDOW_HOURS = 9
SEED = 2026112801


def write_rows(name, rows):
    path = OUT / name
    with path.with_suffix(".tmp").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def time_string(seconds):
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def read_header(name):
    source = OUT / "source"
    path = source / "raw" / name
    if path.exists():
        with path.open("rb") as file:
            return file.read(256)
    cache = source / "headers"
    cache.mkdir(exist_ok=True)
    path = cache / name
    if path.exists():
        return path.read_bytes()
    for base in BASES[::-1] * 2:
        try:
            request = Request(base + "sleep-cassette/" + name, headers={"Range": "bytes=0-255"})
            with urlopen(request, timeout=30) as response:
                data = response.read(256)
            if len(data) != 256:
                raise ValueError("Short EDF header")
            path.write_bytes(data)
            return data
        except (OSError, ValueError) as error:
            print(f"Header retry {name}: {error}", flush=True)
    raise RuntimeError(f"Cannot read {name}")


def recording_window(row, lights):
    record = row["record_id"]
    header = read_header(row["psg_file"])
    h, m, s = map(int, header[176:184].decode("ascii").split("."))
    recording_start = 3600 * h + 60 * m + s
    duration = int(header[236:244]) * float(header[244:252])
    if duration <= 0:
        raise ValueError(f"Invalid duration for {record}")
    marker = lights.get((int(record[3:5]), int(row["night"])))
    info = dict(record_id=record, subject_id=row["subject_id"], psg_file=row["psg_file"],
                edf_start=time_string(recording_start), duration_sec=duration,
                lights_off="", lights_off_offset_sec="", first_epoch="", eligible_blocks=0)
    if marker is None:
        return info, []
    offset = (marker - recording_start) % 86400
    first = ceil(offset / 30)
    end = min(floor((offset + WINDOW_HOURS * 3600) / 30), floor(duration / 30))
    grid = [(e, 1 + (e - first) // (3 * 120))
            for e in range(first, end - BLOCK + 1, BLOCK)]
    info.update(lights_off=time_string(marker), lights_off_offset_sec=offset,
                first_epoch=first, eligible_blocks=len(grid))
    return info, grid


def main():
    lock_path = OUT / "protocol.json"
    if lock_path.exists():
        raise ValueError("Experiment C windows are already locked")
    metadata = OUT / "source/SC-subjects.xls"
    hashes = dict((name, digest) for digest, name in
                  (line.split() for line in (metadata.parent / "SHA256SUMS.txt").read_text().splitlines()))
    if hash_file(metadata) != hashes[metadata.name]:
        raise ValueError("Lights-off spreadsheet checksum mismatch")
    sheet = xlrd.open_workbook(metadata).sheet_by_index(0)
    if sheet.row_values(0)[4] != "LightsOff":
        raise ValueError("Unexpected spreadsheet columns")
    lights = {(int(s), int(n)): round(t * 86400)
              for s, n, _, _, t in map(sheet.row_values, range(1, sheet.nrows))
              if isinstance(t, float) and 0 <= t < 1}
    psg_rows = records()
    with ThreadPoolExecutor(max_workers=12) as pool:
        features = list(pool.map(lambda row: recording_window(row, lights), psg_rows))
    windows = {info["record_id"]: grid for info, grid in features}
    old_plan = json.loads((OLD / "plan.json").read_text())
    for name, digest in old_plan["locked_files"].items():
        if hash_file(OLD / name) != digest:
            raise ValueError(f"Original plan file differs: {name}")
    old_blocks = read_rows(OLD / "blocks.csv")
    anchors = defaultdict(list)
    for block in old_blocks:
        if block["arm"] == "base":
            anchors[int(block["roster"]), block["record_id"]].append(int(block["first_epoch"]))
    cells = defaultdict(set)
    for row in read_rows(OLD / "candidate_blocks.csv"):
        cells[int(row["roster"])].add(
            (row["arm"], row["subject_id"], row["record_id"], int(row["night_third"])))
    candidates, selected, roster_status = [], [], []
    for roster, strata in sorted(cells.items()):
        if len(strata) != 30:
            raise ValueError(f"Expected ten locked strata per arm for roster {roster}")
        options = {}
        for arm, subject, record, third in sorted(strata):
            base = anchors[roster, record]
            options[arm, subject, record, third] = [
                e for e, section in windows[record]
                if section == third and all(abs(e - a) >= BLOCK for a in base)
            ]
        unavailable = ["|".join(map(str, k)) for k, epochs in options.items() if not epochs]
        paired = not unavailable
        roster_status.append(dict(roster=roster, paired=int(paired), unavailable=";".join(unavailable)))
        rng = random.Random(SEED + roster)
        for (arm, subject, record, third), epochs in options.items():
            cell = [dict(roster=roster, arm=arm, subject_id=subject, record_id=record,
                         night_third=third, first_epoch=e, last_epoch=e + BLOCK - 1,
                         paired_roster=int(paired)) for e in epochs]
            candidates.extend(cell)
            if paired:
                selected.extend(dict(draw=draw, **rng.choice(cell))
                                for draw in range(1, old_plan["draws_per_roster"] + 1))
    if not selected:
        raise ValueError("No complete rosters for the label-blind comparison")
    anchors_rows = [r for r in old_blocks if r["arm"] == "base"]
    write_rows("recording_metadata.csv", [info for info, _ in features])
    write_rows("candidate_blocks.csv", candidates)
    write_rows("paired_rosters.csv", roster_status)
    write_rows("selected_blocks.csv", selected)
    write_rows("anchor_blocks.csv", anchors_rows)
    names = ["recording_metadata.csv", "candidate_blocks.csv", "paired_rosters.csv",
             "selected_blocks.csv", "anchor_blocks.csv"]
    lock = dict(
        window="First 9 hours after LightsOff (first complete 30-second epoch at or after marker). "
               "Use non-overlapping consecutive 60-epoch blocks, retaining only full blocks in PSG "
               "and within the 9-hour window. Three 3-hour strata on the resulting grid.",
        comparison="Same original five-person 1200-label historical anchor, subjects, recordings, "
                   "per-record stratum assignments, and fixed development evaluation epochs. "
                   "Exclude any candidate that intersects previously labeled anchor epochs.",
        draw="Independently draw one eligible block per arm/record/third, across five draws.",
        missing_labels="Keep fixed blind draws; exact-600 primary comparison uses only draw pairs "
                       "with five-class labels at every newly selected epoch in all arms. "
                       "Sensitivity includes all draws and masks unscorable selected epochs.",
        contrast="add_people minus deepen, and (new window contrast) minus (old window contrast).",
        seed=SEED, head_seed=0, head_epochs=30, bootstrap_seed=0, bootstrap_draws=10000,
        source="https://physionet.org/content/sleep-edfx/1.0.0/",
        lights_off_sha256=hash_file(metadata), manifest_sha256=hash_file(MANIFEST),
        original_plan_sha256=hash_file(OLD / "plan.json"),
        evaluation_sha256=hash_file(OLD / "evaluation_index.npz"),
        source_sha256={p.name: hash_file(p) for p in (Path(__file__), Path(__file__).with_name("download.py"))},
        files={name: hash_file(OUT / name) for name in names},
        paired_rosters=[r["roster"] for r in roster_status if r["paired"]],
    )
    pending = lock_path.with_suffix(".tmp")
    pending.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    pending.replace(lock_path)
    print(f"Locked {len(lock['paired_rosters'])}/{len(roster_status)} rosters, "
          f"{len(candidates)} candidate rows, {len(selected)} selected blocks", flush=True)


if __name__ == "__main__":
    main()
