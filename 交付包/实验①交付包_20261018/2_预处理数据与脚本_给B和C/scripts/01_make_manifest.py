"""Step 1: build dataset_manifest.csv from filenames only (no EDF reading).

Parses RECORDS (PSG files) and SHA256SUMS.txt (hypnogram files) from
PhysioNet sleep-edfx 1.0.0, pairs each night's PSG with its hypnogram,
and writes split/dataset_manifest.csv.

Rule (team doc): subject = first 5 chars of filename, night = 6th char.
e.g. SC4001E0-PSG.edf -> subject SC400, night 1, record SC4001.
"""
import sys
from pathlib import Path

import pandas as pd

from common import RAW, SPLIT

RECORDS = RAW / "RECORDS"
CHECKSUMS = RAW / "SHA256SUMS.txt"
OUT = SPLIT / "dataset_manifest.csv"


def load_psg_records() -> list[str]:
    stems = []
    with open(RECORDS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "sleep-cassette" not in line:
                continue
            fname = line.split("/")[-1]
            if fname.endswith("-PSG.edf"):
                stems.append(fname[: -len("-PSG.edf")])   # e.g. SC4001E0
    return sorted(stems)


def load_hypnograms() -> dict[str, str]:
    """record_id (SC4001, first 6 chars) -> hypnogram filename.

    Hypnogram names are <record_id><2-letter suffix>-Hypnogram.edf
    (e.g. SC4001EC, SC4512EW) — the 2-letter suffix is irregular, so pair by
    the 6-char record prefix.
    """
    hyps = []
    with open(CHECKSUMS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "sleep-cassette" not in line or "Hypnogram" not in line:
                continue
            fname = line.split("/")[-1]
            record_id = fname[:6]                          # e.g. SC4001
            hyps.append((record_id, fname))
    return {record_id: fname for record_id, fname in hyps}


def main():
    stems = load_psg_records()
    hyps = load_hypnograms()
    SPLIT.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in stems:
        # record_id = first 6 chars (subject 5 + night 1); hypnogram shares that prefix
        record_id = stem[:6]
        hyp_fname = hyps.get(record_id, None)
        rows.append({
            "record_id": record_id,
            "subject_id": record_id[:5],
            "night": record_id[5],
            "psg_file": f"{stem}-PSG.edf",
            "hyp_file": hyp_fname,
        })
    df = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    assert df["hyp_file"].notna().all(), "some nights lack a hypnogram"
    df.to_csv(OUT, index=False)

    n_subjects = df["subject_id"].nunique()
    n_records = len(df)
    nights_per_subj = df.groupby("subject_id")["night"].nunique()
    print(f"records: {n_records}  subjects: {n_subjects}")
    print(f"subjects with 1 night: {(nights_per_subj == 1).sum()}, "
          f"2 nights: {(nights_per_subj == 2).sum()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
