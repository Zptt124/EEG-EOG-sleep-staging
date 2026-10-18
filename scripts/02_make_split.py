"""Step 2: generate and publish split_subjects.csv — the group's FIRST deliverable.

Subject-wise split of the 78 SC subjects into train/val/test (~54/8/16),
stratified by age group (optional rule in the team doc), fixed seed recorded
in the file. Only uses filenames + the age table; no EDF reading.

Output format (team doc): one row per subject:
    subject_id, record_ids, split, seed
"""
import sys

import numpy as np
import pandas as pd

from common import SPLIT, SPLIT_SEED, SPLIT_COUNTS, RAW

MANIFEST = SPLIT / "dataset_manifest.csv"
AGE_XLS = RAW / "SC-subjects.xls"
OUT = SPLIT / "split_subjects.csv"

AGE_BINS = [0, 30, 50, 200]      # young / middle / old


def load_ages() -> dict[str, int]:
    """subject_id -> age, from PhysioNet SC-subjects.xls (subject index 0..77 = SC400..SC482)."""
    import xlrd
    wb = xlrd.open_workbook(AGE_XLS)
    sh = wb.sheet_by_index(0)
    assert sh.cell_value(0, 0) == "subject" and sh.cell_value(0, 2) == "age"
    ages = {}
    for r in range(1, sh.nrows):
        idx = int(sh.cell_value(r, 0))
        ages[f"SC4{idx:02d}"] = int(sh.cell_value(r, 2))
    return ages


def stratified_split(subjects: list[str], ages: dict[str, int], rng, counts=None) -> dict[str, list[str]]:
    """Assign subjects to split by age-group strata, proportional to target counts.

    Within each age bin: shuffle with the rng, then take n_val/n_test
    proportionally (round-half-up), rest goes to train. Any rounding drift is
    corrected afterwards by moving subjects out of over-filled splits.
    """
    counts = counts or SPLIT_COUNTS
    bin_of = {s: int(pd.cut([ages[s]], AGE_BINS, right=False, labels=False)[0]) for s in subjects}
    groups: dict[int, list[str]] = {}
    for s in subjects:
        groups.setdefault(bin_of[s], []).append(s)

    total_val, total_test = counts["val"], counts["test"]
    n = len(subjects)
    split_of: dict[str, str] = {}
    for members in groups.values():
        rng.shuffle(members)
        n_val = min(len(members), max(0, int(round(len(members) / n * total_val))))
        n_test = min(len(members) - n_val, max(0, int(round(len(members) / n * total_test))))
        for i, s in enumerate(members):
            split_of[s] = "val" if i < n_val else ("test" if i < n_val + n_test else "train")

    out = {k: [s for s in subjects if split_of[s] == k] for k in ("train", "val", "test")}

    # correct rounding drift deterministically (move from over-filled to under-filled)
    rng.shuffle(subjects)
    for key in ("val", "test", "train"):
        target = counts[key]
        if len(out[key]) > target:
            for s in out[key][target:]:
                # find an under-filled split to receive it
                for other in ("train", "val", "test"):
                    if other != key and len(out[other]) < counts[other]:
                        out[other].append(s)
                        break
            out[key] = out[key][:target]
    for s in list(subjects):
        if sum(s in out[k] for k in out) != 1:
            # should not happen; put into train as last resort
            for k in ("val", "test"):
                out[k] = [x for x in out[k] if x != s]
            if s not in out["train"]:
                out["train"].append(s)
    for k in out:
        out[k] = sorted(out[k])
    return out


def split_counts(n: int) -> dict[str, int]:
    """~70/10/20 of the ACTUAL subject count (team doc: 以实际受试者数为准).

    For the 78 SC subjects this gives the doc's example numbers 54/8/16.
    """
    if n == 78:
        return dict(SPLIT_COUNTS)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.10))
    return {"train": n_train, "val": n_val, "test": n - n_train - n_val}


def main():
    SPLIT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, dtype=str)
    subjects = sorted(manifest["subject_id"].unique())
    n = len(subjects)
    counts = split_counts(n)
    print(f"{n} subjects, target counts {counts} (seed {SPLIT_SEED})")
    assert sum(counts.values()) == n, "split counts must sum to #subjects"

    rng = np.random.default_rng(SPLIT_SEED)
    method = "age-stratified"
    try:
        ages = load_ages()
        if any(s not in ages for s in subjects):
            raise ValueError(f"{sum(s not in ages for s in subjects)} subjects missing from age table")
        split_of = stratified_split(subjects, ages, rng, counts)
        print("age-stratified split (bins: <30 / 30-49 / 50+)")
    except ImportError as e:
        method = "plain-shuffle (no xlrd)"
        print(f"age stratification unavailable ({e}); falling back to plain shuffle")
        order = rng.permutation(subjects)
        split_of = {"train": [], "val": [], "test": []}
        for key, take in zip(("train", "val", "test"), counts.values()):
            split_of[key] = sorted(order[:take])
            order = order[take:]
        ages = {}
    except (ValueError, AssertionError) as e:
        method = "plain-shuffle (age table unusable)"
        print(f"age stratification failed ({e}); falling back to plain shuffle")
        order = rng.permutation(subjects)
        split_of = {"train": [], "val": [], "test": []}
        for key, take in zip(("train", "val", "test"), counts.values()):
            split_of[key] = sorted(order[:take])
            order = order[take:]
        ages = {}

    rec = manifest.groupby("subject_id")["record_id"].apply(lambda s: ";".join(sorted(s)))
    split_map = {s: k for k, lst in split_of.items() for s in lst}
    rows = [{
        "subject_id": s,
        "record_ids": rec[s],
        "split": split_map[s],
        "seed": SPLIT_SEED,
    } for s in subjects]
    df = pd.DataFrame(rows).sort_values("subject_id")

    # self-check FIRST, publish AFTER (team doc: 发布前自检)
    split_counts_by = df["split"].value_counts().to_dict()
    for k in ("train", "val", "test"):
        assert split_counts_by.get(k, 0) == counts[k], f"{k}: {split_counts_by.get(k)} != {counts[k]}"
    assert len(df) == n and df["subject_id"].is_unique
    for split_name, sub in split_of.items():
        recs = set(manifest[manifest["subject_id"].isin(sub)]["record_id"])
        listed = {r for s in df[df["split"] == split_name]["record_ids"] for r in s.split(";")}
        assert recs == listed, f"record mismatch in {split_name}"
    df.to_csv(OUT, index=False)
    print(f"split counts: {split_counts_by}  method: {method}")
    if ages:
        print("age distribution per split:")
        print(df.assign(age=df["subject_id"].map(ages))
                .groupby("split")["age"].agg(["count", "mean", "min", "max"]).round(1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
