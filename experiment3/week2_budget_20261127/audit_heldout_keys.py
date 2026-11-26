"""Check held-out prediction keys without reading labels or calculating scores."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path


KEYS = ("sample_id", "subject_id", "record_id", "epoch_index")


def read_keys(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(KEYS) <= set(reader.fieldnames or []):
            raise ValueError(f"Missing key fields in {path}")
        return Counter(tuple(row[name] for name in KEYS) for row in reader)


def reference(epoch_index, split_csv):
    with split_csv.open(newline="", encoding="utf-8-sig") as stream:
        held = {row["subject_id"] for row in csv.DictReader(stream) if row["pretrain_role"] == "heldout_val"}
    if len(held) != 8:
        raise ValueError(f"Expected eight held-out participants, got {len(held)}")
    full = read_keys(epoch_index)
    keys = Counter({key: n for key, n in full.items() if key[1] in held})
    if set(key[1] for key in keys) != held or any(n != 1 for n in keys.values()):
        raise ValueError("Missing participant or duplicate key in the reference")
    return keys


def write_coverage(path, expected):
    by_night = defaultdict(list)
    for _, subject, record, epoch in expected:
        by_night[subject, record].append(int(epoch))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["subject_id", "record_id", "valid_epochs", "first_epoch", "last_epoch", "gaps_in_range"])
        for (subject, record), epochs in sorted(by_night.items()):
            low, high = min(epochs), max(epochs)
            writer.writerow([subject, record, len(epochs), low, high, high - low + 1 - len(epochs)])
    return len(by_night)


def differences(expected, actual):
    missing = list((expected - actual).elements())
    extra = list((actual - expected).elements())
    return missing, extra


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--epoch-index", type=Path, required=True, help="Frozen W4 train-pool epoch index without labels")
    cli.add_argument("--pretrain-split", type=Path, required=True)
    cli.add_argument("--coverage-output", type=Path, required=True)
    cli.add_argument("predictions", nargs="*", type=Path, help="Frozen B prediction files")
    args = cli.parse_args()
    expected = reference(args.epoch_index, args.pretrain_split)
    nights = write_coverage(args.coverage_output, expected)
    subjects = len({k[1] for k in expected})
    print(f"Held-out reference: {subjects} subjects, {nights} nights, {len(expected):,} valid epochs")
    issues = []
    for path in args.predictions:
        missing, extra = differences(expected, read_keys(path))
        if missing or extra:
            issues.append(path)
        print(f"{path.name}: missing={len(missing)}, extra_or_duplicate={len(extra)}")
    if not args.predictions:
        print("Reference only. No prediction file audited.")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
