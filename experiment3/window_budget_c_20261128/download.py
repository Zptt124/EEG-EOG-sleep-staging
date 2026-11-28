"""Download verified Sleep-EDF recordings used by the development rosters."""
import argparse
import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
OLD = Path(os.environ.get("BUDGET_C_OLD", ROOT / "output/budget_replication_20261128"))
OUT = ROOT / "output/budget_unlabeled_window_c_20261128"
MANIFEST = ROOT / "manifests/dataset_manifest.csv"
BASES = ("https://physionet.org/files/sleep-edfx/1.0.0/",
         "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/")


def hash_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def records():
    selected = {r["record_id"] for r in read_rows(OLD / "blocks.csv")}
    manifest = {r["record_id"]: r for r in read_rows(MANIFEST)}
    subjects = {r["subject_id"] for r in read_rows(MANIFEST.with_name("split_subjects.csv"))
                if r["split"] == "train"}
    if any(manifest[k]["subject_id"] not in subjects for k in selected):
        raise ValueError("A requested recording is outside the training split")
    return [manifest[k] for k in sorted(selected)]


def get_file(relative, target, digest):
    if target.exists() and hash_file(target) == digest:
        return
    partial = target.with_suffix(".partial")
    for base in BASES * 2:
        try:
            with urlopen(base + relative, timeout=60) as response, partial.open("wb") as file:
                shutil.copyfileobj(response, file, length=1 << 20)
            if hash_file(partial) != digest:
                raise ValueError(f"Digest mismatch for {relative}")
            partial.replace(target)
            return
        except (OSError, ValueError) as error:
            print(f"Retry {relative}: {error}", flush=True)
    raise RuntimeError(f"Unable to download {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypnograms", action="store_true")
    args = parser.parse_args()
    source = OUT / "source"
    hashes = {name: digest for digest, name in
              (line.split() for line in (source / "SHA256SUMS.txt").read_text().splitlines())}
    if hash_file(source / "SC-subjects.xls") != hashes["SC-subjects.xls"]:
        raise ValueError("Metadata checksum mismatch")
    rows = records()
    column = "hyp_file" if args.hypnograms else "psg_file"
    raw = source / "raw"
    raw.mkdir(exist_ok=True)

    def download(row):
        name = row[column]
        get_file("sleep-cassette/" + name, raw / name, hashes["sleep-cassette/" + name])
        print(f"Verified {name}", flush=True)

    print(f"Downloading {len(rows)} {column} files for training rosters", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(download, rows))
    print("Download complete", flush=True)


if __name__ == "__main__":
    main()
