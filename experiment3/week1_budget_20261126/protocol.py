"""Shared paths, keys and invariants for the paired budget experiment."""

import csv
import hashlib

ARMS = ("base", "add_people", "deepen")
BLOCK_SIZE = 60
KEY = ("sample_id", "subject_id", "record_id", "epoch_index")
SOURCES = {
    "epoch": "experiment4/week4/manifests/pretrain_epoch_index.csv",
    "split": "manifests/split_subjects.csv",
    "pretrain": "manifests/pretrain_split_w4_v01.csv",
    "dev": "manifests/val_reference_v1.csv",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
