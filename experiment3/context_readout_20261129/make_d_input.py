"""Build the D K1 training cache from locked blocks and physical epoch features."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from train_bigru import sha256


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def block(row):
    first = int(row["first_epoch"])
    if row.get("last_epoch") and int(row["last_epoch"]) != first + 59:
        raise ValueError(f"Expected a 60-epoch block: {row}")
    return {(row["record_id"], e) for e in range(first, first + 60)}


def build(args):
    if sha256(args.blocks) != args.selection_sha256:
        raise ValueError("Selections do not match the supplied lock SHA-256")
    starts, units = defaultdict(set), defaultdict(set)
    for row in rows(args.anchor):
        starts[int(row["roster"])].update(block(row))
    for row in rows(args.blocks):
        unit = int(row["roster"]), int(row["draw"]), row["strategy"]
        units[unit].update(block(row))
    if any(len(v) != 1200 for v in starts.values()) or not units or any(
            len(v) != 600 or (k[0] not in starts) or (starts[k[0]] & v) for k, v in units.items()):
        raise ValueError("Expect 1,200 anchor epochs and 600 disjoint additions per unit")
    needed_centers = set().union(*(starts[roster] | units[roster, draw, strategy]
                                  for roster, draw, strategy in units))
    labels = {}
    for row in rows(args.labels):
        key = row["record_id"], int(row["epoch_index"])
        if key in needed_centers:
            if key in labels:
                raise ValueError(f"Duplicate label: {key}")
            labels[key] = int(row["y_true"])
    if labels.keys() != needed_centers or not set(labels.values()) <= {-1, 0, 1, 2, 3, 4}:
        raise ValueError("Supply the truth for each locked center, using -1 for unscorable")
    if any(labels[key] < 0 for roster in {k[0] for k in units} for key in starts[roster]):
        raise ValueError("Anchor contains an unscorable center")
    train = sorted(k for k, y in labels.items() if y >= 0)

    with np.load(args.c_cache, allow_pickle=False) as file:
        c = {name: file[name] for name in ("subjects", "subject_index", "sample_id", "truth", "eval_windows",
                                            "psg_record", "psg_length", "x", "feature_record", "feature_epoch")}
    lengths = dict(zip(c["psg_record"], c["psg_length"]))
    if args.metadata:
        for row in rows(args.metadata):
            r, n = row["record_id"], int(float(row["duration_sec"]) // 30)
            if r in lengths and lengths[r] != n:
                raise ValueError(f"PSG duration differs from the common cache: {r}")
            lengths[r] = n
    eval_keys = [(c["feature_record"][j], int(c["feature_epoch"][j]))
                 for j in c["eval_windows"][:, c["eval_windows"].shape[1] // 2]]
    if any(r[:-1] in c["subjects"] for r, e in train):
        raise ValueError("A development evaluation subject would be supervised")

    def window(key):
        r, e = key
        if r not in lengths or not 0 <= e < lengths[r]:
            raise ValueError(f"Missing PSG length or invalid physical center: {key}")
        return [(r, max(e - 1, 0)), key, (r, min(e + 1, lengths[r] - 1))]

    needed = {k for center in train + eval_keys for k in window(center)}
    c_pos = {(r, int(e)): i for i, (r, e) in enumerate(zip(c["feature_record"], c["feature_epoch"]))
             if (r, int(e)) in needed}
    info = json.loads((args.bundle / "features_info.json").read_text(encoding="utf-8"))
    index = args.bundle / "index.csv"
    if sha256(index) != info["index_sha256"]:
        raise ValueError("Shared index hash mismatch")
    files = [args.bundle / f"{name}.npy" for name in ("eeg_std", "eog_std")]
    if any(sha256(path) != info["files"][path.stem]["sha256"] for path in files):
        raise ValueError("Shared feature hash mismatch")
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in files]
    remaining = needed - c_pos.keys()
    shared = {(row["record_id"], int(row["epoch_index"])): i for i, row in enumerate(rows(index))
              if (row["record_id"], int(row["epoch_index"])) in remaining}
    extra = {}
    for path in args.supplement:
        with np.load(path, allow_pickle=False) as npz:
            for r, e, x in zip(npz["feature_record"], npz["feature_epoch"], npz["x"], strict=True):
                key = r, int(e)
                if key in extra:
                    raise ValueError(f"Duplicate supplemental epoch: {key}")
                extra[key] = x
    missing = sorted(remaining - shared.keys() - extra.keys())
    if missing:
        if args.missing_output:
            with args.missing_output.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("record_id", "epoch_index"))
                writer.writerows(missing)
        raise ValueError(f"Missing {len(missing)} physical feature rows. First keys: {missing[:5]}")
    keys = sorted(needed)
    positions = {k: i for i, k in enumerate(keys)}
    x = np.stack([c["x"][c_pos[k]] if k in c_pos else
                  np.concatenate([arr[shared[k]] for arr in arrays]) if k in shared else extra[k] for k in keys])
    if x.shape != (len(keys), 384) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Expected finite float32 frozen 384-D features")
    train_pos = {k: i for i, k in enumerate(train)}
    sorted_units = sorted(units)
    mask = np.full((len(units), 1800), -1, np.int32)
    effective = []
    for i, u in enumerate(sorted_units):
        chosen = [train_pos[k] for k in sorted(starts[u[0]] | units[u]) if labels[k] >= 0]
        mask[i, :len(chosen)] = chosen
        effective.append(len(chosen) - 1200)
    with args.output.with_suffix(".tmp").open("wb") as file:
        np.savez_compressed(file, x=x, feature_record=[r for r, _ in keys], feature_epoch=[e for _, e in keys],
            psg_record=list(lengths), psg_length=list(lengths.values()), train_y=[labels[key] for key in train],
            train_windows=[[positions[k] for k in window(center)] for center in train],
            eval_windows=[[positions[k] for k in window(center)] for center in eval_keys],
            unit_mask=mask, effective_additions=effective, roster=[k[0] for k in sorted_units],
            draw=[k[1] for k in sorted_units], strategy=[k[2] for k in sorted_units],
            **{name: c[name] for name in ("subjects", "subject_index", "sample_id", "truth")})
    args.output.with_suffix(".tmp").replace(args.output)
    print(f"Prepared {len(units)} units | SHA-256 {sha256(args.output)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("blocks", "anchor", "labels", "c-cache", "bundle", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--missing-output", type=Path)
    build(parser.parse_args())
