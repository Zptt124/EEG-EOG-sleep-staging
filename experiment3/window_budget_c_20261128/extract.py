"""Supplement the frozen cache for locked Sleep-EDF experiment C epochs."""
from collections import defaultdict
from functools import lru_cache
import importlib
import json
from pathlib import Path
import os
import sys
import time

import mne
import numpy as np
import torch

from download import ROOT, OLD, OUT, MANIFEST, hash_file, read_rows
from common_run import BLOCK, BUNDLE, load_lock

SOURCE = Path(os.environ.get("BUDGET_C_W4_SOURCE", ROOT / "experiment4/week4/src"))
PREPROCESS = ROOT / "scripts"
os.environ.setdefault("EEGDM_VENDOR", str(ROOT / "experiment3/vendor/EEGDM"))
sys.path[:0] = [str(SOURCE), str(PREPROCESS)]
w4 = importlib.import_module("04_extract_features")
preprocess = importlib.import_module("04_preprocess")

CHANNELS = ("EEG Fpz-Cz", "EOG horizontal")
BATCH = 24


def extract_epochs(model, raw, epochs):
    epochs = np.asarray(epochs)
    parts = np.split(epochs, np.flatnonzero(np.diff(epochs) != 1) + 1)
    # Read contiguous runs so sparse epochs need no full-night array.
    signals = np.concatenate([
        (raw.get_data(picks=list(CHANNELS), start=int(part[0]) * 3000,
                      stop=(int(part[-1]) + 1) * 3000) * 1e6)
        .astype(np.float32).reshape(2, -1, 3000) for part in parts
    ], axis=1)
    results = []
    for modality in range(2):
        pieces = []
        for start in range(0, len(epochs), BATCH):
            signal = w4.C.transform_uv(signals[modality, start:start + BATCH])
            activation = w4.gate_activations(model, torch.from_numpy(signal[:, None]).cuda(), modality)
            sd, _ = w4.pooled(activation[:, -3:], 1)
            pieces.append(sd.cpu().numpy())
        results.append(np.concatenate(pieces))
    result = np.concatenate(results, axis=1)
    if result.shape != (len(epochs), 384) or not np.isfinite(result).all():
        raise ValueError("Unexpected extracted feature shape or nonfinite values")
    return result


def read_labels(path, epochs):
    hyp = mne.read_annotations(path)
    grid = preprocess.build_epoch_grid(hyp)
    labels = np.full(len(epochs), -1, dtype=np.int8)
    for i, epoch in enumerate(epochs):
        if epoch < len(grid):
            labels[i] = preprocess.LABEL_MAP.get(grid[epoch], -1)
    return labels


def process(model, row, epochs, metadata, shared, arrays, hashes, raw_dir, dest):
    record = row["record_id"]
    psg = raw_dir / row["psg_file"]
    hyp = raw_dir / row["hyp_file"]
    for path in (psg, hyp):
        if hash_file(path) != hashes["sleep-cassette/" + path.name]:
            raise ValueError(f"Download checksum mismatch: {path.name}")
    with psg.open("rb") as file:
        header = file.read(256)
    header_cache = raw_dir.parent / "headers" / psg.name
    if header_cache.exists() and header_cache.read_bytes() != header:
        raise ValueError(f"Header changed between planning and extraction: {record}")
    with hyp.open("rb") as file:
        hyp_clock = file.read(184)[168:184]
    if hyp_clock != header[168:184]:
        raise ValueError(f"PSG and hypnogram start dates or clocks differ for {record}")
    target = read_labels(hyp, epochs)
    raw = mne.io.read_raw_edf(psg, include=list(CHANNELS), preload=False, verbose="ERROR")
    try:
        if (raw.info["sfreq"] != 100 or set(raw.ch_names) != set(CHANNELS)
                or raw.info["meas_date"].strftime("%H:%M:%S") != metadata["edf_start"]
                or raw.n_times / 100 != float(metadata["duration_sec"])
                or (epochs[-1] + 1) * 3000 > raw.n_times):
            raise ValueError(f"Unexpected signal alignment for {record}")
        overlap = [i for i, e in enumerate(epochs) if int(e) in shared]
        for i in overlap:
            if target[i] != shared[int(epochs[i])][1]:
                raise ValueError(f"Raw labels disagree with cache for {record} epoch {epochs[i]}")
        check = [overlap[i] for i in sorted({0, len(overlap) // 2, len(overlap) - 1})] if overlap else []
        missing = [i for i, e in enumerate(epochs) if int(e) not in shared]
        indices = sorted(set(check + missing))
        delta = 0.0
        x = np.empty((len(missing), 384), dtype=np.float32)
        if indices:
            extracted = extract_epochs(model, raw, epochs[indices])
            lookup = {i: value for i, value in zip(indices, extracted)}
            for i in check:
                cache_index, _ = shared[int(epochs[i])]
                reference = np.concatenate([a[cache_index] for a in arrays])
                delta = max(delta, float(np.max(np.abs(reference - lookup[i]))))
                if not np.allclose(reference, lookup[i], atol=2e-4, rtol=5e-3):
                    raise ValueError(f"Feature replay mismatch in {record} epoch {epochs[i]}: {delta:g}")
            if missing:
                x = np.stack([lookup[i] for i in missing])
    finally:
        raw.close()
    with dest.with_suffix(".partial").open("wb") as file:
        np.savez(file, epoch=epochs, y=target, missing_epoch=epochs[missing], x=x,
                 checked_overlap=len(check), max_abs_overlap_error=delta)
    dest.with_suffix(".partial").replace(dest)
    return len(missing), int(np.count_nonzero(target < 0)), delta


def main():
    lock = load_lock()
    info = json.loads((BUNDLE / "features_info.json").read_text())
    ckpt = Path(os.environ.get("BUDGET_C_CHECKPOINT", ROOT / "output/budget_external_ying/ckpt_step30000.pt"))
    if hash_file(ckpt) != info["checkpoint_sha256"]:
        raise ValueError("Frozen W4 checkpoint hash mismatch")
    epochs_by_record = defaultdict(set)
    for row in read_rows(OUT / "selected_blocks.csv"):
        first = int(row["first_epoch"])
        epochs_by_record[row["record_id"]].update(range(first, first + BLOCK))
    shared_by_record = defaultdict(dict)
    for idx, r in enumerate(read_rows(BUNDLE / "index.csv")):
        if r["record_id"] in epochs_by_record:
            if r["role"] != "pretrain_train":
                raise ValueError("Selected a recording outside development training")
            shared_by_record[r["record_id"]][int(r["epoch_index"])] = (idx, int(r["y_true"]))
    arrays = [np.load(BUNDLE / f"{m}_std.npy", mmap_mode="r") for m in ("eeg", "eog")]
    files = {r["record_id"]: r for r in read_rows(MANIFEST)}
    metadata = {r["record_id"]: r for r in read_rows(OUT / "recording_metadata.csv")}
    source = OUT / "source"
    hashes = {name: digest for digest, name in
              (line.split() for line in (source / "SHA256SUMS.txt").read_text().splitlines())}
    dest = OUT / "supplement"
    dest.mkdir(exist_ok=True)
    raw_dir = source / "raw"
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, _, _ = w4.load_model(ckpt, "ema_model")
    model = model.cuda()
    for layer in model.layers:
        layer.layer.kernel.forward = lru_cache(maxsize=2)(layer.layer.kernel.forward)
    pending = {r for r in epochs_by_record if not (dest / f"{r}.npz").exists()}
    print(f"W4 30k EMA loaded | {len(lock['paired_rosters'])} rosters | "
          f"{len(pending)}/{len(epochs_by_record)} recordings pending", flush=True)
    last_progress = time.monotonic()
    while pending:
        for record in sorted(pending):
            row = files[record]
            if not all((raw_dir / row[key]).exists() for key in ("psg_file", "hyp_file")):
                continue
            epochs = np.array(sorted(epochs_by_record[record]), dtype=np.int32)
            missing, invalid, delta = process(model, row, epochs, metadata[record],
                                              shared_by_record[record], arrays, hashes,
                                              raw_dir, dest / f"{record}.npz")
            pending.remove(record)
            last_progress = time.monotonic()
            print(f"{len(epochs_by_record) - len(pending)}/{len(epochs_by_record)} {record}: "
                  f"{len(epochs)} selected epochs, {missing} supplemental, {invalid} unscorable, "
                  f"feature replay max abs {delta:.3g}", flush=True)
        if pending and time.monotonic() - last_progress > 900:
            raise TimeoutError(f"Waiting on verified raw recordings: {sorted(pending)}")
        if pending:
            time.sleep(2)


if __name__ == "__main__":
    main()
