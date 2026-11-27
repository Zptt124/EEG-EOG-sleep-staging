"""Extract cached W4 features from the public Ying multi-night EEG cohort."""
import argparse
import hashlib
import importlib
import io
import os
from pathlib import Path
from functools import lru_cache
import sys
import tempfile
import time
import zipfile

import mne
import numpy as np
from scipy.signal import resample_poly
import torch

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA = "21c8514ea7761ef383d70b3e26ac19ea33ad0e7ccacfe690812ad359b0118e5b"
CHANNELS = ("F4-A1", "EOG-R", "EOG-L")
STAGES = {"Wake": 0, "NREM1": 1, "NREM2": 2, "NREM3": 3, "REM": 4, "NotScored": -1}


def label_rows(archive, member):
    stem = member.removesuffix(".edf")
    labels = []
    for expert in (1, 2, 3):
        path = f"{stem}_Expert{expert}"
        csv = np.loadtxt(io.BytesIO(archive.read(path + ".csv")), dtype=np.int64, delimiter=",", ndmin=1)
        text = archive.read(path + ".txt").decode("utf-8-sig")
        from_intervals = np.full(len(csv), -9, dtype=np.int64)
        for line in text.splitlines():
            if not line.strip():
                continue
            stage, start, end = line.split()
            lo, hi = int(start), int(end)
            if lo == hi == len(csv) * 30:
                continue
            if not (lo % 30 == hi % 30 == 0 and 0 <= lo < hi <= len(csv) * 30):
                raise ValueError(f"Unexpected label interval in {path}: {line}")
            if (from_intervals[lo//30:hi//30] != -9).any():
                raise ValueError(f"Overlap in labels: {path}")
            from_intervals[lo//30:hi//30] = STAGES[stage]
        if not np.array_equal(from_intervals, csv) or not np.isin(csv, np.arange(-1, 5)).all():
            raise ValueError(f"CSV and TXT annotations differ: {path}")
        labels.append(csv)
    # SLP010 Day 1 has an Expert 1 file five epochs shorter at its tail.
    # All interval files use the EDF's zero-time origin.
    common = min(map(len, labels))
    labels = [v[:common] for v in labels]
    a, b, c = labels
    # Majority vote; Expert 1 supplies the label when all three disagree.
    consensus = np.where(b == c, b, a)
    consensus[np.any(np.stack(labels) < 0, axis=0)] = -1
    return consensus.astype(np.int8), np.stack(labels).astype(np.int8)


def extract(w4, model, archive, member, scratch, batch):
    result, labels = label_rows(archive, member)
    with scratch.open("wb") as w, archive.open(member) as r:
        while chunk := r.read(8 << 20):
            w.write(chunk)
    raw = mne.io.read_raw_edf(scratch, include=CHANNELS, preload=False, verbose="ERROR")
    try:
        if raw.info["sfreq"] != 500 or set(raw.ch_names) != set(CHANNELS):
            raise ValueError(f"Unexpected montage/rate in {member}: {raw.ch_names}, {raw.info['sfreq']}")
        available = raw.n_times // 15000
        if available != len(result) and not (
            Path(member).stem == "SLP010_Day1" and available - len(result) == 5
        ):
            raise ValueError(f"Signal/label alignment: {member} {raw.n_times=} {len(result)=}")
        x = raw.get_data(picks=list(CHANNELS), stop=len(result) * 15000).astype(np.float32)
    finally:
        raw.close()
    # Frontopolar bipolar derivation is unavailable. Use a frontal EEG and a
    # differential EOG, with the same amplitude and W4 encoder transform.
    x[1] -= x[2]
    planar = resample_poly(x[:2], 1, 5, axis=1) * 1e6
    del x
    output = []
    for modality in (0, 1):
        transformed = w4.C.transform_uv(planar[modality].reshape(-1, 3000))
        pieces = []
        for start in range(0, len(result), batch):
            t = torch.from_numpy(transformed[start:start+batch, None]).cuda()
            act = w4.gate_activations(model, t, modality)
            s, _ = w4.pooled(act[:, -3:], 1)
            pieces.append(s.cpu().numpy())
        output.append(np.concatenate(pieces))
    features = np.concatenate(output, axis=1)
    if not (features.shape == (len(result), 384) and np.isfinite(features).all()):
        raise ValueError(f"Invalid features: {member}")
    return result, labels, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--w4-source", type=Path, default=ROOT / "experiment4/week4/src",
                        help="Directory containing the team's pinned W4 04_extract_features.py")
    parser.add_argument("--eegdm-vendor", type=Path,
                        default=Path(os.environ.get("EEGDM_VENDOR", ROOT / "experiment3/vendor/EEGDM")),
                        help="Root of the official EEGDM source checkout")
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--idle-timeout", type=float, default=600)
    args = parser.parse_args()
    if not (args.w4_source / "04_extract_features.py").is_file():
        parser.error("Set --w4-source to the W4 src directory linked in the README")
    if not args.eegdm_vendor.is_dir():
        parser.error("Set --eegdm-vendor to the official EEGDM checkout")
    os.environ["EEGDM_VENDOR"] = str(args.eegdm_vendor.resolve())
    sys.path.insert(0, str(args.w4_source.resolve()))
    w4 = importlib.import_module("04_extract_features")
    path = args.directory
    ckpt = path / "ckpt_step30000.pt"
    if hashlib.sha256(ckpt.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise ValueError("Checkpoint hash mismatch")
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, _, opts = w4.load_model(ckpt, "ema_model")
    model = model.cuda()
    # Cache the identical deterministic S4 kernels for all batches of 1000 points.
    for layer in model.layers:
        layer.layer.kernel.forward = lru_cache(maxsize=2)(layer.layer.kernel.forward)
    print("W4 30k EMA loaded", opts, flush=True)
    dest = path / "features"
    dest.mkdir(exist_ok=True)
    expected = {f"SLP{i:03d}_Day{d}" for i in range(1, 21) for d in (1, 2)
                if (i, d) not in {(4, 2), (20, 2)}}
    elapsed = time.monotonic()
    last_progress = elapsed
    with tempfile.TemporaryDirectory(prefix="ying-") as work:
        while True:
            completed = {p.stem for p in dest.glob("*.npz")}
            if completed == expected:
                print(f"Done: {len(completed)} recordings", flush=True)
                return
            for zip_path in sorted((path / "raw").glob("SLP*.zip")):
                with zipfile.ZipFile(zip_path) as archive:
                    actual = {Path(m).stem: m for m in archive.namelist() if m.endswith(".edf")}
                    needed = {s for s in expected - completed if s.startswith(zip_path.stem + "_")}
                    for key in sorted(needed):
                        start = time.monotonic()
                        result, labels, x = extract(
                            w4, model, archive, actual[key], Path(work) / "signal.edf", args.batch
                        )
                        with (dest / f"{key}.partial").open("wb") as f:
                            np.savez(f, y=result, experts=labels, x=x)
                        (dest / f"{key}.partial").replace(dest / f"{key}.npz")
                        completed.add(key)
                        last_progress = time.monotonic()
                        print(f"{len(completed)}/{len(expected)} {key} epochs={len(result)} "
                              f"seconds={last_progress-start:.1f} total={last_progress-elapsed:.0f}", flush=True)
            if time.monotonic() - last_progress > args.idle_timeout:
                raise TimeoutError(f"Waiting on {sorted(expected - completed)}")
            time.sleep(2)


if __name__ == "__main__":
    main()


