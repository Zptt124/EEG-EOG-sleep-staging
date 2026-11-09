"""Produce the six B Week 3 bundles using the pinned team extractor."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8d218342c6ad6c8afa649824c1a18ccd189f9d07"
MEAN = ROOT / "reports/week3_c_readout_20261107/mean_seed0_v01"
FIELDS = ("sample_id", "subject_id", "record_id", "epoch_index", "split", "y_true")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def identity(rows):
    return [tuple(row[key] for key in FIELDS) for row in rows]


def seed_mean_cache(b, modality, out, records, expected, timesteps):
    """Validate the earlier mean arrays and prepare B's per-record resume files."""
    source = MEAN / "features" / modality
    cfg = load_json(source / "feature_config.json")
    official = b["c"].config(modality, 14)
    expected_cfg = dict(official, version="E3_features_mean_v01", reduction="mean",
                        record_scope_sha256=b["sha256"](MEAN / "records.txt"))
    require(cfg == expected_cfg, f"{modality} mean source configuration differs")
    require(identity(read_rows(source / "feature_index.csv")) == expected,
            f"{modality} mean source sample index differs")
    completed = load_json(source / "summary.json")
    require(completed["status"] == "complete" and completed["samples"] == len(expected)
            and not completed["test_used"], f"{modality} mean cache is incomplete")
    profiles = {row["record_id"]: row for row in read_rows(source / "resource_profile.csv")}
    first = records[0][0]
    channel = 0 if modality == "eeg" else 1
    torch.manual_seed(0)
    extractor = b["Extractor"](14, "mean", 1)
    extractor.cache_frozen_kernels()
    with np.load(b["c"].DATA / f"{first}.npz", allow_pickle=False) as data:
        fresh = extractor(data["signal"][:4, channel]).astype(np.float16)
    existing = np.load(source / f"{first}.npy", mmap_mode="r", allow_pickle=False)[:4]
    require(np.array_equal(fresh, existing), f"{modality} cached mean replay differs")
    del extractor, existing
    torch.cuda.empty_cache()
    settings = argparse.Namespace(modality=modality, condition=14, reduce="mean", diffusion_t=1,
                                  budget="label25", save_full=True)
    out.mkdir(exist_ok=True)
    configuration = b["config"](settings, timesteps)
    cfg_path = out / "feature_config.json"
    if cfg_path.exists():
        require(load_json(cfg_path) == configuration, f"{out.name} configuration differs")
    write_json(cfg_path, configuration)

    for record, split in records:
        done = out / f"{record}.done.json"
        raw = source / f"{record}.npy"
        linked = out / raw.name
        profile = profiles[record]
        n = int(profile["n_samples"])
        require(profile["split"] == split and profile["batch_size"] == "4"
                and profile["cache_frozen_kernels"] == "True", f"{record} mean profile differs")
        if not linked.exists():
            os.link(raw, linked)
        require(os.path.samefile(raw, linked), f"{record} full cache does not match its source")
        if done.exists():
            continue
        arr = np.load(raw, mmap_mode="r", allow_pickle=False)
        require(arr.dtype == np.float16 and arr.shape == (n, 20, 30, 128), f"{record} mean shape differs")
        pooled = np.empty((n, 2560), dtype=np.float32)
        for offset in range(0, n, 64):
            block = arr[offset:offset + 64].astype(np.float32)
            pooled[offset:offset + len(block)] = block.mean(axis=2).reshape(len(block), 2560)
        require(np.isfinite(pooled).all(), f"{record} mean contains nonfinite values")
        partial = out / f"{record}.partial.pooled.npy"
        np.save(partial, pooled)
        partial.replace(out / f"{record}.pooled.npy")
        write_json(done, dict(record_id=record, split=split, n_samples=n,
                             seconds=float(profile["seconds"]),
                             samples_per_second=float(profile["samples_per_second"]),
                             peak_gpu_mb=float(profile["peak_gpu_mb"]), batch_size=4,
                             cache_frozen_kernels=True, reused_mean_cache=True))
    return dict(source=str(source), replay_record=first, replay_samples=len(fresh),
                replay_max_abs_diff=0, full_cache="hardlinked to the original mean arrays")


def validate_bundle(out, expected, b):
    summary = load_json(out / "summary.json")
    require(summary["status"] == "complete" and summary["samples"] == len(expected)
            and summary["records"] == 43 and not summary["test_used"]
            and summary["split_counts"] == {"train": 32310, "val": 21186},
            f"{out.name} has incomplete coverage")
    require(identity(read_rows(out / "sample_index.csv")) == expected,
            f"{out.name} sample index differs")
    npz = out / summary["npz"]
    with np.load(npz, allow_pickle=False) as payload:
        require(payload.files == ["sample_id", "x"], f"{npz.name} keys differ")
        require(payload["sample_id"].tolist() == [row[0] for row in expected], f"{npz.name} IDs differ")
        x = payload["x"]
        require(x.shape == (53496, 2560) and x.dtype == np.float32
                and np.isfinite(x).all(), f"{npz.name} feature matrix differs")
    files = [npz, *(out / name for name in ("sample_index.csv", "feature_config.json",
                                            "resource_profile.csv", "summary.json", "SHA256SUMS.txt"))]
    hashes = {path.name: b["sha256"](path) for path in files}
    for line in (out / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(hashes[name] == digest, f"{out.name}/{name} checksum differs")
    return files, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extractor_script = args.checkout / "scripts/26_b_extract_eegdm_w3.py"
    smoke = args.output / "eeg_std_t1_cond14_label25_smoke2/summary.json"
    summary = load_json(smoke)
    reference = summary["reference_check"]
    require(summary["status"] == "smoke" and summary["records"] == 2
            and summary["samples"] == reference["rows"] and reference["max_abs_diff"] <= 1e-7
            and reference["reference_sha256"] ==
            "047bafc735d64297648b173447ec5d8425063a25b3f85562a7dbf12295800fb7",
            "Published EEG std consistency check did not pass")
    progress_file = args.output / "progress.json"
    status = dict(revision=REVISION, std_consistency=reference, completed=[], test_used=False)

    def progress(stage, **fields):
        updated = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
        info = dict(status, stage=stage, updated_at=updated, **fields)
        write_json(progress_file, info)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    try:
        progress("checking_inputs")
        b = runpy.run_path(str(extractor_script))
        require(b["sha256"](extractor_script) ==
                "77f768fad4c6b8e1e74eeb3fe9ff0c81ddc5434f7023a13afdb7be1a270e48a1"
                and b["sha256"](b["C_SCRIPT"]) ==
                "d660cd5768fb4e90e0e713423ab0bfb054837b26670119e13e2f9e8cf447a235",
                "B/C extractor sources differ from the pinned checkout")
        torch.set_num_threads(4)
        timesteps = b["checkpoint_timesteps"]()
        require(all(n == 50 for _, n in timesteps), "Checkpoint timestep count differs")
        upstream_config = b["c"].UPSTREAM / "conf/pretrain/base.yaml"
        upstream_text = upstream_config.read_text(encoding="utf-8")
        scheduler = yaml.safe_load(upstream_text)["model"]["noise_sch_kwargs"]
        require(scheduler["num_train_timesteps"] == 50, "Official scheduler timestep count differs")
        subjects = b["allowed_subjects"]("label25")
        records = [(r, s) for r, s in sorted(b["c"].record_splits().items())
                   if s in ("train", "val") and r[:5] in subjects]
        expected = identity(read_rows(MEAN / "features/eeg/feature_index.csv"))
        require(len(subjects) == 22 and len(records) == 43 and len(expected) == 53496
                and {row[1] for row in expected} == subjects
                and {(row[2], row[4]) for row in expected} == set(records), "B sample scope differs")
        require(b["sha256"](MEAN / "records.txt") == load_json(MEAN / "plan.json")["records_sha256"],
                "Mean record scope checksum differs")
        manifest = dict(revision=REVISION, scope=dict(records=43, train=32310, dev=21186),
                        test_used=False, checkpoint_timestep_fields=timesteps,
                        official_scheduler_timesteps=scheduler["num_train_timesteps"],
                        official_scheduler_config_sha256=b["sha256"](upstream_config), bundles={})
        handoff = args.output / "handoff"
        handoff.mkdir(exist_ok=True)
        smoke_copy = handoff / "std_t1_smoke_summary.json"
        if not smoke_copy.exists():
            os.link(smoke, smoke_copy)
        for pooling, timestep in (("mean", 1), ("std", 10), ("std", 25)):
            for modality in ("eeg", "eog"):
                name = f"{modality}_{pooling}_t{timestep}_cond14_label25"
                out = args.output / name
                provenance = None
                if pooling == "mean":
                    progress("verifying_and_reusing_mean", current=name)
                    provenance = seed_mean_cache(b, modality, out, records, expected, timesteps)
                progress("extracting" if pooling == "std" else "building_mean_bundle", current=name)
                command = [sys.executable, "-u", str(extractor_script), "--modality", modality,
                           "--reduce", pooling, "--diffusion-t", str(timestep), "--cache-kernels",
                           "--output", str(out)]
                if pooling == "mean":
                    command.append("--save-full")
                subprocess.run(command, cwd=args.checkout, check=True)
                files, hashes = validate_bundle(out, expected, b)
                destination = handoff / name
                destination.mkdir(exist_ok=True)
                for file in files:
                    link = destination / file.name
                    if not link.exists():
                        os.link(file, link)
                    require(os.path.samefile(file, link), f"Handoff file differs: {link}")
                manifest["bundles"][name] = dict(sha256=hashes, mean_cache_reuse=provenance)
                write_json(handoff / "delivery_manifest.json", manifest)
                status["completed"].append(name)
                progress("bundle_complete", current=name)
        progress("complete")
    except Exception as error:
        progress("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
