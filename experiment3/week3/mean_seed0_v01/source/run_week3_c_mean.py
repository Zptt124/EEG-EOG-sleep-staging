"""Run the Week 3 C single-seed extension with one-second mean pooling."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import extract_features as extraction
import run_week3_c_readout as readout

SEED = 0


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_reduction(row, sources):
    record = row["record_id"]
    batch_size = 4
    extractor = extraction.Extractor(14, "mean")
    extractor.cache_frozen_kernels()
    std_reducer = extraction.LatentActivityReducer(
        query=["gate"], reduce=["std"], L=6000, window_size=200, window_step=200
    ).cuda().eval()
    with np.load(extraction.DATA / f"{record}.npz", allow_pickle=False) as raw:
        signals = raw["signal"][:batch_size]

    checks = {}
    for channel, (name, std_source) in enumerate(sources.items()):
        x = torch.from_numpy(extraction.transform(signals[:, channel])).cuda()
        with torch.inference_mode():
            latent = extractor.extractor((x, None))
            cached_axes = (slice(None), slice(None), 0, slice(None), 0, 0, slice(None))
            mean = extractor.reducers[6000](latent)[cached_axes]
            std = std_reducer(latent)[cached_axes]
            expected = latent[:, 0, :, 0].reshape(batch_size, 20, 30, 200, 128).mean(dim=3)
            torch.testing.assert_close(mean, expected)
            historical = np.load(std_source / f"{record}.npy", mmap_mode="r", allow_pickle=False)
            std_stored = std.cpu().numpy().astype(np.float16).astype(np.float32)
            std_reference = historical[:batch_size].astype(np.float32)
            np.testing.assert_allclose(std_stored, std_reference, rtol=2e-3, atol=5e-5)
            readout.require(torch.isfinite(mean).all().item(), f"{name} mean is nonfinite")
        checks[name] = dict(shape=list(mean.shape), record_id=record,
                            historical_std_max_abs=float(np.max(np.abs(std_stored - std_reference))))
    return checks


def audit_mean_cache(modality, folder, records_file, expected_rows, split):
    cfg = load_json(folder / "feature_config.json")
    expected_cfg = extraction.config(modality.lower(), 14, "mean")
    expected_cfg["record_scope_sha256"] = readout.sha(records_file)
    readout.require(cfg == expected_cfg, f"{modality} mean extraction configuration differs")
    summary = load_json(folder / "summary.json")
    readout.require(summary["status"] == "complete" and not summary["test_used"]
                    and summary["samples"] == len(expected_rows), f"{modality} extraction incomplete")
    rows = readout.read_csv(folder / "feature_index.csv")
    readout.align_indices(expected_rows, rows)
    grouped = readout.audit_index(rows, split, strict=False)
    for record, items in grouped.items():
        array = np.load(folder / f"{record}.npy", mmap_mode="r", allow_pickle=False)
        readout.require(array.shape == (len(items), 20, 30, 128) and array.dtype == np.float16,
                        f"Invalid {modality} mean feature header for {record}")
    return summary


def compare_pooling(output, prior, train_dir):
    official = readout.official_module()
    results = {}
    keys = ["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]
    for name in ("C0", "C1"):
        std = official.load_predictions(prior / name / "predictions_val.csv")
        mean = official.load_predictions(train_dir / name / "predictions_val.csv")
        readout.require(std[keys].equals(mean[keys]), f"Development sample order differs for {name}")
        results[name] = official.paired_bootstrap_delta(mean, std, n_boot=2000, seed=SEED)
    readout.write_json(output / "mean_minus_std.json", dict(seed=SEED, primary="C1",
                                                          paired=results, test_used=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-reference", type=Path, required=True)
    parser.add_argument("--std-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    readout.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    config = load_json(readout.CONFIG)
    prior = args.std_report / "seed0_v01"
    for seed in config["seeds"]:
        run = args.std_report / f"seed{seed}_v01"
        progress = load_json(run / "progress.json")
        readout.require(progress["stage"] == "complete" and progress["dev_samples"] == 21186
                        and not progress["test_used"] and load_json(run / "config.json") == config,
                        f"Primary comparison seed {seed} is incomplete or differs in configuration")
    std_audit = load_json(prior / "input_audit.json")
    readout.require(std_audit["development_reference_sha256"] == readout.sha(args.dev_reference),
                    "Development reference differs from the std run")

    args.output.mkdir(parents=True, exist_ok=False)
    progress_path = args.output / "progress.json"
    status = dict(seed=SEED, pooling="mean", test_used=False)

    def progress(stage, **values):
        data = dict(status, stage=stage, **values)
        readout.write_json(progress_path, data)
        print(json.dumps(data), flush=True)

    try:
        progress("checking_extraction")
        rows, train, dev, audit = readout.inspect_inputs(args.dev_reference)
        original_runner = args.std_report / "run_week3_c_readout_std_original.py"
        for relative, expected in std_audit["input_sha256"].items():
            path = readout.ROOT / relative
            if path == Path(readout.__file__):
                path = original_runner
            readout.require(readout.sha(path) == expected, f"Std protocol source changed: {relative}")
        positions = sorted(np.concatenate((train, dev)))
        expected_rows = [rows[i] for i in positions]
        records = sorted({row["record_id"] for row in expected_rows})
        record_file = args.output / "records.txt"
        record_file.write_text("\n".join(records) + "\n", encoding="utf-8")
        status.update(records=len(records), train_samples=len(train), dev_samples=len(dev))
        readout.write_json(args.output / "plan.json", dict(
            status, head_config=config, std_seed0_input_audit=std_audit,
            records_sha256=readout.sha(record_file), mean_extractor_sha256=readout.sha(Path(extraction.__file__)),
            extension_runner_sha256=readout.sha(Path(__file__)), training_source_audit=audit,
            primary_comparison="one-second mean + LFT vs one-second std + LFT",
            secondary_comparison="one-second mean + linear vs one-second std + linear",
        ))
        checks = verify_reduction(rows[train[0]], readout.SOURCES)
        readout.write_json(args.output / "extraction_preflight.json", checks)
        torch.cuda.empty_cache()

        sources = {name: args.output / "features" / name.lower() for name in readout.MODALITIES}
        split = readout.read_csv(readout.SPLIT)
        for modality, folder in sources.items():
            progress("extracting", modality=modality)
            subprocess.run([
                sys.executable, "-u", extraction.__file__, "--modality", modality.lower(),
                "--pooling", "mean", "--batch-size", "4", "--cache-kernels", "--records-file",
                str(record_file), "--output", str(folder),
            ], check=True)
            audit_mean_cache(modality, folder, record_file, expected_rows, split)

        mean_rows = readout.read_csv(sources["EEG"] / "feature_index.csv")
        _, mean_train, mean_dev = readout.selected_positions(
            mean_rows, readout.read_csv(readout.SUBSETS), "label25"
        )
        train_dir = args.output / "train"
        train_dir.mkdir()
        readout.write_json(train_dir / "config.json", dict(config, pooling="mean"))
        readout.write_json(train_dir / "smoke.json", readout.smoke(
            config, mean_rows, mean_train, device, sources
        ))
        progress("training")
        readout.run_pair(config, mean_rows, mean_train, mean_dev, SEED, device, train_dir, sources)
        compare_pooling(args.output, prior, train_dir)
        progress("complete")
    except Exception as error:
        progress("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
