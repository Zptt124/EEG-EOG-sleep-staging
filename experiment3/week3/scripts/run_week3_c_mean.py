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


def verify_reduction(row, sources, pooling, diffusion_t):
    record = row["record_id"]
    batch_size = 4
    extractor = extraction.Extractor(14, pooling, diffusion_t)
    extractor.cache_frozen_kernels()
    readout.require(extractor.extractor.diffusion_steps.item() == diffusion_t,
                    "Extractor timestep differs")
    std_reducer = extraction.LatentActivityReducer(
        query=["gate"], reduce=["std"], L=6000, window_size=200, window_step=200
    ).cuda().eval() if pooling == "mean" else extractor.reducers[6000]
    with np.load(extraction.DATA / f"{record}.npz", allow_pickle=False) as raw:
        signals = raw["signal"][:batch_size]

    checks = {}
    for channel, (name, std_source) in enumerate(sources.items()):
        x = torch.from_numpy(extraction.transform(signals[:, channel])).cuda()
        with torch.inference_mode():
            latent = extractor.extractor((x, None))
            cached_axes = (slice(None), slice(None), 0, slice(None), 0, 0, slice(None))
            pooled = extractor.reducers[6000](latent)[cached_axes]
            std = std_reducer(latent)[cached_axes]
            windows = latent[:, 0, :, 0].reshape(batch_size, 20, 30, 200, 128)
            expected = windows.mean(dim=3) if pooling == "mean" else windows.std(dim=3)
            torch.testing.assert_close(pooled, expected)
            historical = np.load(std_source / f"{record}.npy", mmap_mode="r", allow_pickle=False)
            std_stored = std.cpu().numpy().astype(np.float16).astype(np.float32)
            std_reference = historical[:batch_size].astype(np.float32)
            if diffusion_t == 1:
                np.testing.assert_allclose(std_stored, std_reference, rtol=2e-3, atol=5e-5)
            else:
                readout.require(not np.array_equal(std_stored, std_reference),
                                f"{name} timestep change left cached std unchanged")
            readout.require(torch.isfinite(pooled).all().item(), f"{name} features are nonfinite")
        checks[name] = dict(shape=list(pooled.shape), record_id=record, diffusion_t=diffusion_t,
                            historical_std_max_abs=float(np.max(np.abs(std_stored - std_reference))))
    return checks


def audit_variant_cache(modality, folder, records_file, expected_rows, split, pooling, diffusion_t):
    cfg = load_json(folder / "feature_config.json")
    expected_cfg = extraction.config(modality.lower(), 14, pooling, diffusion_t)
    expected_cfg["record_scope_sha256"] = readout.sha(records_file)
    readout.require(cfg == expected_cfg, f"{modality} extraction configuration differs")
    summary = load_json(folder / "summary.json")
    readout.require(summary["status"] == "complete" and not summary["test_used"]
                    and summary["samples"] == len(expected_rows), f"{modality} extraction incomplete")
    rows = readout.read_csv(folder / "feature_index.csv")
    readout.align_indices(expected_rows, rows)
    grouped = readout.audit_index(rows, split, strict=False)
    for record, items in grouped.items():
        array = np.load(folder / f"{record}.npy", mmap_mode="r", allow_pickle=False)
        readout.require(array.shape == (len(items), 20, 30, 128) and array.dtype == np.float16,
                        f"Invalid {modality} feature header for {record}")
    return summary


def compare_to_baseline(output_file, prior, train_dir):
    official = readout.official_module()
    results = {}
    keys = ["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]
    for name in ("C0", "C1"):
        baseline = official.load_predictions(prior / name / "predictions_val.csv")
        variant = official.load_predictions(train_dir / name / "predictions_val.csv")
        readout.require(baseline[keys].equals(variant[keys]), f"Development sample order differs for {name}")
        results[name] = official.paired_bootstrap_delta(variant, baseline, n_boot=2000, seed=SEED)
    readout.write_json(output_file, dict(seed=SEED, primary="C1", paired=results, test_used=False))


def run_variant(std_report, dev_reference, output, pooling, diffusion_t):
    torch.set_num_threads(4)
    device = torch.device("cuda")
    readout.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    config = load_json(readout.CONFIG)
    prior = std_report / "seed0_v01"
    for seed in config["seeds"]:
        run = std_report / f"seed{seed}_v01"
        progress = load_json(run / "progress.json")
        readout.require(progress["stage"] == "complete" and progress["dev_samples"] == 21186
                        and not progress["test_used"] and load_json(run / "config.json") == config,
                        f"Primary comparison seed {seed} is incomplete or differs in configuration")
    std_audit = load_json(prior / "input_audit.json")
    readout.require(std_audit["development_reference_sha256"] == readout.sha(dev_reference),
                    "Development reference differs from the std run")

    output.mkdir(parents=True, exist_ok=False)
    progress_path = output / "progress.json"
    status = dict(seed=SEED, pooling=pooling, diffusion_t=diffusion_t, random_noise=False,
                  test_used=False)

    def progress(stage, **values):
        data = dict(status, stage=stage, **values)
        readout.write_json(progress_path, data)
        print(json.dumps(data), flush=True)

    try:
        progress("checking_extraction")
        rows, train, dev, audit = readout.inspect_inputs(dev_reference)
        original_runner = std_report / "run_week3_c_readout_std_original.py"
        for relative, expected in std_audit["input_sha256"].items():
            path = readout.ROOT / relative
            if path == Path(readout.__file__):
                path = original_runner
            readout.require(readout.sha(path) == expected, f"Std protocol source changed: {relative}")
        positions = sorted(np.concatenate((train, dev)))
        expected_rows = [rows[i] for i in positions]
        records = sorted({row["record_id"] for row in expected_rows})
        record_file = output / "records.txt"
        record_file.write_text("\n".join(records) + "\n", encoding="utf-8")
        status.update(records=len(records), train_samples=len(train), dev_samples=len(dev))
        variant = f"one-second {pooling}, t={diffusion_t}"
        baseline = "one-second std, t=1"
        readout.write_json(output / "plan.json", dict(
            status, head_config=config, std_seed0_input_audit=std_audit,
            records_sha256=readout.sha(record_file), extractor_sha256=readout.sha(Path(extraction.__file__)),
            extension_runner_sha256=readout.sha(Path(__file__)), training_source_audit=audit,
            primary_comparison=f"{variant} + LFT vs {baseline} + LFT",
            secondary_comparison=f"{variant} + linear vs {baseline} + linear",
        ))
        checks = verify_reduction(rows[train[0]], readout.SOURCES, pooling, diffusion_t)
        readout.write_json(output / "extraction_preflight.json", checks)
        torch.cuda.empty_cache()

        sources = {name: output / "features" / name.lower() for name in readout.MODALITIES}
        split = readout.read_csv(readout.SPLIT)
        for modality, folder in sources.items():
            progress("extracting", modality=modality)
            subprocess.run([
                sys.executable, "-u", extraction.__file__, "--modality", modality.lower(),
                "--pooling", pooling, "--diffusion-t", str(diffusion_t),
                "--batch-size", "4", "--cache-kernels", "--records-file",
                str(record_file), "--output", str(folder),
            ], check=True)
            audit_variant_cache(modality, folder, record_file, expected_rows, split, pooling, diffusion_t)

        variant_rows = readout.read_csv(sources["EEG"] / "feature_index.csv")
        _, variant_train, variant_dev = readout.selected_positions(
            variant_rows, readout.read_csv(readout.SUBSETS), "label25"
        )
        train_dir = output / "train"
        train_dir.mkdir()
        variant_config = dict(config, pooling=pooling, diffusion_t=diffusion_t)
        readout.write_json(train_dir / "config.json", variant_config)
        readout.write_json(train_dir / "smoke.json", readout.smoke(
            config, variant_rows, variant_train, device, sources
        ))
        progress("training")
        readout.run_pair(variant_config, variant_rows, variant_train, variant_dev, SEED, device,
                         train_dir, sources)
        result_name = "mean_minus_std.json" if pooling == "mean" else "timestep_minus_t1.json"
        compare_to_baseline(output / result_name, prior, train_dir)
        progress("complete")
    except Exception as error:
        progress("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-reference", type=Path, required=True)
    parser.add_argument("--std-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_variant(args.std_report, args.dev_reference, args.output, "mean", 1)


if __name__ == "__main__":
    main()
