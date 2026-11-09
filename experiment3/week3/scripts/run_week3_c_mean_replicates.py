"""Repeat the per-second mean vs std comparison with the remaining seeds."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_week3_c_readout as readout
from run_week3_c_last5 import compare_layers as paired_readout_deltas
from run_week3_c_mean import audit_variant_cache, load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--std-report", type=Path, required=True)
    parser.add_argument("--dev-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    readout.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    segments = args.std_report / "segments_seed0_v01"
    readout.require(load_json(segments / "progress.json")["stage"] == "complete"
                    and (segments / "segments_minus_30s.json").is_file(),
                    "The six-segment comparison is incomplete")
    pilot = args.std_report / "mean_seed0_v01"
    readout.require(load_json(pilot / "progress.json")["stage"] == "complete"
                    and load_json(pilot / "train/progress.json")["stage"] == "complete",
                    "The mean seed 0 run is incomplete")
    base = load_json(readout.CONFIG)
    config = dict(base, pooling="mean")
    readout.require(load_json(pilot / "train/config.json") == config, "Mean pilot head configuration differs")
    seeds = [seed for seed in base["seeds"] if seed != 0]
    args.output.mkdir(exist_ok=False)
    status = dict(seeds=seeds, completed_seeds=[], pooling="mean", diffusion_t=1,
                  budget=base["budget"], layers=base["layers"], test_used=False)

    def progress(stage, **values):
        value = dict(status, stage=stage, **values)
        readout.write_json(args.output / "progress.json", value)
        print(json.dumps(value), flush=True)

    try:
        progress("checking_inputs")
        std_rows, std_train, std_dev, audit = readout.inspect_inputs(args.dev_reference)
        expected = [std_rows[i] for i in sorted(np.concatenate((std_train, std_dev)))]
        sources = {name: pilot / "features" / name.lower() for name in readout.MODALITIES}
        split = readout.read_csv(readout.SPLIT)
        for name, folder in sources.items():
            audit_variant_cache(name, folder, pilot / "records.txt", expected, split, "mean", 1)
        rows = readout.read_csv(sources["EEG"] / "feature_index.csv")
        _, train, dev = readout.selected_positions(rows, readout.read_csv(readout.SUBSETS), base["budget"])
        reference_hashes = dict(audit["input_sha256"])
        reference_hashes["scripts/run_week3_c_readout.py"] = readout.sha(
            args.std_report / "run_week3_c_readout_std_original.py"
        )
        for seed in base["seeds"]:
            std = args.std_report / f"seed{seed}_v01"
            std_status = load_json(std / "progress.json")
            std_audit = load_json(std / "input_audit.json")
            readout.require(std_status["stage"] == "complete" and not std_status["test_used"]
                            and std_status["dev_samples"] == len(dev)
                            and load_json(std / "config.json") == base
                            and std_audit["input_sha256"] == reference_hashes
                            and std_audit["development_reference_sha256"] ==
                            audit["development_reference_sha256"], f"Std seed {seed} audit differs")
        status.update(train_samples=len(train), dev_samples=len(dev))
        audit["mean_source_sha256"] = {
            str(path.relative_to(readout.ROOT)): readout.sha(path)
            for path in [pilot / "records.txt", pilot / "plan.json",
                         *(folder / file for folder in sources.values()
                           for file in ("feature_config.json", "feature_index.csv"))]
        }
        audit["replication_runner_sha256"] = readout.sha(Path(__file__))
        readout.write_json(args.output / "input_audit.json", audit)
        readout.write_json(args.output / "config.json", config)
        readout.write_json(args.output / "smoke.json", readout.smoke(config, rows, train, device, sources))

        official = readout.official_module()
        comparisons = []
        for seed in base["seeds"]:
            std = args.std_report / f"seed{seed}_v01"
            if seed == 0:
                mean = pilot / "train"
                paired = load_json(pilot / "mean_minus_std.json")["paired"]
            else:
                progress("training_and_evaluation", current_seed=seed)
                mean = args.output / f"seed{seed}_v01"
                mean.mkdir()
                readout.write_json(mean / "config.json", config)
                readout.run_pair(config, rows, train, dev, seed, device, mean, sources)
                paired = paired_readout_deltas(official, mean, std, seed)
                readout.write_json(mean / "mean_minus_std.json", dict(seed=seed, primary="C1",
                                                                     paired=paired, test_used=False))
            for head, metrics in paired.items():
                for metric in ("macro_f1", "n1_f1", "accuracy"):
                    estimate = metrics[metric]
                    comparisons.append(dict(seed=seed, head=head, metric=metric, **{
                        key: estimate[key] for key in ("point", "lo", "hi")
                    }))
            readout.write_csv(args.output / "pooling_comparison.csv", comparisons)
            if seed != 0:
                status["completed_seeds"].append(seed)
        progress("complete")
    except Exception as error:
        progress("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
