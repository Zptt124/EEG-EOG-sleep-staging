"""Compare EEG/EOG last5 readouts with the matched Week 3 C all20 runs."""
import argparse
import json
from pathlib import Path

import torch

import run_week3_c_readout as readout
from run_week3_c_lowbudget import load_json


def compare_layers(official, last5, all20, seed):
    comparison = {}
    identity = ["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]
    for head in ("C0", "C1"):
        variant = official.load_predictions(last5 / head / "predictions_val.csv")
        baseline = official.load_predictions(all20 / head / "predictions_val.csv")
        readout.require(variant[identity].equals(baseline[identity]),
                        f"Development identities differ for seed {seed}, {head}")
        comparison[head] = official.paired_bootstrap_delta(variant, baseline, n_boot=2000, seed=0)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--std-report", type=Path, required=True)
    parser.add_argument("--dev-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    readout.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    base = load_json(readout.CONFIG)
    readout.require(base["budget"] == "label25" and base["layers"] == list(range(20)),
                    "Unexpected all20 configuration")
    prior = load_json(args.std_report / "label5_v01/progress.json")
    readout.require(prior["stage"] == "complete" and prior["completed_seeds"] == base["seeds"],
                    "Five-person extension has not finished")
    layers = base["layers"][-5:]
    config = dict(base, layers=layers)
    args.output.mkdir(exist_ok=False)
    status = dict(budget=base["budget"], pooling="std", diffusion_t=1, layers=layers,
                  modalities=readout.MODALITIES, seeds=base["seeds"], completed_seeds=[],
                  test_used=False)

    def progress(stage, **values):
        value = dict(status, stage=stage, **values)
        readout.write_json(args.output / "progress.json", value)
        print(json.dumps(value), flush=True)

    try:
        progress("checking_inputs")
        rows, train, dev, audit = readout.inspect_inputs(args.dev_reference)
        reference = dict(audit["input_sha256"])
        reference["scripts/run_week3_c_readout.py"] = readout.sha(
            args.std_report / "run_week3_c_readout_std_original.py"
        )
        for seed in base["seeds"]:
            all20 = args.std_report / f"seed{seed}_v01"
            existing = load_json(all20 / "progress.json")
            prior_audit = load_json(all20 / "input_audit.json")
            readout.require(existing["stage"] == "complete" and not existing["test_used"]
                            and existing["dev_samples"] == len(dev)
                            and load_json(all20 / "config.json") == base
                            and prior_audit["input_sha256"] == reference
                            and prior_audit["development_reference_sha256"] ==
                            audit["development_reference_sha256"], f"All20 seed {seed} audit differs")
        status.update(train_samples=len(train), dev_samples=len(dev))
        audit.update(layers_used=layers, variant_runner_sha256=readout.sha(Path(__file__)))
        readout.write_json(args.output / "input_audit.json", audit)
        readout.write_json(args.output / "config.json", config)
        readout.write_json(args.output / "smoke.json", readout.smoke(config, rows, train, device))

        official = readout.official_module()
        comparisons = []
        for seed in config["seeds"]:
            progress("training_and_evaluation", current_seed=seed)
            folder = args.output / f"seed{seed}_v01"
            folder.mkdir()
            readout.write_json(folder / "config.json", config)
            readout.run_pair(config, rows, train, dev, seed, device, folder)
            all20 = args.std_report / folder.name
            paired = compare_layers(official, folder, all20, seed)
            readout.write_json(folder / "last5_minus_all20.json",
                               dict(seed=seed, paired=paired, test_used=False))
            row = {"seed": seed}
            for metric in ("macro_f1", "n1_f1", "accuracy"):
                for head in ("C0", "C1"):
                    row[f"{head}_{metric}_last5_minus_all20"] = paired[head][metric]["point"]
            for name, location in (("last5", folder), ("all20", all20)):
                result = load_json(location / "paired_results.json")
                row[f"C1_minus_C0_macro_f1_{name}"] = result["paired_C1_minus_C0"]["macro_f1"]["point"]
            comparisons.append(row)
            readout.write_csv(args.output / "layer_comparison.csv", comparisons)
            status["completed_seeds"].append(seed)
        progress("complete")
    except Exception as error:
        progress("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
