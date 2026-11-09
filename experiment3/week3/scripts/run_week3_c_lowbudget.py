"""Repeat the std, t=1 C0/C1 comparison with the fixed five-person budget."""
import argparse
import json
from pathlib import Path

import torch

import run_week3_c_readout as readout


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--std-report", type=Path, required=True)
    parser.add_argument("--dev-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    readout.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    baseline_config = load_json(readout.CONFIG)
    readout.require(baseline_config["budget"] == "label25", "Unexpected baseline budget")
    config = dict(baseline_config, budget="label10",
                  normalization="per modality/layer/hidden, shared across time, fitted on the 5 labeled training participants")
    args.output.mkdir(exist_ok=False)
    progress_path = args.output / "progress.json"
    status = dict(budget="label10", labeled_subjects=5, pooling="std", diffusion_t=1,
                  seeds=config["seeds"], completed_seeds=[], test_used=False)

    def progress(stage, **values):
        data = dict(status, stage=stage, **values)
        readout.write_json(progress_path, data)
        print(json.dumps(data), flush=True)

    try:
        progress("checking_inputs")
        rows, _, dev, audit = readout.inspect_inputs(args.dev_reference)
        subjects, train, _ = readout.selected_positions(
            rows, readout.read_csv(readout.SUBSETS), "label10"
        )
        readout.require(subjects < set(audit["labeled_subjects"]), "Five-person budget is not nested")
        baselines = {}
        source_hashes = audit["input_sha256"]
        for seed in config["seeds"]:
            baseline = args.std_report / f"seed{seed}_v01"
            baseline_status = load_json(baseline / "progress.json")
            baseline_audit = load_json(baseline / "input_audit.json")
            readout.require(baseline_status["stage"] == "complete" and
                            baseline_status["dev_samples"] == len(dev) and
                            not baseline_status["test_used"] and
                            load_json(baseline / "config.json") == baseline_config,
                            f"Fourteen-person seed {seed} is incomplete or differs in configuration")
            readout.require(
                baseline_audit["development_reference_sha256"] == audit["development_reference_sha256"],
                f"Development reference differs for seed {seed}",
            )
            for name, digest in baseline_audit["input_sha256"].items():
                if name == "scripts/run_week3_c_readout.py":
                    actual = readout.sha(args.std_report / "run_week3_c_readout_std_original.py")
                else:
                    actual = source_hashes[name]
                readout.require(actual == digest, f"Input source differs for seed {seed}: {name}")
            baselines[seed] = load_json(baseline / "paired_results.json")
        audit.update(labeled_subjects=sorted(subjects), labeled_epochs=len(train), budget="label10")
        status.update(train_samples=len(train), dev_samples=len(dev))
        readout.write_json(args.output / "input_audit.json", audit)
        readout.write_json(args.output / "config.json", config)
        readout.write_json(args.output / "smoke.json", readout.smoke(config, rows, train, device))

        comparisons = []
        for seed in config["seeds"]:
            progress("training_and_evaluation", current_seed=seed)
            folder = args.output / f"seed{seed}_v01"
            folder.mkdir()
            readout.write_json(folder / "config.json", config)
            readout.run_pair(config, rows, train, dev, seed, device, folder)
            low = load_json(folder / "paired_results.json")
            high = baselines[seed]
            row = {"seed": seed}
            for metric in ("macro_f1", "n1_f1", "accuracy"):
                for name in ("C0", "C1"):
                    row[f"{name}_{metric}_5"] = low["metrics"][name][metric]
                    row[f"{name}_{metric}_14"] = high["metrics"][name][metric]
                row[f"C1_minus_C0_{metric}_5"] = low["paired_C1_minus_C0"][metric]["point"]
                row[f"C1_minus_C0_{metric}_14"] = high["paired_C1_minus_C0"][metric]["point"]
            comparisons.append(row)
            readout.write_csv(args.output / "budget_comparison.csv", comparisons)
            status["completed_seeds"].append(seed)
        progress("complete")
    except Exception as error:
        progress("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
