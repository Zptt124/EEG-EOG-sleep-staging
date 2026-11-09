"""Screen six shared five-second LFT readouts against the 30-second LFT."""
import argparse
import json
from pathlib import Path

import torch

import run_week3_c_readout as readout
from run_week3_c_last5 import compare_layers as paired_readout_deltas
from run_week3_c_mean import load_json


@torch.inference_mode()
def check_segment_mapping(config, rows, train, device):
    """Check vectorized output against six explicit five-second forwards."""
    _, model = readout.build_heads(config, 0, device).values()
    model.eval()
    data = readout.CachedEpochs(rows, train, config["layers"])
    raw, _ = next(data.groups(2, 1))
    x = torch.as_tensor(raw, device=device, dtype=torch.float32)
    vectorized = model(x)
    direct = torch.stack([
        readout.CachedLFT.forward(model, segment)
        for segment in x.split(5, dim=3)
    ], dim=1).mean(dim=1)
    torch.testing.assert_close(vectorized, direct, rtol=1e-5, atol=3e-5)
    return dict(input_shape=list(x.shape), logits_shape=list(vectorized.shape),
                max_abs_diff=float((vectorized - direct).abs().max().item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--std-report", type=Path, required=True)
    parser.add_argument("--dev-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", action="store_true", help="Run the remaining seeds after seed 0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    readout.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    base = load_json(readout.CONFIG)
    readout.require(base["budget"] == "label25" and base["layers"] == list(range(20)),
                    "Unexpected all20 configuration")
    last5 = load_json(args.std_report / "last5_v01/progress.json")
    readout.require(last5["stage"] == "complete" and last5["completed_seeds"] == base["seeds"],
                    "Last5 comparison has not finished")
    config = dict(base, lft_segment_seconds=5, lft_aggregation="mean_logits")
    seeds = [seed for seed in base["seeds"] if seed != 0] if args.replicates else [0]
    if args.replicates:
        pilot = args.std_report / "segments_seed0_v01"
        mean = load_json(args.std_report / "mean_replicates_v02/progress.json")
        readout.require(mean["stage"] == "complete" and mean["completed_seeds"] == seeds,
                        "Mean pooling replicates have not finished")
        readout.require(load_json(pilot / "progress.json")["stage"] == "complete"
                        and load_json(pilot / "config.json") == config,
                        "Six-segment seed 0 is incomplete or differs in configuration")
    args.output.mkdir(exist_ok=False)
    status = dict(budget=base["budget"], diffusion_t=1, pooling="std",
                  layers=base["layers"], lft_segments=6, lft_segment_seconds=5,
                  lft_aggregation="mean_logits", test_used=False)
    if args.replicates:
        status.update(seeds=seeds, completed_seeds=[])
    else:
        status["seed"] = 0

    def progress(stage, **values):
        value = dict(status, stage=stage, **values)
        readout.write_json(args.output / "progress.json", value)
        print(json.dumps(value), flush=True)

    try:
        progress("checking_inputs")
        rows, train, dev, audit = readout.inspect_inputs(args.dev_reference)
        reference_hashes = dict(audit["input_sha256"])
        reference_hashes["scripts/run_week3_c_readout.py"] = readout.sha(
            args.std_report / "run_week3_c_readout_std_original.py"
        )
        for seed in seeds:
            prior = args.std_report / f"seed{seed}_v01"
            prior_status = load_json(prior / "progress.json")
            baseline_audit = load_json(prior / "input_audit.json")
            readout.require(prior_status["stage"] == "complete" and not prior_status["test_used"]
                            and prior_status["dev_samples"] == len(dev)
                            and load_json(prior / "config.json") == base
                            and baseline_audit["input_sha256"] == reference_hashes
                            and baseline_audit["development_reference_sha256"] ==
                            audit["development_reference_sha256"], f"Seed {seed} baseline differs")
        if args.replicates:
            pilot_audit = load_json(pilot / "input_audit.json")
            readout.require(pilot_audit["input_sha256"] == audit["input_sha256"]
                            and pilot_audit["development_reference_sha256"] ==
                            audit["development_reference_sha256"], "Six-segment pilot inputs differ")
        status.update(train_samples=len(train), dev_samples=len(dev))
        audit["segment_runner_sha256"] = readout.sha(Path(__file__))
        readout.write_json(args.output / "input_audit.json", audit)
        readout.write_json(args.output / "config.json", config)
        checks = check_segment_mapping(config, rows, train, device)
        smoke = readout.smoke(config, rows, train, device)
        readout.write_json(args.output / "smoke.json", dict(segment_mapping=checks, **smoke))

        official = readout.official_module()
        comparisons = []
        if args.replicates:
            result = load_json(pilot / "segments_minus_30s.json")
            comparisons.append(segment_summary(0, result["paired"]))
            readout.write_csv(args.output / "segment_comparison.csv", comparisons)
        for seed in seeds:
            progress("training_and_evaluation", current_seed=seed)
            folder = args.output / (f"seed{seed}_v01" if args.replicates else "train")
            folder.mkdir()
            readout.write_json(folder / "config.json", config)
            readout.run_pair(config, rows, train, dev, seed, device, folder)
            prior = args.std_report / f"seed{seed}_v01"
            paired = paired_readout_deltas(official, folder, prior, seed)
            result_folder = folder if args.replicates else args.output
            readout.write_json(result_folder / "segments_minus_30s.json",
                               dict(seed=seed, primary="C1", paired=paired, test_used=False))
            if args.replicates:
                comparisons.append(segment_summary(seed, paired))
                readout.write_csv(args.output / "segment_comparison.csv", comparisons)
                status["completed_seeds"].append(seed)
        progress("complete")
    except Exception as error:
        progress("failed", error=str(error))
        raise


def segment_summary(seed, paired):
    row = {"seed": seed}
    for metric in ("macro_f1", "n1_f1", "accuracy"):
        estimate = paired["C1"][metric]
        for field in ("point", "lo", "hi"):
            row[f"{metric}_{field}"] = estimate[field]
    return row


if __name__ == "__main__":
    main()
