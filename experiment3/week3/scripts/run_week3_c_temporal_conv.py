"""Screen a temporal convolution readout with the all20 LFT parameter budget."""
import argparse
import json
from pathlib import Path

import torch

import run_week3_c_readout as readout
from run_week3_c_last5 import compare_layers as paired_readout_deltas
from run_week3_c_mean import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--std-report", type=Path, required=True)
    parser.add_argument("--dev-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    readout.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    segments = load_json(args.std_report / "segments_replicates_v01/progress.json")
    readout.require(segments["stage"] == "complete" and segments["completed_seeds"] == [1, 2],
                    "Six-segment replicates have not finished")
    base = load_json(readout.CONFIG)
    config = dict(base, temporal_conv=dict(width=512, classifier_width=5248))
    lft = args.std_report / "seed0_v01"
    six_segment = args.std_report / "segments_seed0_v01"
    prior = load_json(lft / "progress.json")
    readout.require(base["budget"] == "label25" and base["layers"] == list(range(20))
                    and load_json(lft / "config.json") == base and prior["stage"] == "complete"
                    and not prior["test_used"], "Seed 0 all20 protocol differs")
    args.output.mkdir(exist_ok=False)
    status = dict(seed=0, treatment="temporal_conv", budget=base["budget"], diffusion_t=1,
                  pooling="std", layers=base["layers"], test_used=False)

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
        baseline_audit = load_json(lft / "input_audit.json")
        readout.require(prior["dev_samples"] == len(dev)
                        and baseline_audit["input_sha256"] == reference
                        and baseline_audit["development_reference_sha256"] ==
                        audit["development_reference_sha256"], "Seed 0 source audit differs")
        status.update(train_samples=len(train), dev_samples=len(dev))
        smoke = readout.smoke(config, rows, train, device)
        conv_params = smoke["heads"]["C1"]["parameters"]
        lft_params = prior["parameters"]["C1"]
        readout.require(abs(conv_params - lft_params) / lft_params <= 0.01, "Parameter count differs by over 1%")
        audit["temporal_conv_runner_sha256"] = readout.sha(Path(__file__))
        readout.write_json(args.output / "input_audit.json", audit)
        readout.write_json(args.output / "config.json", config)
        readout.write_json(args.output / "smoke.json", smoke)

        progress("training_and_evaluation", conv_parameters=conv_params, lft_parameters=lft_params)
        output = args.output / "train"
        output.mkdir()
        readout.run_pair(config, rows, train, dev, 0, device, output)
        official = readout.official_module()
        for name, reference_run in (("30s_lft", lft), ("six_segment_lft", six_segment / "train")):
            paired = paired_readout_deltas(official, output, reference_run, 0)
            readout.write_json(args.output / f"temporal_conv_minus_{name}.json",
                               dict(seed=0, primary="C1", paired=paired, test_used=False))
        progress("complete")
    except Exception as error:
        progress("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
