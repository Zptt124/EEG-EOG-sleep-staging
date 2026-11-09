"""Screen t=2 and t=3 on clean inputs against the matched t=1 readouts."""
import argparse
from pathlib import Path
import shutil

from run_week3_c_mean import load_json, run_variant
from run_week3_c_readout import require, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--std-report", type=Path, required=True)
    parser.add_argument("--dev-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completed_mean = args.std_report / "mean_seed0_v01"
    require(load_json(completed_mean / "progress.json")["stage"] == "complete"
            and (completed_mean / "mean_minus_std.json").is_file(),
            "Mean pooling extension has not finished")
    steps = (2, 3)
    bytes_per_step = sum(
        load_json(completed_mean / "features" / modality / "summary.json")["total_feature_bytes"]
        for modality in ("eeg", "eog")
    )
    require(shutil.disk_usage(args.output.parent).free >= len(steps) * bytes_per_step + 2**31,
            "Insufficient disk space for the two EEG/EOG feature caches")
    args.output.mkdir(exist_ok=False)
    finished = []
    status = dict(seed=0, baseline_t=1, timesteps=steps, pooling="std",
                  random_noise=False, test_used=False)
    progress_path = args.output / "progress.json"
    for timestep in steps:
        write_json(progress_path, dict(status, stage="running", current_t=timestep,
                                       completed_t=finished))
        try:
            run_variant(args.std_report, args.dev_reference, args.output / f"t{timestep}",
                        "std", timestep)
        except Exception as error:
            write_json(progress_path, dict(status, stage="failed", current_t=timestep,
                                           completed_t=finished, error=str(error)))
            raise
        finished.append(timestep)
    write_json(progress_path, dict(status, stage="complete", completed_t=finished))


if __name__ == "__main__":
    main()
