"""Refit all twelve W4 C heads at their training-side selected epochs."""

import argparse
import fcntl
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch

import run_week3_c_readout as c
from run_week3_c_inner_cv import fit_phase, save_checkpoint
from run_week4_c_screening import stage_arm


ARMS = ("random_init", "pretrained")


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def run(args):
    torch.set_num_threads(4)
    device = torch.device("cuda")
    c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    b_path = args.b_checkout.resolve() / "scripts/27_b_w3_linear_probe.py"
    sys.path.insert(0, str(b_path.parent))
    b = runpy.run_path(str(b_path))
    screen = args.root / "screening_1s_seed0"
    baseline = load(screen / "plan.json")
    expected = dict(baseline)
    expected.pop("head_seed")
    screenings = {}
    for seed in (0, 1, 2):
        folder = args.root / f"screening_1s_seed{seed}"
        plan = load(folder / "plan.json")
        c.require(plan.pop("head_seed") == seed and plan == expected,
                  f"Screening seed {seed} has a different protocol")
        c.require(load(folder / "progress.json")["stage"] == "complete",
                  f"Screening seed {seed} incomplete")
        screenings[seed] = load(folder / "summary.json")
    c.require(baseline["pooling"] == 1 and not baseline["dev_used"]
              and not baseline["test_used"] and c.sha(b_path) == baseline["linear_probe_sha256"],
              "Unexpected refit protocol")
    plan = dict(screening_plan=baseline, selections=screenings,
                screening_sha256={str(seed): c.sha(args.root / f"screening_1s_seed{seed}" / "summary.json")
                                  for seed in screenings},
                operation="full five-subject refit at OOF-selected epochs",
                pooling_status="1s continuation pending A final pooling comparison",
                dev_used=False, test_used=False)
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        c.require(load(plan_path) == json.loads(json.dumps(plan)), "Refit inputs changed")
    else:
        c.write_json(plan_path, plan)
    cache = args.root / "features"
    for relative, digest in baseline["cache_sha256"].items():
        arm, record = relative.split("/")
        c.require(c.sha(cache / arm / f"{record}.npz") == digest, "Feature cache changed")
    records = baseline["cache"]["records"]
    data = {arm: stage_arm(cache, arm, records, 1, screen / "staged" / arm) for arm in ARMS}
    c.require(data[ARMS[0]][0] == data[ARMS[1]][0], "Backbone sample alignment changed")
    config = baseline["lft"]
    completed = []
    for seed in screenings:
        for head in ("lft", "linear"):
            for arm in ARMS:
                name = f"seed{seed}_{head}_{arm}"
                folder = args.output / name
                folder.mkdir(exist_ok=True)
                receipt = folder / "complete.json"
                if receipt.exists():
                    c.require(c.sha(folder / "final_head.pt") == load(receipt)["sha256"],
                              f"Refit model changed: {name}")
                    completed.append(name)
                    continue
                epochs = screenings[seed]["arms"][f"{head}_{arm}"]["epoch"]
                rows, pooled, sources = data[arm]

                def progress(stage, **details):
                    state = dict(run=name, stage=stage, completed=completed, **details)
                    c.write_json(args.output / "progress.json", state)
                    if not details.get("updates"):
                        print(json.dumps(state), flush=True)

                progress("starting_refit", epochs=epochs)
                start = time.perf_counter()
                if head == "lft":
                    # B's CPU linear trainer enables this global flag. Restore
                    # the LFT setting used by every screening run before refit.
                    torch.use_deterministic_algorithms(False)
                    fit_data = c.CachedEpochs(rows, np.arange(len(rows)), config["layers"], sources)
                    model, mean, std, _, _ = fit_phase(
                        config, seed, device, fit_data, folder / "normalization.pt",
                        folder, epochs, progress,
                    )
                    state = dict(model=model.state_dict(), mean=mean.cpu(), std=std.cpu(),
                                 config=config, seed=seed, epoch=epochs, head="lft")
                    del model
                else:
                    (mean, std), (fit,) = b["standardize"](pooled)
                    labels = np.array([row["y_true"] for row in rows], dtype=np.int64)
                    model, _, updates = b["train_head"](fit, labels, seed, epochs)
                    state = dict(model=model.state_dict(), mean=mean, std=std, seed=seed,
                                 epoch=epochs, head="linear", updates=updates)
                save_checkpoint(folder / "final_head.pt", state)
                c.write_json(receipt, dict(epoch=epochs, sha256=c.sha(folder / "final_head.pt"),
                                          seconds=time.perf_counter() - start,
                                          train_samples=len(rows), dev_used=False, test_used=False))
                completed.append(name)
                progress("refit_complete")
    c.write_json(args.output / "summary.json", dict(
        runs={name: load(args.output / name / "complete.json") for name in completed},
        dev_used=False, test_used=False))
    c.write_json(args.output / "progress.json", dict(stage="complete", completed=completed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b-checkout", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except Exception as error:
            c.write_json(args.output / "progress.json",
                         dict(stage="failed", error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    main()
