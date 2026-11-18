"""Lock the W4 C 1s configuration and score the twelve refitted heads."""

from datetime import datetime
import argparse
import fcntl
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch

import run_week3_c_readout as c
from run_week4_c_screening import stage_arm


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def run(args):
    torch.set_num_threads(4)
    scripts = args.b_checkout.resolve() / "scripts"
    sys.path.insert(0, str(scripts))
    b = runpy.run_path(str(scripts / "27_b_w3_linear_probe.py"))
    adjustment = runpy.run_path(str(scripts / "28_b_n1_score_adjust.py"))
    evaluator, common = b["ev"], b["w"]
    refit = args.root / "refit_1s_v01"
    receipt = load(refit / "summary.json")
    c.require(load(refit / "progress.json")["stage"] == "complete"
              and len(receipt["runs"]) == 12, "All twelve refits must be complete")
    source_plan = load(args.root / "screening_1s_seed0/plan.json")
    alphas = {}
    for name, item in receipt["runs"].items():
        c.require(c.sha(refit / name / "final_head.pt") == item["sha256"]
                  and not item["dev_used"] and not item["test_used"],
                  f"Refit model or provenance changed: {name}")
        seed_tag, head, arm = name.split("_", 2)
        oof_path = args.root / f"screening_1s_{seed_tag}" / f"{head}_{arm}/oof_train.csv"
        alpha, oof_f1, raw_f1 = adjustment["pick_alpha"](pd.read_csv(oof_path))
        alphas[name] = dict(alpha=alpha, oof_adjusted_n1_f1=oof_f1,
                            oof_raw_n1_f1=raw_f1, oof_sha256=c.sha(oof_path))
    lock = dict(pooling_seconds=1, diffusion_t=1, layers=12, hidden=64, budget="label10",
                backbone_seed=0, head_seeds=[0, 1, 2], feature_weights="ema_model",
                backbone_steps=dict(random_init=0, pretrained=30000),
                basis="A recommended retaining 1s std; paired three-seed training-side CV completed",
                refit_summary_sha256=c.sha(refit / "summary.json"), alpha_selection=alphas,
                development_reference_sha256=c.sha(args.b_checkout / "manifests/val_reference_v1.csv"),
                test_used=False)
    lock_path = args.output / "configuration_lock.json"
    if lock_path.exists():
        c.require(load(lock_path) == lock, "Locked final evaluation configuration changed")
    else:
        c.write_json(lock_path, lock)

    def progress(stage, **details):
        value = dict(stage=stage, updated_at=datetime.now().astimezone().isoformat(), **details)
        c.write_json(args.output / "progress.json", value)
        print(json.dumps(value), flush=True)

    progress("extracting_development_features")
    cache = args.root / "features_development_1s"
    if not (cache / "DONE").exists():
        workspace = Path(__file__).resolve().parents[2]
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).with_name("extract_week4_c_screening.py")),
            "--d-checkout", str(args.d_checkout.resolve()),
            "--archive", str(workspace / "output/week3_cd_lft/checkpoints.partial.zip"),
            "--output", str(cache), "--batch-size", "16", "--scope", "development",
            "--pool-seconds", "1",
        ], check=True)
    cache_plan = load(cache / "plan.json")
    c.require(cache_plan["mode"] == "development" and cache_plan["pooling_seconds"] == [1]
              and not cache_plan["test_used"], "Unexpected development extraction protocol")
    data = {arm: stage_arm(cache, arm, cache_plan["records"], 1,
                           args.root / "staged_development" / arm)
            for arm in ("random_init", "pretrained")}
    rows = data["random_init"][0]
    c.require(rows == data["pretrained"][0] and len(rows) == 21186,
              "Development backbone sample alignment changed")
    reference = pd.read_csv(args.b_checkout / "manifests/val_reference_v1.csv")
    evaluator.check_against_reference(pd.DataFrame(rows), reference)
    scores, predictions = [], {}
    for name in receipt["runs"]:
        folder = args.output / "runs" / name
        folder.mkdir(parents=True, exist_ok=True)
        seed_tag, head, arm = name.split("_", 2)
        prediction_path = folder / "predictions_val.csv"
        progress("scoring_frozen_head", run=name)
        if not prediction_path.exists():
            state = torch.load(refit / name / "final_head.pt", map_location="cpu", weights_only=False)
            _, pooled, sources = data[arm]
            if head == "lft":
                model = c.build_heads(source_plan["lft"], int(seed_tag[4:]), torch.device("cuda"))["C1"]
                model.load_state_dict(state["model"], strict=True)
                model.eval()
                dev_data = c.CachedEpochs(rows, np.arange(len(rows)), source_plan["lft"]["layers"], sources)
                probability = c.predict_development(
                    {"C1": model}, dev_data, source_plan["lft"], torch.device("cuda"),
                    state["mean"].cuda(), state["std"].cuda(),
                )["C1"]
                del model
            else:
                x = torch.from_numpy((pooled - state["mean"]) / state["std"])
                model = torch.nn.Linear(x.shape[1], 5)
                model.load_state_dict(state["model"], strict=True)
                probability = common.predict_probs(model, lambda ids: x[ids], len(x))
            c.require(probability.shape == (21186, 5) and np.isfinite(probability).all(),
                      "Invalid development probabilities")
            partial = prediction_path.with_suffix(".partial")
            common.write_prediction_table(partial, rows, probability)
            partial.replace(prediction_path)
        frame = evaluator.load_predictions(prediction_path)
        evaluator.check_against_reference(frame, reference)
        metric = evaluator.evaluate(prediction_path, reference=args.b_checkout / "manifests/val_reference_v1.csv")
        after = adjustment["describe"](adjustment["adjusted"](frame, alphas[name]["alpha"]))
        metric.update(alphas[name], adjusted=after,
                      n1_ap=float(average_precision_score(frame["y_true"] == 1, frame["p_N1"])),
                      model_sha256=receipt["runs"][name]["sha256"], test_used=False)
        c.write_json(folder / "metrics.json", metric)
        evaluator.per_subject_table(frame).to_csv(folder / "by_subject.csv", index=False)
        scores.append(dict(run=name, seed=int(seed_tag[4:]), head=head, backbone=arm,
                           epoch=receipt["runs"][name]["epoch"], macro_f1=metric["macro_f1"],
                           n1_f1=metric["n1_f1"], n1_ap=metric["n1_ap"], alpha=metric["alpha"],
                           adjusted_macro_f1=after["macro_f1"], adjusted_n1_f1=after["n1_f1"]))
        predictions[name] = frame
    pd.DataFrame(scores).to_csv(args.output / "metrics_summary.csv", index=False)
    pooled = {}
    for head in ("linear", "lft"):
        for arm in ("random_init", "pretrained"):
            frames = []
            for seed in (0, 1, 2):
                frame = predictions[f"seed{seed}_{head}_{arm}"].copy()
                frame["sample_id"] = f"seed{seed}_" + frame["sample_id"]
                frames.append(frame)
            pooled[f"{head}_{arm}"] = pd.concat(frames, ignore_index=True)
    comparisons = {
        f"{head}_pretrained_minus_random": evaluator.paired_bootstrap_delta(
            pooled[f"{head}_pretrained"], pooled[f"{head}_random_init"], 2000, 0)
        for head in ("linear", "lft")
    }
    comparisons["pretrained_lft_minus_linear"] = evaluator.paired_bootstrap_delta(
        pooled["lft_pretrained"], pooled["linear_pretrained"], 2000, 0)
    c.write_json(args.output / "paired_comparisons.json", comparisons)
    c.write_json(args.output / "summary.json", dict(
        runs=scores, paired=comparisons, completed_at=datetime.now().astimezone().isoformat(),
        configuration_lock_sha256=c.sha(lock_path), test_used=False))
    progress("complete", runs=len(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b-checkout", type=Path, required=True)
    parser.add_argument("--d-checkout", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except Exception as error:
            c.write_json(args.output / "progress.json",
                         dict(stage="failed", error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    main()
