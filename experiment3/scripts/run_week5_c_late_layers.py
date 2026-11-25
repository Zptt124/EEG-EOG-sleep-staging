"""Run C's W5 30k all-layer/last-three comparison with B's W4 probe."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TEAM = Path(os.environ.get("W5_C_TEAM_SOURCE", ROOT / "tmp/c_w4_xmodal_review_20261119")).resolve()
ASSETS = Path(os.environ.get("W5_C_ASSETS", ROOT / "tmp/w4_crossmodal_assets_20261119")).resolve()
OUTPUTS = Path(os.environ.get("W5_C_OUTPUTS", ROOT / "experiment3/runs")).resolve()
INDEX_SHA = "d7c0e4db60b69d3f2a7aacf2e5b97416e3688c6b189b637b31438e4ea873d08e"
MODALITIES = ("eeg", "eog")
BUDGETS = ("label10", "label25")
SEEDS = (0, 1, 2)
PROBE = TEAM / "scripts/34_b_w4_probe.py"
SUBSETS = TEAM / "manifests/label_subsets_v2.csv"
REFERENCE = TEAM / "manifests/val_reference_v1.csv"
IDS = ["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]


@dataclass(frozen=True)
class RunSpec:
    mode: str
    backbone: str
    feedback_layer: int
    archive_name: str
    archive_sha: str
    checkpoint_sha: str

    @property
    def out(self):
        return OUTPUTS / f"W5_C_{self.mode}_30k_layer_probe_v01"

    @property
    def arms(self):
        return {f"{self.mode}30k_all12": (1, 12), f"{self.mode}30k_last3": (10, 12)}


SPECS = {
    spec.mode: spec for spec in (
        RunSpec("plain", "W4_D_main_m64x12_seed0_v01", 0, "w4_d_main30k_feat_pool1s_v01.zip",
                "9072c803208fbd1c3abfb848e11124d4c130bf013887b0e10698d33b883cb013",
                "21c8514ea7761ef383d70b3e26ac19ea33ad0e7ccacfe690812ad359b0118e5b"),
        RunSpec("selfcond", "W4_D_selfcond_l9_mean_30k_v01", 9, "w4_d_selfcond30k_feat_pool1s_v01.zip",
                "7a1f8cdf2ebb68a6d8294570d83dd4d3662baa8d57cf5b3b2006f98e892af348",
                "453ecba128875418729a1927008057cd1abe709fc35dea7cbba1fba74a75b013"),
    )
}


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_csv_bytes(body):
    return list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))


def prepare_features(spec):
    features = spec.out / "features"
    asset = ASSETS / spec.archive_name
    if (features / "receipt.json").exists():
        receipt = json.loads((features / "receipt.json").read_text())
        if (receipt["source_sha256"], receipt["checkpoint_sha256"], receipt["source_index_sha256"]) != (
                spec.archive_sha, spec.checkpoint_sha, INDEX_SHA):
            raise ValueError("Feature receipt differs from expected source")
        for name, expected in receipt["prepared_sha256"].items():
            if sha(features / name) != expected:
                raise ValueError(f"Prepared input changed: {name}")
        return
    if sha(asset) != spec.archive_sha:
        raise ValueError(f"W4 feature archive checksum mismatch: {spec.mode}")
    features.mkdir(parents=True, exist_ok=True)
    files = []
    timing = {}
    with zipfile.ZipFile(asset) as archive:
        prefix = next(name.removesuffix("features_info.json") for name in archive.namelist()
                      if name.endswith("/features_info.json"))
        meta_body = archive.read(prefix + "features_info.json")
        meta = json.loads(meta_body)
        index_body = archive.read(prefix + "index.csv")
        if (meta["checkpoint_sha256"] != spec.checkpoint_sha or meta["weights"] != "ema_model"
                or meta["run_config_name"] != spec.backbone or meta["step"] != 30000
                or meta["switches"]["self_cond_layer"] != spec.feedback_layer
                or meta["switches"]["cross_modal"] != "none" or meta["pool_sec"] != 1
                or meta["index_sha256"] != INDEX_SHA or hashlib.sha256(index_body).hexdigest() != INDEX_SHA):
            raise ValueError("Wrong checkpoint, mode, or epoch index")
        rows = read_csv_bytes(index_body)
        ids = np.asarray([row["sample_id"] for row in rows])
        counts = Counter(row["split"] for row in rows)
        if len(set(ids)) != len(ids) or counts != {"train": 32310, "val": 21186}:
            raise ValueError(f"Unexpected feature coverage: {counts}")
        if any("row_index" in row and int(row["row_index"]) != i for i, row in enumerate(rows)):
            raise ValueError("Source array order differs from row_index")
        for name, body in (("index.csv", index_body), ("source_features_info.json", meta_body)):
            (features / name).write_bytes(body)
            files.append(name)
        for modality in MODALITIES:
            stem = f"{modality}_std"
            body = archive.read(prefix + stem + ".npy")
            if hashlib.sha256(body).hexdigest() != meta["files"][stem]["sha256"]:
                raise ValueError(f"Source feature checksum mismatch: {stem}")
            data = np.load(io.BytesIO(body), allow_pickle=False)
            del body
            if data.shape != (len(ids), 768) or data.dtype != np.float32 or not np.isfinite(data).all():
                raise ValueError(f"Wrong shape, dtype, or non-finite: {stem}")
            for arm, (first, last) in spec.arms.items():
                name = f"{arm}_{modality}.npz"
                selected = np.ascontiguousarray(data[:, (first - 1) * 64:last * 64])
                start = time.perf_counter()
                np.savez(features / name, sample_id=ids, x=selected)
                timing[name] = dict(seconds=time.perf_counter() - start, bytes=(features / name).stat().st_size)
                files.append(name)
            del data
    receipt = dict(source=str(asset.relative_to(ROOT)), source_sha256=spec.archive_sha,
                   checkpoint_sha256=spec.checkpoint_sha, source_run=meta["run_config_name"],
                   source_index_sha256=meta["index_sha256"], counts=dict(counts),
                   layers=spec.arms, columns_per_layer=64, preprocessing=meta["readout"],
                   prepared_sha256={name: sha(features / name) for name in files},
                   cache_write_observations=timing, test_used=False,
                   protocol_file_sha256={path.name: sha(path) for path in (
                       PROBE, TEAM / "scripts/27_b_w3_linear_probe.py", TEAM / "scripts/28_b_n1_score_adjust.py",
                       TEAM / "scripts/29_b_w3_evaluate.py", TEAM / "scripts/w2_common.py",
                       TEAM / "src/evaluation/evaluate.py", SUBSETS, REFERENCE,
                   )})
    (features / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Bound W4 {spec.mode} epoch features and prepared all12/last3 caches", flush=True)


def run_path(spec, arm, budget, seed):
    return spec.out / "probes" / f"W4_C_W5_{arm}_{budget}_sseed2024_seed{seed}_v01"


def run_probe(spec, arm, budget, seed):
    destination = run_path(spec, arm, budget, seed)
    if (destination / "config.yaml").is_file() and (destination / "metrics.json").is_file():
        return destination
    if destination.exists():
        raise ValueError(f"Incomplete run directory requires review: {destination}")
    config = dict(week=5, backbone=spec.backbone, backbone_seed=0,
                  step=30000, checkpoint_sha256=spec.checkpoint_sha, self_conditioning=bool(spec.feedback_layer),
                  layers=list(range(spec.arms[arm][0], spec.arms[arm][1] + 1)), diffusion_t=1,
                  pool="per-second std, mean over 30 positions", input_noise=False)
    features = spec.out / "features"
    cmd = [sys.executable, "-u", str(PROBE), "--group", "C_W5", "--arm", arm,
           "--budget", budget, "--seed", str(seed), "--modalities", "eeg+eog",
           "--index", str(features / "index.csv"), "--subsets", str(SUBSETS),
           "--runs-root", str(destination.parent), "--feature-config", json.dumps(config)]
    for modality in MODALITIES:
        cmd += [f"--{modality}-pooled", str(features / f"{arm}_{modality}.npz")]
    log = spec.out / "logs" / (destination.name + ".log")
    print(f"START {destination.name}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(cmd, cwd=TEAM, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Probe failed: {destination.name}\n{log.read_text()}")
    print(log.read_text().strip(), flush=True)
    return destination


def load_evaluator():
    sys.path.insert(0, str(TEAM / "scripts"))
    spec = importlib.util.spec_from_file_location("w5_c_b_evaluate", TEAM / "scripts/29_b_w3_evaluate.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


def measure_readout(spec):
    import torch
    from torch.utils.benchmark import Timer

    threads, batch = 4, 512
    torch.set_num_threads(threads)
    features_dir = spec.out / "features"
    index = pd.read_csv(features_dir / "index.csv")
    positions = index.index[index["split"] == "val"][:batch].to_numpy()
    receipt = json.loads((features_dir / "receipt.json").read_text())
    rows = []
    for arm in spec.arms:
        columns, cache_bytes, cache_write_seconds = [], 0, 0.0
        for modality in MODALITIES:
            name = f"{arm}_{modality}.npz"
            with np.load(features_dir / name, allow_pickle=False) as features:
                columns.append(features["x"][positions])
            observation = receipt["cache_write_observations"][name]
            cache_bytes += observation["bytes"]
            cache_write_seconds += observation["seconds"]
        data = np.concatenate(columns, axis=1)
        state = torch.load(run_path(spec, arm, "label10", 0) / "final.ckpt", map_location="cpu", weights_only=False)
        features = torch.from_numpy((data - state["mean"]) / state["std"])
        model = torch.nn.Linear(data.shape[1], 5).eval()
        model.load_state_dict(state["state"])
        with torch.inference_mode():
            timing = Timer("model(features)", globals={"model": model, "features": features},
                           num_threads=threads).blocked_autorange(min_run_time=0.7)
        rows.append(dict(arm=arm, total_dims=data.shape[1], linear_parameters=sum(p.numel() for p in model.parameters()),
                         cache_bytes=cache_bytes, cache_write_seconds_single_observation=cache_write_seconds,
                         head_batch_size=batch, head_cpu_threads=threads, head_forward_median_us=timing.median * 1e6,
                         head_forward_iqr_us=timing.iqr * 1e6))
    pd.DataFrame(rows).to_csv(spec.out / "analysis/readout_cost.csv", index=False)


def collect(spec, evaluator, budgets, seeds, pilot):
    analysis = spec.out / "analysis"
    entries, frames, metrics = [], {}, []
    for budget in budgets:
        for seed in seeds:
            predictions, folds = [], []
            for arm in spec.arms:
                folder = run_path(spec, arm, budget, seed)
                row = dict(run_id=folder.name, group="C", arm=arm, budget=budget, seed=seed,
                           predictions=str(folder / "predictions_val.csv"))
                df, fingerprint = evaluator.load(row["predictions"])
                if len(df) != 21186 or df["subject_id"].nunique() != 8:
                    raise ValueError(f"Unexpected dev coverage: {folder.name}")
                result = evaluator.run_metrics(row, df, fingerprint)
                reported = json.loads((folder / "metrics.json").read_text())
                for key in ("macro_f1", "n1_f1", "n1_ap"):
                    if not np.isclose(result[key], reported[key], rtol=0, atol=1e-12):
                        raise ValueError(f"Metric did not replay: {folder.name} {key}")
                result.update(selected_epoch=reported["selected_epoch"],
                              n1_f1_adjusted=reported["n1_adjust"]["n1_f1_adjusted"],
                              macro_f1_adjusted=reported["n1_adjust"]["macro_f1_adjusted"],
                              alpha=reported["n1_adjust"]["alpha"])
                entries.append(row)
                frames[folder.name] = df
                metrics.append(result)
                predictions.append(df[IDS].sort_values("sample_id").reset_index(drop=True))
                oof = pd.read_csv(folder / "oof_train.csv")
                expected = {"label10": (11936, 5), "label25": (32310, 14)}[budget]
                if (len(oof), oof.subject_id.nunique()) != expected or oof["sample_id"].duplicated().any():
                    raise ValueError(f"Unexpected OOF coverage: {folder.name}")
                if oof.groupby("subject_id")["fold"].nunique().ne(1).any():
                    raise ValueError(f"OOF folds split an individual subject: {folder.name}")
                folds.append(oof[IDS + ["fold"]].sort_values("sample_id").reset_index(drop=True))
            if not predictions[0].equals(predictions[1]) or not folds[0].equals(folds[1]):
                raise ValueError(f"Different prediction identities or OOF folds: {budget} seed={seed}")
    table = pd.DataFrame(metrics)
    if pilot:
        table.to_csv(analysis / "pilot_label10_seed0.csv", index=False)
        print("Pilot verified: both arms, 21,186 aligned dev rows and matching OOF folds", flush=True)
        return
    registry = pd.DataFrame(entries)
    all_layers, last_layers = spec.arms
    paired, people = evaluator.paired(registry, frames, last_layers, all_layers)
    if len(paired) != 2 or len(people) != 16:
        raise ValueError("Missing budget or subject comparisons")
    table.to_csv(analysis / "metrics_by_run.csv", index=False)
    registry.to_csv(analysis / "run_registry.csv", index=False)
    keys = ["macro_f1", "n1_f1", "n1_ap", "n1_f1_adjusted", "macro_f1_adjusted"]
    table.groupby(["budget", "arm"])[keys].mean().reset_index().to_csv(analysis / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(analysis / "paired_last3_minus_all12.csv", index=False)
    pd.DataFrame(people).to_csv(analysis / "paired_by_subject.csv", index=False)
    measure_readout(spec)
    print(table.groupby(["budget", "arm"])[keys].mean().round(5).to_string(), flush=True)
    print(pd.DataFrame(paired).to_string(index=False), flush=True)
    (analysis / "verification.json").write_text(json.dumps(dict(
        runs=len(table), paired_budgets=len(paired), head_seeds=list(seeds), backbone_seeds=[0],
        dev_subjects=8, dev_rows_per_run=21186, identities_aligned=True,
        matched_oof_folds=True, metrics_recomputed=True, subject_bootstrap=evaluator.N_BOOT,
        self_conditioning=bool(spec.feedback_layer), test_used=False,
    ), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "full"))
    parser.add_argument("--backbone", choices=SPECS, default="plain")
    args = parser.parse_args()
    spec = SPECS[args.backbone]
    for name in ("logs", "probes", "analysis"):
        (spec.out / name).mkdir(parents=True, exist_ok=True)
    pilot = args.stage == "pilot"
    if not pilot and not (spec.out / "analysis/pilot_label10_seed0.csv").is_file():
        raise SystemExit("Run and verify the pilot first")
    prepare_features(spec)
    budgets = ("label10",) if pilot else BUDGETS
    seeds = (0,) if pilot else SEEDS
    with ThreadPoolExecutor(max_workers=2) as pool:
        for budget in budgets:
            for seed in seeds:
                children = [pool.submit(run_probe, spec, arm, budget, seed) for arm in spec.arms]
                for child in children:
                    child.result()
    collect(spec, load_evaluator(), budgets, seeds, pilot)


if __name__ == "__main__":
    main()
