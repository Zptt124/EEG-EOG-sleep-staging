"""Repeat the Sleep-EDF budget comparison with the team's existing temporal head."""

import argparse
from collections import defaultdict
from importlib import import_module
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
CONTEXT_SOURCE = ROOT / "scripts"
sys.path.insert(0, str(CONTEXT_SOURCE))
ContextHead = import_module("13_context_head").ContextHead
sys.path.insert(0, str(ROOT / "experiment3/week3_budget_replication_20261128"))
if __name__ == "__main__":
    # The earlier run is also named run.py.
    sys.path.remove(str(Path(__file__).resolve().parent))
from run import ARMS, BLOCK_SIZE, FEATURE_HASH, features, labels, macro, paired
from run import read_csv, require, sha256, summarize, write_csv, write_json

BATCH = 128
EPOCHS = 30
PRIMARY = "new_minus_deepen"


def load_inputs(source, bundle):
    prior = json.loads((source / "plan.json").read_text())
    for filename, digest in prior["locked_files"].items():
        require(sha256(source / filename) == digest, f"Changed frozen input: {filename}")
    metadata = bundle / "features_info.json"
    require(sha256(metadata) == FEATURE_HASH == prior["feature_info_sha256"], "Feature receipt mismatch")
    info = json.loads(metadata.read_text())
    require(sha256(bundle / "index.csv") == info["index_sha256"], "Changed feature index")
    rows = read_csv(bundle / "index.csv")
    arrays = []
    for name in ("eeg_std", "eog_std"):
        path = bundle / f"{name}.npy"
        require(sha256(path) == info["files"][name]["sha256"], f"Feature checksum mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(array.shape == (len(rows), 192) and array.dtype == np.float32, f"Invalid shape: {name}")
        arrays.append(array)
    expected = prior["dev8"] + prior["pretrain_eval8"]
    order = {person: i for i, person in enumerate(expected)}
    eval_ids = sorted((i for i, row in enumerate(rows) if row["subject_id"] in order),
                      key=lambda i: (order[rows[i]["subject_id"]], i))
    target = labels(rows, eval_ids)
    subjects = np.array([order[rows[i]["subject_id"]] for i in eval_ids])
    with np.load(source / "evaluation_index.npz", allow_pickle=False) as archive:
        require(np.array_equal(archive["subjects"], expected)
                and np.array_equal(archive["sample_id"], [rows[i]["sample_id"] for i in eval_ids])
                and np.array_equal(archive["truth"], target)
                and np.array_equal(archive["subject_index"], subjects), "Evaluation keys changed")

    keys = [(r["subject_id"], r["record_id"], int(r["epoch_index"])) for r in rows]
    positions = {key: i for i, key in enumerate(keys)}
    require(len(keys) == len(positions), "Duplicated subject/night/epoch")
    windows = np.empty((len(keys), 3), dtype=np.int64)
    for i, (person, night, epoch) in enumerate(keys):
        windows[i] = [positions.get((person, night, epoch - 1), i),
                      i, positions.get((person, night, epoch + 1), i)]

    grouped = defaultdict(list)
    for row in read_csv(source / "blocks.csv"):
        subject = row["subject_id"]
        require(subject not in order, "Evaluation subject used in training")
        key = int(row["roster"]), int(row["draw"]), row["arm"]
        first = int(row["first_epoch"])
        ids = [positions[subject, row["record_id"], e] for e in range(first, first + BLOCK_SIZE)]
        require(all(rows[i]["role"] == "pretrain_train" for i in ids), "Training key outside pool")
        grouped[key].extend(ids)
    return prior, rows, arrays, windows, grouped, (eval_ids, target, subjects)


def fit(train_x, train_y, eval_x, seed):
    centers = train_x[:, 1]
    mean = centers.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(centers.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    train_x = torch.from_numpy((train_x - mean) / std).cuda()
    train_y = torch.from_numpy(train_y).cuda()
    eval_x = torch.from_numpy((eval_x - mean) / std).cuda()
    torch.manual_seed(seed)
    model = ContextHead(train_x.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, fused=True)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        for ids in torch.randperm(len(train_y), generator=generator).cuda().split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(train_x[ids]), train_y[ids]).backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(batch).argmax(dim=1) for batch in eval_x.split(512)]).cpu().numpy()


def run(args):
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    prior, rows, arrays, windows, grouped, evaluation = load_inputs(args.source, args.bundle)
    eval_ids, truth, subjects = evaluation
    eval_x = features(arrays, windows[eval_ids].ravel()).reshape(len(eval_ids), 3, 384)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "primary": PRIMARY,
        "rule": "advance if mean >= 1 pp, lower > 0 and six low-overlap roster mean > 0; "
                "below practical target if upper < 1 pp; otherwise inconclusive",
        "fit_counts": {"primary": 450, "seed_sensitivity": 90},
        "head": "Team ContextHead, projection 128 / ReLU / dropout 0.2 / BiGRU 64x2 / center readout",
        "window": "previous, center, next in same person/night and adjacent epoch; center padding at gaps",
        "supervision": "center epochs in exact prior masks only; unlabelled neighbours supply features only",
        "standardization": "fit per arm on labelled center epochs only",
        "optimizer": "CUDA fused AdamW, lr 0.001, weight decay 0.0001, batch 128, 30 fixed epochs",
        "analysis": "same 10000 paired bootstrap and low-overlap subset as the linear comparison",
        "inputs_sha256": {name: sha256(args.source / name)
                          for name in ("plan.json", "blocks.csv", "evaluation_index.npz")},
        "feature_receipt_sha256": FEATURE_HASH,
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in
                          (Path(__file__).resolve(), CONTEXT_SOURCE / "13_context_head.py",
                           ROOT / "experiment3/week3_budget_replication_20261128/run.py")},
        "torch": str(torch.__version__),
    }
    protocol_file = out / "protocol.json"
    if protocol_file.exists():
        require(json.loads(protocol_file.read_text()) == protocol, "Previous run used a different protocol")
    else:
        write_json(protocol_file, protocol)
    print(f"Protocol locked | {len(truth)} evaluation epochs | 540 fits", flush=True)

    totals = np.zeros((3, 30, 5, 3, 16, 25), dtype=np.int64)
    scores = []
    durations = []
    for roster in range(1, 31):
        tick = time.perf_counter()
        final = out / f"roster_{roster:02d}.npz"
        if final.exists():
            with np.load(final, allow_pickle=False) as saved:
                totals[:, roster - 1] = saved["counts"]
                scores.extend(json.loads(str(saved["scores"])))
            continue
        base = grouped[roster, 0, "base"]
        require(len(base) == len(set(base)) == 1200, "Invalid common mask")
        roster_scores = []
        predictions = {}
        for draw in range(1, 6):
            extras = [grouped[roster, draw, arm] for arm in ARMS]
            require(all(len(x) == 600 for x in extras)
                    and len(set(base + sum(extras, []))) == 3000, "Invalid paired budgets")
            for arm_index, (arm, extra) in enumerate(zip(ARMS, extras)):
                ids = sorted(base + extra)
                train_x = features(arrays, windows[ids].ravel()).reshape(len(ids), 3, 384)
                train_y = labels(rows, ids)
                for seed in range(3 if roster <= 3 else 1):
                    pred = fit(train_x, train_y, eval_x, seed)
                    conf = np.bincount(subjects * 25 + truth * 5 + pred, minlength=400).reshape(16, 25)
                    totals[seed, roster - 1, draw - 1, arm_index] = conf
                    predictions[f"{draw}_{arm}_{seed}"] = pred.astype(np.uint8)
                    roster_scores.append(dict(roster=roster, draw=draw, arm=arm, seed=seed,
                                              macro_f1=macro(conf.sum(axis=0)).item()))
        with final.with_suffix(".tmp").open("wb") as staged:
            np.savez_compressed(staged, counts=totals[:, roster - 1],
                                scores=json.dumps(roster_scores), **predictions)
        final.with_suffix(".tmp").replace(final)
        scores.extend(roster_scores)
        duration = time.perf_counter() - tick
        durations.extend([duration / len(roster_scores)] * len(roster_scores))
        eta = (540 - len(scores)) * np.mean(durations) / 60
        print(f"Roster {roster:02d}/30 | {len(scores)}/540 fits | {duration:.1f}s | "
              f"fit ETA {eta:.1f} min", flush=True)
        if eta > args.maximum_hours * 60:
            print("Measured remaining runtime exceeds the overnight limit; paused.", flush=True)
            return
    write_csv(out / "scores.csv", scores)
    print("All fits complete | starting paired bootstrap", flush=True)
    with TemporaryDirectory() as temporary:
        analysis = Path(temporary)
        summarize(totals, prior, analysis)
        result = json.loads((analysis / "summary.json").read_text())
    comparison = result["combined"][PRIMARY]
    lower, upper = comparison["interval_95_pp"]
    if upper < 1:
        decision = "below_practical_target"
    elif lower > 0 and comparison["mean_pp"] >= 1 and comparison["low_overlap_mean_pp"] > 0:
        decision = "advance_to_external_bigru"
    else:
        decision = "inconclusive"
    seeded = paired(macro(totals[:, :3].sum(axis=-2)))
    result.update(
        decision=decision, primary=PRIMARY,
        arm_f1=dict(zip(ARMS, macro(totals[0].sum(axis=-2)).mean(axis=(0, 1)).tolist())),
        seed_check={"first_three_rosters_new_minus_deepen_pp": seeded[..., 2].mean(axis=(1, 2)).tolist()},
        elapsed_seconds=time.perf_counter() - start,
    )
    write_json(out / "summary.json", result)
    print(f"Primary {comparison['mean_pp']:+.2f} pp [{lower:+.2f}, {upper:+.2f}] | {decision}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "output/budget_replication_20261128")
    parser.add_argument("--bundle", type=Path, default=ROOT / "experiment3/deliveries/d-w6-shared-cache-v01-20261126/full")
    parser.add_argument("--output", type=Path, default=ROOT / "output/budget_bigru_20261128")
    parser.add_argument("--maximum-hours", type=float, default=6)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(parser.parse_args())
