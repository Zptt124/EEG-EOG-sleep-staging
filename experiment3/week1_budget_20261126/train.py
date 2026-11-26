"""Train the fixed-budget probe and write separate pool36 and dev8 predictions."""

import argparse
import json
import platform
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from protocol import ARMS, BLOCK_SIZE, KEY, SOURCES, read_csv, require, sha256, write_csv

BATCH = 128
PROBS = ("p_W", "p_N1", "p_N2", "p_N3", "p_REM")


def load_plan(plan, data):
    manifest = json.loads((plan / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["locks"].items():
        require(sha256(plan / name) == digest, f"Changed roster file: {name}")
    for key, digest in manifest["sources"].items():
        require(sha256(data / SOURCES[key]) == digest, f"Changed data manifest: {key}")
    return manifest


def load_bundle(bundle, expected_info_hash, plan, data):
    metadata = bundle / "features_info.json"
    require(sha256(metadata) == expected_info_hash.lower(), "Feature receipt differs from the approved digest")
    info = json.loads(metadata.read_text(encoding="utf-8"))
    require(info["step"] == 30000 and info["weights"] == "ema_model", "Expected 30k EMA features")
    require(info["pool_sec"] == 1 and info["dim_per_modality"] == 192, "Expected 192 dimensions per modality and 1s pooling")
    index = bundle / "index.csv"
    require(sha256(index) == info["index_sha256"], "Feature index hash mismatch")
    rows = read_csv(index)
    roles = {r["subject_id"]: r["role"] for r in read_csv(plan / "subject_roles.csv")}
    partitions = {"eligible_pool": "train", "held_out": "train", "dev": "val"}
    require(all(roles.get(r["subject_id"]) in partitions for r in rows), "Unexpected or sealed subject in cache")
    require(all(partitions[roles[r["subject_id"]]] == r["split"] for r in rows), "Feature split mismatch")
    reference = read_csv(data / SOURCES["epoch"]) + read_csv(data / SOURCES["dev"])
    actual_keys = {tuple(r[k] for k in KEY) for r in rows}
    require(len(actual_keys) == len(rows) == info["n_samples"] == len(reference), "Feature index row count or uniqueness mismatch")
    require(actual_keys == {tuple(r[k] for k in KEY) for r in reference}, "Feature index keys differ from frozen train/dev epochs")
    positions = {(r["subject_id"], r["record_id"], int(r["epoch_index"])): i for i, r in enumerate(rows)}
    require(len(positions) == len(rows), "Duplicated subject/record/epoch")
    arrays = []
    for name in ("eeg_std", "eog_std"):
        path = bundle / f"{name}.npy"
        require(sha256(path) == info["files"][name]["sha256"], f"Hash mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(array.shape == (len(rows), 192) and array.dtype == np.float32, f"Invalid array: {name}")
        arrays.append(array)
    return rows, positions, arrays


def masks(plan, roster, positions, rows):
    arms = {arm: [] for arm in ARMS}
    nights = {arm: Counter() for arm in ARMS}
    thirds = {arm: Counter() for arm in ARMS}
    for block in read_csv(plan / roster / "blocks.csv"):
        arm, subject, record = (block[k] for k in ("arm", "subject_id", "record_id"))
        first, last = (int(block[k]) for k in ("first_epoch", "last_epoch"))
        require(last - first + 1 == BLOCK_SIZE, "Wrong block length")
        arms[arm].extend(positions[subject, record, epoch] for epoch in range(first, last + 1))
        nights[arm][subject, record] += 1
        thirds[arm][block["night_third"]] += 1
    require(len(set(sum(arms.values(), []))) == 2400, "Overlapping or missing blocks")
    require(all(len(nights[arm]) == 5 and set(nights[arm].values()) == {count}
                for arm, count in zip(ARMS, (4, 2, 2))), "Incorrect block allocation")
    require(nights["base"].keys() == nights["deepen"].keys(), "Deepen must use the same five nights")
    require(thirds["add_people"] == thirds["deepen"], "Unmatched within-night positions")
    subjects = {arm: {s for s, _ in nights[arm]} for arm in ARMS}
    require(len(subjects["base"] | subjects["add_people"]) == 10, "Expected ten unique subjects")
    fitted = subjects["base"] | subjects["add_people"]
    roles = {r["subject_id"]: r["role"] for r in read_csv(plan / "subject_roles.csv")}
    pool = {s for s, role in roles.items() if role == "eligible_pool"}
    require(fitted <= pool, "Fitting subject outside eligible pool")
    cohorts = {cohort: {r["subject_id"] for r in read_csv(plan / roster / "evaluation_subjects.csv")
                        if r["cohort"] == cohort} for cohort in ("pool36", "dev8")}
    require(cohorts["pool36"] == pool - fitted and len(cohorts["pool36"]) == 36, "Invalid pool36")
    require(cohorts["dev8"] == {s for s, role in roles.items() if role == "dev"} and
            len(cohorts["dev8"]) == 8, "Invalid dev8")
    fitting_ids = {arm: sorted(arms["base"] + (arms[arm] if arm != "base" else [])) for arm in ARMS}
    scoring_ids = {cohort: [i for i, row in enumerate(rows) if row["subject_id"] in members]
                   for cohort, members in cohorts.items()}
    return fitting_ids, scoring_ids


def fit(x, y, seed, epochs):
    # Probe 34 delegates to probe 27. Keep its float64 fit statistics,
    # initialization, per-epoch shuffle and AdamW update order.
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    x = torch.from_numpy((x - mean) / std)
    y = torch.as_tensor(y, dtype=torch.long)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Linear(x.shape[1], 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    curve = []
    updates = 0
    for epoch in range(1, epochs + 1):
        total = 0.0
        for ids in torch.randperm(len(y)).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x[ids]), y[ids])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
            updates += 1
        curve.append(dict(epoch=epoch, loss=total / len(y), cumulative_updates=updates))
    require(updates == epochs * ((len(y) + BATCH - 1) // BATCH), "Unexpected update count")
    require(all(np.isfinite(row["loss"]) for row in curve), "Non-finite loss")
    return model, mean, std, curve


def features(arrays, ids):
    x = np.concatenate([array[ids] for array in arrays], axis=1)
    require(np.isfinite(x).all(), "Non-finite features")
    return x


def labels(rows, ids):
    y = np.array([int(rows[i]["y_true"]) for i in ids])
    require(np.isin(y, range(5)).all(), "Invalid stage label")
    return y


def score(out, cohort, ids, rows, arrays, model, mean, std):
    x = torch.from_numpy((features(arrays, ids) - mean) / std)
    with torch.inference_mode():
        probabilities = torch.cat([model(batch).softmax(1) for batch in x.split(2048)]).numpy()
    y = labels(rows, ids)
    predictions = probabilities.argmax(axis=1)
    table = []
    for j, i in enumerate(ids):
        table.append({**{k: rows[i][k] for k in KEY}, "y_true": int(y[j]), "y_pred": int(predictions[j]),
                      **dict(zip(PROBS, map(float, probabilities[j])))})
    write_csv(out / f"predictions_{cohort}.csv", table)
    return float(f1_score(y, predictions, labels=list(range(5)), average="macro", zero_division=0))


def main(args):
    manifest = load_plan(args.plan, args.data)
    rows, positions, arrays = load_bundle(args.bundle, args.feature_info_sha256, args.plan, args.data)
    fitting, scoring = masks(args.plan, args.roster, positions, rows)
    if not args.fit_only:
        review_dir = args.output / "01" / "loss_review"
        review_config = json.loads((review_dir / "base" / "config.json").read_text(encoding="utf-8"))
        require(review_config["feature_info_sha256"] == args.feature_info_sha256.lower() and
                review_config["plan_sha256"] == sha256(args.plan / "manifest.json") and
                review_config["code"] == code_hashes() and
                review_config["epochs"] == manifest["epochs"] and
                review_config["seed"] == manifest["head_seed"], "The loss-only run used a different configuration")
        require(sha256(review_dir / "base" / "loss.csv") == args.reviewed_loss_sha256.lower(),
                "Use the digest of the reviewed 01 loss curve")
        dev_truth = {r["sample_id"]: int(r["y_true"]) for r in read_csv(args.data / SOURCES["dev"])}
        require(all(int(rows[i]["y_true"]) == dev_truth[rows[i]["sample_id"]] for i in scoring["dev8"]),
                "Dev truth differs from the canonical reference")
    torch.set_num_threads(4)
    target = args.output / args.roster / ("loss_review" if args.fit_only else "paired")
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), f"Run already exists: {target}")
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as staging:
        pending = Path(staging) / "run"
        pending.mkdir()
        run(args, manifest, rows, arrays, fitting, scoring, pending)
        pending.rename(target)


def run(args, manifest, rows, arrays, fitting, scoring, target):
    metrics = {}
    for arm in ("base",) if args.fit_only else ARMS:
        ids = fitting[arm]
        model, mean, std, curve = fit(features(arrays, ids), labels(rows, ids), manifest["head_seed"], manifest["epochs"])
        out = target / arm
        out.mkdir()
        write_csv(out / "loss.csv", curve)
        np.savez(out / "normalization.npz", mean=mean, std=std)
        torch.save(model.state_dict(), out / "head.pt")
        config = dict(roster=args.roster, arm=arm, feature_info_sha256=args.feature_info_sha256.lower(),
                      plan_sha256=sha256(args.plan / "manifest.json"), code=code_hashes(),
                      seed=manifest["head_seed"], epochs=manifest["epochs"], train_samples=len(ids),
                      updates=curve[-1]["cumulative_updates"], batch=BATCH, drop_last=False,
                      torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version())
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        if not args.fit_only:
            metrics[arm] = {cohort: score(out, cohort, eval_ids, rows, arrays, model, mean, std)
                            for cohort, eval_ids in scoring.items()}
    if args.fit_only:
        tail = curve[-5:]
        slope = float(np.polyfit([row["epoch"] for row in tail], [row["loss"] for row in tail], 1)[0])
        summary = dict(last_five_loss_slope_per_epoch=slope, first_loss=curve[0]["loss"], final_loss=curve[-1]["loss"],
                       loss_sha256=sha256(target / "base" / "loss.csv"))
    else:
        summary = {c: dict(macro_f1={arm: metrics[arm][c] for arm in ARMS},
                           add_people_gain_pp=100 * (metrics["add_people"][c] - metrics["base"][c]),
                           deepen_gain_pp=100 * (metrics["deepen"][c] - metrics["base"][c]),
                           add_people_minus_deepen_pp=100 * (metrics["add_people"][c] - metrics["deepen"][c]))
                   for c in scoring}
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def code_hashes():
    return {name: sha256(Path(__file__).with_name(name)) for name in ("train.py", "protocol.py", "plan.py")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("plan", "data", "bundle", "output"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--roster", choices=[f"{i:02d}" for i in range(1, 11)], required=True)
    parser.add_argument("--feature-info-sha256", required=True, help="Independently approved D feature receipt digest")
    parser.add_argument("--fit-only", action="store_true", help="Run roster 01 base without scoring")
    parser.add_argument("--reviewed-loss-sha256", help="Roster 01 loss curve digest after training-only review")
    args = parser.parse_args()
    if args.fit_only != (args.reviewed_loss_sha256 is None) or (args.fit_only and args.roster != "01"):
        parser.error("Use --fit-only for roster 01, or provide --reviewed-loss-sha256 for paired scoring")
    main(args)
