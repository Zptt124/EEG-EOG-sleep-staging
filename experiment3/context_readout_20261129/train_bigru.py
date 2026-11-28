"""Train the shared CD BiGRU with 3, 11, or 21 physical epochs per center."""

import argparse
import csv
from hashlib import file_digest
from importlib import import_module
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
HEAD_SOURCE = ROOT / "scripts"
if not (HEAD_SOURCE / "13_context_head.py").exists():
    HEAD_SOURCE = ROOT / "tmp/github-week4-main/scripts"
sys.path.insert(0, str(HEAD_SOURCE))
ContextHead = import_module("13_context_head").ContextHead

STAGES = ("W", "N1", "N2", "N3", "REM")
C_SETTINGS = ((1, 0, False), (5, 0, False), (10, 0, False), (5, 0, True),
              (1, 1, False), (5, 1, False), (1, 2, False), (5, 2, False))


def sha256(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def configure(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    return torch.device(device)


def batches(n, generator, classes=None):
    if classes is None:
        yield from torch.randperm(n, generator=generator).split(128)
        return
    # Replacement is required to let rare stages appear as often as common ones.
    for first in range(0, n, 128):
        size = min(128, n - first)
        stages = torch.randint(len(classes), (size,), generator=generator)
        ids = torch.empty(size, dtype=torch.long)
        for j, pool in enumerate(classes):
            match = stages == j
            ids[match] = pool[torch.randint(len(pool), (int(match.sum()),), generator=generator)]
        yield ids


def fit(train_x, train_y, seed, balanced, device):
    if (train_x.ndim != 3 or train_x.shape[1] not in (3, 11, 21)
            or train_x.shape[2] != 384 or len(train_x) != len(train_y) or len(train_y) == 0
            or not np.isfinite(train_x).all() or not np.isin(train_y, np.arange(5)).all()):
        raise ValueError("Training requires finite 384-D windows of scored centers")
    centers = train_x[:, train_x.shape[1] // 2]
    mean = centers.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(centers.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    train_x = torch.from_numpy((train_x - mean) / std).to(device)
    y = torch.tensor(train_y, dtype=torch.long)
    classes = [pool for stage in range(5) if (pool := torch.where(y == stage)[0]).numel()] if balanced else None
    y = y.to(device)
    torch.manual_seed(seed)
    model = ContextHead(train_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4,
                                  fused=device.type == "cuda")
    generator = torch.Generator().manual_seed(seed)
    for _ in range(30):
        for ids in batches(len(y), generator, classes):
            ids = ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(train_x[ids]), y[ids]).backward()
            optimizer.step()
    return model.eval(), mean, std


def infer(model, mean, std, x, windows, device):
    logits = []
    with torch.inference_mode():
        for inds in np.array_split(windows, range(512, len(windows), 512)):
            batch = torch.from_numpy((x[inds] - mean) / std).to(device)
            logits.append(model(batch).cpu())
    logits = torch.cat(logits)
    return logits.argmax(dim=1).numpy().astype(np.uint8), logits.softmax(dim=1).numpy()


def validate(data):
    x, train_y, mask = (data[name] for name in ("x", "train_y", "unit_mask"))
    train, evaluation = data["train_windows"], data["eval_windows"]
    width = train.shape[1] if train.ndim == 2 else 0
    if (x.ndim != 2 or x.shape[1] != 384 or x.dtype != np.float32 or not np.isfinite(x).all()
            or width not in (3, 11, 21) or evaluation.ndim != 2 or evaluation.shape[1] != width
            or train_y.ndim != 1 or len(train) != len(train_y)
            or mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer)
            or np.any((mask < -1) | (mask >= len(train)))):
        raise ValueError("Invalid feature, training label, or unit mask axes")
    records, epochs = data["feature_record"], data["feature_epoch"]
    if len(records) != len(epochs) or len(epochs) != len(x):
        raise ValueError("Invalid feature key axes")
    if len(data["psg_record"]) != len(data["psg_length"]):
        raise ValueError("Invalid PSG length axes")
    lengths = dict(zip(data["psg_record"], data["psg_length"]))
    if len(lengths) != len(data["psg_record"]) or len(set(zip(records, epochs))) != len(x):
        raise ValueError("Duplicate recording boundaries or feature keys")
    for windows in (train, evaluation):
        if not np.issubdtype(windows.dtype, np.integer) or np.any((windows < 0) | (windows >= len(x))):
            raise ValueError("A context feature is missing")
        center = windows[:, width // 2]
        upper = np.asarray([lengths[r] for r in records[center]]) - 1
        if (np.any(epochs[center] < 0) or np.any(epochs[center] > upper)
                or np.any(records[windows] != records[center, None])
                or np.any(epochs[windows] != np.clip(epochs[center, None] + np.arange(width) - width // 2,
                                                    0, upper[:, None]))):
            raise ValueError("A context window differs from the same-PSG physical rule")
    if any(r[:-1] in data["subjects"] for r in set(records[train[:, width//2]])):
        raise ValueError("A development evaluation subject would be supervised")
    if any(len(np.unique(ids[ids >= 0])) != (ids >= 0).sum() for ids in mask):
        raise ValueError("Duplicate supervised center in a unit")
    ids, groups, subjects, truth = (data[k] for k in ("sample_id", "subject_index", "subjects", "truth"))
    center = evaluation[:, width // 2]
    expected_ids = [f"{r}_{e:04d}" for r, e in zip(records[center], epochs[center])]
    if (len(evaluation) != 41498 or len(subjects) != 16 or len(np.unique(subjects)) != 16
            or len(np.unique(ids)) != len(evaluation) or not np.array_equal(ids, expected_ids)
            or truth.shape != (len(evaluation),) or not np.isin(truth, np.arange(5)).all()
            or groups.shape != (len(evaluation),) or not np.issubdtype(groups.dtype, np.integer)
            or not np.array_equal(np.unique(groups), np.arange(16))
            or not np.array_equal(subjects[groups], [r[:-1] for r in records[center]])):
        raise ValueError("Invalid development evaluation metadata")
    if any(len(data[k]) != len(mask) for k in ("roster", "draw", "strategy", "effective_additions")):
        raise ValueError("Invalid unit metadata axes")
    keys = list(zip(data["roster"], data["draw"], data["strategy"]))
    if (len(set(keys)) != len(keys) or any(not str(s).isascii() or not str(s).replace("_", "").isalnum()
                                          for _, _, s in keys)
            or np.any((mask >= 0).sum(axis=1) != 1200 + data["effective_additions"])
            or not np.isin(train_y[mask[mask >= 0]], np.arange(5)).all()):
        raise ValueError("Invalid supervised keys or effective budget")
    return width // 2


def write_predictions(path, data, pred, probabilities):
    center = data["eval_windows"][:, data["eval_windows"].shape[1] // 2]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("sample_id", "subject_id", "record_id", "epoch_index", "y_true", "y_pred",
                         *(f"p_{stage}" for stage in STAGES)))
        writer.writerows((s, p, r, e, y, yh, *pr) for s, p, r, e, y, yh, pr in zip(
            data["sample_id"], data["subjects"][data["subject_index"]], data["feature_record"][center],
            data["feature_epoch"][center], data["truth"], pred, probabilities))


def run(args):
    device = configure(args.device)
    data_sha = sha256(args.input)
    with np.load(args.input, allow_pickle=False) as npz:
        data = {name: npz[name] for name in npz.files if name != "old_eval_windows"}
    radius = validate(data)
    names = ((1, 0, False),) if args.experiment == "D" else C_SETTINGS
    if args.variant:
        names = tuple(setting for name, setting in zip(
            ("K1", "K5", "K10", "K5B", "K1s1", "K5s1", "K1s2", "K5s2"), C_SETTINGS)
            if name in args.variant and setting in names)
        if len(names) != len(args.variant):
            raise ValueError("Unsupported experiment variant")
    if args.timing and not args.variant:
        names = ((5, 0, False),) if args.experiment == "C" else ((1, 0, False),)
    if any(k > radius for k, _, _ in names):
        raise ValueError("Input cache lacks the requested physical context")
    run_sha = None
    if not args.timing:
        lock = json.loads(args.joint_lock.read_text(encoding="utf-8"))
        run_sha = sha256(args.joint_lock)
        files = lock["sha256"]
        needed = {"analysis.py", "train_bigru.py", "13_context_head.py", "evaluation_index",
                  "physical_neighbor_rule", "cache_C", "cache_D", "selections_C", "selections_D",
                  "rosters", "seeds", "protocol"}
        if (set(lock["approved_by"]) != {"C", "D"} or not all(lock["approved_by"].values())
                or lock["tau_pp"] != 1 or not needed <= files.keys()
                or not all(len(files[k]) == 64 and all(c in "0123456789abcdef" for c in files[k]) for k in needed)
                or files["cache_" + args.experiment] != data_sha
                or files["train_bigru.py"] != sha256(Path(__file__))
                or files["13_context_head.py"] != sha256(HEAD_SOURCE / "13_context_head.py")
                or files["analysis.py"] != sha256(Path(__file__).with_name("analysis.py"))):
            raise ValueError("Joint lock approvals or code/input fingerprints do not match")
        args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    x, masks = data["x"], data["unit_mask"]
    for k, seed, balanced in names:
        span = slice(radius - k, radius + k + 1)
        for i in range(1 if args.timing else len(masks)):
            roster, draw, strategy = (data[key][i] for key in ("roster", "draw", "strategy"))
            tag = f"{int(roster):02d}_{int(draw):02d}_{strategy}_k{k}_seed{seed}{'_balanced' if balanced else ''}"
            path = args.output / f"{tag}.npz" if args.output else None
            if path and path.exists():
                with np.load(path, allow_pickle=False) as result:
                    if result["joint_sha256"].item() != run_sha or result["input_sha256"].item() != data_sha:
                        raise ValueError(f"Stale result: {path}")
                    if args.csv and not path.with_suffix(".csv").exists():
                        write_predictions(path.with_suffix(".csv"), data, result["pred"], result["probabilities"])
                continue
            supervised = masks[i][masks[i] >= 0]
            tick = perf_counter()
            model, mean, std = fit(x[data["train_windows"][supervised, span]], data["train_y"][supervised],
                                   seed, balanced, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if args.timing:
                print(json.dumps(dict(unit=tag, seconds=perf_counter() - tick,
                                      supervised=len(supervised), epochs=30, results_generated=False,
                                      torch=str(torch.__version__), device=str(device))))
                continue
            pred, pr = infer(model, mean, std, x, data["eval_windows"][:, span], device)
            counts = np.bincount(data["subject_index"] * 25 + data["truth"] * 5 + pred,
                                 minlength=400).reshape(16, 5, 5)
            model_path = path.with_suffix(".pt")
            torch.save(dict(model=model.state_dict(), mean=mean, std=std), model_path.with_suffix(".partial"))
            model_path.with_suffix(".partial").replace(model_path)
            with path.with_suffix(".tmp").open("wb") as file:
                np.savez_compressed(file, pred=pred, probabilities=pr, counts=counts, roster=roster, draw=draw,
                                    strategy=strategy, k=k, seed=seed, balanced=balanced,
                                    joint_sha256=run_sha, input_sha256=data_sha, model_sha256=sha256(model_path))
            path.with_suffix(".tmp").replace(path)
            if args.csv:
                write_predictions(path.with_suffix(".csv"), data, pred, pr)
            print(f"{tag} complete in {perf_counter() - tick:.2f}s", flush=True)
    print(json.dumps(dict(seconds=perf_counter() - start, torch=str(torch.__version__),
                          environment=platform.platform(), device=str(device))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="verified physical context NPZ")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment", choices=("C", "D"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--variant", action="append", choices=("K1", "K5", "K10", "K5B", "K1s1", "K5s1", "K1s2", "K5s2"))
    parser.add_argument("--joint-lock", type=Path)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()
    if not args.timing and (args.joint_lock is None or args.output is None):
        parser.error("Scored runs require --joint-lock and --output")
    run(args)
