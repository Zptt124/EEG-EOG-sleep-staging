"""Audit and complete physical-epoch features for the frozen condition A exercise."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment3/window_budget_c_20261128"))
from run_screening import (BUNDLE, CACHE, D, DEST, MANIFEST, OLD, OUT, hash_file,
                           read_json, read_rows, require, save_json, save_npz, verify_selection)
from download import BASES, get_file
from urllib.request import Request, urlopen

RESULT = ROOT / "output/context_readout_20261129"
STAGES = ("add_people", "random")
RECIPE = (DEST / "bigru_ab/protocol.json")


def physical(record, epoch, radius, length):
    if not 0 <= epoch < length:
        raise ValueError(f"Center outside physical PSG: {record} {epoch}/{length}")
    return [(record, min(max(k, 0), length - 1))
            for k in range(epoch - radius, epoch + radius + 1)]


def read_header(raw):
    if raw.exists():
        with raw.open("rb") as file:
            return file.read(256)
    cached = raw.parent.parent / "headers" / raw.name
    if cached.exists():
        return cached.read_bytes()
    for base in BASES[::-1] * 2:
        try:
            request = Request(base + "sleep-cassette/" + raw.name, headers={"Range": "bytes=0-255"})
            with urlopen(request, timeout=30) as response:
                header = response.read(256)
            if len(header) == 256:
                cached.write_bytes(header)
                return header
        except OSError as error:
            print(f"Header retry {raw.name}: {error}", flush=True)
    raise RuntimeError(f"Unable to read EDF header: {raw.name}")


def inputs():
    anchor, chosen, _ = verify_selection()
    units = {k: v for k, v in chosen.items() if k[2] == "A" and k[3] in STAGES}
    require(len(units) == 260, "Expected 260 locked condition A units")
    require(hash_file(CACHE) == read_json(DEST / "cache_lock.json")["cache_sha256"],
            "Historical screening cache changed")
    feature_info = read_json(BUNDLE / "features_info.json")
    require(hash_file(BUNDLE / "index.csv") == feature_info["index_sha256"], "Shared index changed")
    rows = read_rows(BUNDLE / "index.csv")
    shared = {(r["record_id"], int(r["epoch_index"])): i for i, r in enumerate(rows)}
    require(len(shared) == len(rows), "Duplicate physical epoch in shared index")
    require(hash_file(OLD / "evaluation_index.npz") == read_json(OUT / "protocol.json")["evaluation_sha256"],
            "Evaluation index changed")
    with np.load(OLD / "evaluation_index.npz", allow_pickle=False) as data:
        evaluation = {k: data[k] for k in ("sample_id", "truth", "subject_index", "subjects")}
    by_id = {r["sample_id"]: r for r in rows}
    eval_keys = [(by_id[s]["record_id"], int(by_id[s]["epoch_index"])) for s in evaluation["sample_id"]]
    require(len(eval_keys) == len(set(eval_keys)) == 41498
            and len(evaluation["subjects"]) == 16, "Development evaluation axes changed")
    for i, key in enumerate(eval_keys):
        row = rows[shared[key]]
        require(row["subject_id"] == evaluation["subjects"][evaluation["subject_index"][i]]
                and int(row["y_true"]) == evaluation["truth"][i], f"Evaluation alignment: {key}")
    with np.load(CACHE, allow_pickle=False) as data:
        cache_keys = {(r, int(e)): i for i, (r, e) in enumerate(zip(data["record"], data["epoch"]))}
        center_x, center_y = data["x"], data["y"]
        require(np.array_equal(data["eval_id"], evaluation["sample_id"])
                and np.array_equal(data["eval_y"], evaluation["truth"]), "Screening evaluation changed")
    require(len(cache_keys) == len(center_x) == len(center_y), "Screening center axes differ")
    return anchor, units, feature_info, rows, shared, evaluation, eval_keys, cache_keys, center_x, center_y


def prepare(output, download_only):
    import mne
    import torch
    from extract import CHANNELS, extract_epochs, w4

    output.mkdir(parents=True, exist_ok=True)
    (anchor, units, info, rows, shared, evaluation, eval_keys,
     cached, center_x, center_y) = inputs()
    attempts = {k: sorted(set(anchor[k[0]]) | set(values)) for k, values in units.items()}
    require(all(len(keys) == 1800 for keys in attempts.values()), "Anchor and additions overlap")
    train_keys = sorted({key for keys in attempts.values() for key in keys if center_y[cached[key]] >= 0})
    require(all(center_y[cached[key]] >= 0 for keys in anchor.values() for key in keys), "Invalid anchor label")
    records_train = {r for r, _ in train_keys}
    records_eval = {r for r, _ in eval_keys}
    require(records_train.isdisjoint(records_eval), "Train and development evaluation cross")
    source = OUT / "source"
    hashes = {name: digest for digest, name in
              (line.split() for line in (source / "SHA256SUMS.txt").read_text().splitlines())}
    entries = {r["record_id"]: r for r in read_rows(MANIFEST)}
    require(all(entries[r]["subject_id"] not in evaluation["subjects"] for r in records_train),
            "A development evaluation subject would be supervised")
    known = {r["record_id"]: r for r in read_rows(OUT / "recording_metadata.csv")}
    psg = {r: source / "raw" / entries[r]["psg_file"] for r in records_train | records_eval}

    def duration(record):
        header = read_header(psg[record])
        seconds = int(header[236:244]) * float(header[244:252])
        require(seconds >= 30, f"Invalid EDF duration: {record}")
        if record in known:
            require(seconds == float(known[record]["duration_sec"]), f"Changed PSG duration: {record}")
        return record, seconds

    with ThreadPoolExecutor(max_workers=8) as pool:
        durations = dict(pool.map(duration, sorted(psg)))
    lengths = {r: int(seconds // 30) for r, seconds in durations.items()}
    windows = {key: physical(*key, 10, lengths[key[0]]) for key in train_keys + eval_keys}
    needed = {key for window in windows.values() for key in window}
    missing = defaultdict(list)
    for record, epoch in sorted(needed - shared.keys() - cached.keys()):
        missing[record].append(epoch)
    print(f"{len(train_keys)} training centers, {len(eval_keys)} eval centers, "
          f"{sum(map(len, missing.values()))} uncached epochs in {len(missing)} PSGs", flush=True)

    def fetch(record):
        path = psg[record]
        name = "sleep-cassette/" + path.name
        get_file(name, path, hashes[name])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, sorted(missing)))
    if download_only:
        return

    arrays = []
    for name in ("eeg_std", "eog_std"):
        path = BUNDLE / f"{name}.npy"
        require(hash_file(path) == info["files"][name]["sha256"], f"Shared {name} changed")
        arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
    checkpoint = ROOT / "output/budget_external_ying/ckpt_step30000.pt"
    require(hash_file(checkpoint) == info["checkpoint_sha256"], "Frozen W4 checkpoint changed")
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    supplements = output / "supplements"
    supplements.mkdir(exist_ok=True)
    bindings = {
        "checkpoint_sha256": hash_file(checkpoint),
        "physionet_checksums_sha256": hash_file(source / "SHA256SUMS.txt"),
        "feature_info_sha256": hash_file(BUNDLE / "features_info.json"),
        "screening_cache_sha256": hash_file(CACHE),
        "selection_sha256": hash_file(D / "lock.json"),
        "evaluation_sha256": hash_file(OLD / "evaluation_index.npz"),
        "extractor_sha256": {p.name: hash_file(p) for p in (
            Path(__file__), Path(sys.modules["extract"].__file__),
            Path(w4.__file__), Path(w4.C.__file__))},
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
    }
    supplements_lock = supplements / "protocol.json"
    if supplements_lock.exists():
        require(read_json(supplements_lock) == bindings, "Supplement provenance changed")
    else:
        save_json(supplements_lock, bindings)
    model = None
    supplement_x, provenance = {}, []
    for record, epochs in missing.items():
        path = supplements / f"{record}.npz"
        extra_digest = path.with_suffix(".sha256")
        if not path.exists() or not extra_digest.exists():
            if model is None:
                model, _, _ = w4.load_model(checkpoint, "ema_model")
                model = model.cuda()
                for layer in model.layers:
                    layer.layer.kernel.forward = lru_cache(maxsize=2)(layer.layer.kernel.forward)
            reference = sorted(e for r, e in shared if r == record)
            require(bool(reference), f"No overlap verification epoch: {record}")
            check = [reference[i] for i in sorted({0, len(reference) // 2, len(reference) - 1})]
            raw = mne.io.read_raw_edf(psg[record], include=list(CHANNELS), preload=False, verbose="ERROR")
            try:
                require(raw.info["sfreq"] == 100 and set(raw.ch_names) == set(CHANNELS)
                        and raw.n_times / 100 == durations[record], f"PSG grid mismatch: {record}")
                order = sorted(set(epochs + check))
                values = dict(zip(order, extract_epochs(model, raw, order)))
            finally:
                raw.close()
            reference_x = np.stack([np.concatenate([a[shared[record, e]] for a in arrays]) for e in check])
            replay_x = np.stack([values[e] for e in check])
            error = float(np.max(np.abs(replay_x - reference_x)))
            require(np.allclose(replay_x, reference_x, atol=2e-4, rtol=5e-3),
                    f"Replayed frozen features exceed tolerance: {record}, max abs {error}")
            save_npz(path, epoch=epochs, x=np.stack([values[e] for e in epochs]),
                     overlap=len(check), max_abs=error)
            extra_digest.write_text(hash_file(path) + "\n", encoding="ascii")
        digest = hash_file(path)
        require(extra_digest.read_text().strip() == digest, f"Supplement hash mismatch: {record}")
        with np.load(path, allow_pickle=False) as data:
            require(np.array_equal(data["epoch"], epochs) and data["x"].shape == (len(epochs), 384)
                    and np.isfinite(data["x"]).all(), f"Supplement axes differ: {record}")
            supplement_x.update({(record, e): v for e, v in zip(epochs, data["x"])})
            provenance.append(dict(record_id=record, epochs=len(epochs), overlap=int(data["overlap"]),
                                   max_abs=float(data["max_abs"]), sha256=digest,
                                   psg_sha256=hashes["sleep-cassette/" + psg[record].name]))
        print(f"Verified supplemental feature rows {len(provenance)}/{len(missing)} {record}", flush=True)

    keys = sorted(needed)
    pos = {key: i for i, key in enumerate(keys)}
    feature_x = np.empty((len(keys), 384), np.float32)
    feature_sources = []
    for i, key in enumerate(keys):
        if key in cached:
            feature_x[i] = center_x[cached[key]]
            feature_sources.append("screening")
            if key in shared:
                require(np.array_equal(feature_x[i], np.concatenate([a[shared[key]] for a in arrays])),
                        f"Shared and screening features differ: {key}")
        elif key in shared:
            feature_x[i] = np.concatenate([a[shared[key]] for a in arrays])
            feature_sources.append("shared")
        else:
            feature_x[i] = supplement_x[key]
            feature_sources.append("supplement")
    require(np.isfinite(feature_x).all(), "Nonfinite features")

    def table(centers):
        return np.asarray([[pos[k] for k in windows[key]] for key in centers], dtype=np.int32)

    train_pos = {key: i for i, key in enumerate(train_keys)}
    unit_ids = sorted(units)
    masks = np.full((len(units), 1800), -1, np.int32)
    additions = []
    for i, key in enumerate(unit_ids):
        selected = [train_pos[k] for k in attempts[key] if k in train_pos]
        masks[i, :len(selected)] = selected
        additions.append(len(selected) - 1200)

    # Cache membership is irrelevant for the new mapping. Old K1 training used physical
    # neighbors, while its evaluation duplicated the center when the shared index had a gap.
    old_eval = np.asarray([[pos.get(k, pos[key]) if k in shared else pos[key]
                            for k in ((key[0], key[1] - 1), key, (key[0], key[1] + 1))]
                           for key in eval_keys], dtype=np.int32)
    eval_windows, train_windows = table(eval_keys), table(train_keys)
    old_train = train_windows[:, 9:12]
    audit_file = output / "k1_neighbors.csv"
    with audit_file.with_suffix(".tmp").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "record_id", "center_epoch", "old_prev", "old_next", "new_prev",
                         "new_next", "old_prev_source", "old_next_source", "new_prev_source",
                         "new_next_source", "prev_boundary", "next_boundary",
                         "center_scorable", "feature_pair_differs"))
        for split, centers, old, new in (("train", train_keys, old_train, train_windows[:, 9:12]),
                                         ("eval", eval_keys, old_eval, eval_windows[:, 9:12])):
            for key, o, n in zip(centers, old, new):
                prev, nxt = [int(j) for j in o[[0, 2]]]
                new_prev, new_nxt = [int(j) for j in n[[0, 2]]]
                writer.writerow((split, *key, keys[prev][1], keys[nxt][1], keys[new_prev][1],
                                 keys[new_nxt][1], *(feature_sources[j] for j in (prev, nxt, new_prev, new_nxt)),
                                 int(key[1] == 0), int(key[1] == lengths[key[0]] - 1), 1,
                                 int(not np.array_equal(feature_x[o], feature_x[n]))))
    audit_file.with_suffix(".tmp").replace(audit_file)
    unit_file = output / "k1_baseline_units.csv"
    with unit_file.with_suffix(".tmp").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("roster", "draw", "strategy", "effective_additions", "decision", "reason"))
        for (roster, draw, _, strategy), effective in zip(unit_ids, additions):
            writer.writerow((roster, draw, strategy, effective, "refit",
                             "archived screening stores predictions and counts, without model weights"))
    unit_file.with_suffix(".tmp").replace(unit_file)
    cache = output / "physical_context.npz"
    save_npz(cache, x=feature_x, feature_record=[k[0] for k in keys],
             feature_epoch=[k[1] for k in keys], feature_source=feature_sources,
             psg_record=list(lengths), psg_length=list(lengths.values()),
             train_windows=train_windows, train_y=[center_y[cached[k]] for k in train_keys],
             eval_windows=eval_windows, old_eval_windows=old_eval, unit_mask=masks,
             roster=[k[0] for k in unit_ids], draw=[k[1] for k in unit_ids],
             strategy=[k[3] for k in unit_ids], effective_additions=additions, **evaluation)
    save_json(output / "physical_context.json", dict(bindings, units=len(units),
              training_centers=len(train_keys), evaluation_centers=len(eval_keys), psg_epochs=lengths,
              eval_k1_centers_changed=int(np.any(old_eval != eval_windows[:, 9:12], axis=1).sum()),
              boundary_duplicates={split: {str(radius): int(np.sum(
                  [max(0, radius - e) + max(0, e + radius - lengths[r] + 1) for r, e in centers]))
                  for radius in (1, 5, 10)} for split, centers in (("train_unique", train_keys), ("eval", eval_keys))},
              supplements=provenance, files={p.name: hash_file(p) for p in (cache, audit_file, unit_file)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    prepare(args.output, args.download_only)
