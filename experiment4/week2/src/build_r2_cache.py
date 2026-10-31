"""build_r2_cache.py - Week-2 Experiment 4 (D): build the R2 representation cache.

R2 = A's budget-matched EEG-only and EOG-only SmallCNN encoders (128-d each), concatenated (256-d).
This is NOT A's dual-channel encoder.

For every (budget, downstream seed) pair:
  1. Load W2_A_EEG_<budget>_sseed2024_seed<k>_v01 and W2_A_EOG_..._seed<k>_v01 encoders (eval mode, no_grad).
  2. Verify encoder_weights.pt / normalization.json SHA256 against A's encoder_index.csv and SHA256SUMS.txt.
  3. Forward every sample in the R1 sample_index.csv (54 train + 8 val subjects; test never touched),
     using each encoder's own normalization (fitted by A on that budget's labeled subjects only).
  4. Check: A's classifier layer from best.ckpt applied to our validation features must reproduce
     A's predictions_val.csv (max prob diff <= PROB_TOL; argmax differences only on near-ties).
  5. Save eeg_feat.npy / eog_feat.npy [N, 128] float32 in the R1 row order, plus manifest and checks.

Labels are not used to build features. The y_true column copied into sample_index.csv is hidden later by
label_guard.py in every training entry point, exactly as for R1.

Usage (Ubuntu):
  python -u build_r2_cache.py --a-runs <A_release_a-cnn-w2-v01-20261031> --a-repo <team_repo>
      --data-dir <A_processed_npz>
      --r1-cache <D_local_cache>/R1_eegdm_cond14_pooled_v01 --out-root <D_local_cache>
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROB_TOL = 5e-3     # GPU float differences vs A's own forward pass (different machine/GPU)
# An argmax change is tolerated only on a near-tie: A's top-2 margin <= 2 * that sample's own max prob
# difference (+1e-6 for A's 6-decimal rounding). This is the necessary condition for a flip caused by
# float noise; any other argmax change means the features differ from A's forward pass.
BUDGETS = ["label10", "label25"]
SEEDS = [0, 1, 2, 3, 4]
MODS = {"eeg": ("EEG", 0), "eog": ("EOG", 1)}  # key -> (A run tag, NPZ signal channel)
PROB = ["p_W", "p_N1", "p_N2", "p_N3", "p_REM"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def check_encoder(run_dir, enc_index):
    # Verify both exported files against A's index and against the SHA256SUMS.txt shipped with them
    enc = run_dir / "encoder_export"
    row = enc_index.get(run_dir.name)
    if row is None:
        fail(f"{run_dir.name} not found in encoder_index.csv")
    got = {"encoder_weights.pt": sha256(enc / "encoder_weights.pt"),
           "normalization.json": sha256(enc / "normalization.json")}
    if got["encoder_weights.pt"] != row["weights_sha256"] or got["normalization.json"] != row["normalization_sha256"]:
        fail(f"{run_dir.name}: SHA256 differs from encoder_index.csv")
    sums = {}
    for line in (enc / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = h
    for name, h in got.items():
        if sums.get(name) != h:
            fail(f"{run_dir.name}: {name} SHA256 differs from SHA256SUMS.txt")
    return got, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-runs", type=Path, required=True, help="folder with unzipped W2_A_* run folders")
    ap.add_argument("--a-repo", type=Path, required=True, help="team repo checkout with scripts/ and reports/week2/")
    ap.add_argument("--data-dir", type=Path, required=True, help="A's verified NPZ folder (uV)")
    ap.add_argument("--r1-cache", type=Path, required=True, help="R1 cache, provides sample_index.csv row order")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--version", default="v01")
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    sys.path.insert(0, str(args.a_repo / "scripts"))
    from cnn_model import SmallCNN, load_encoder  # A's single model implementation

    out_dirs = {(b, s): args.out_root / f"R2_cnn_{b}_seed{s}_{args.version}" for b in BUDGETS for s in SEEDS}
    for d in out_dirs.values():
        if (d / "cache_manifest.json").exists():
            fail(f"{d} already has a cache, use a new --version")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t0 = time.time()

    # ---------- row order and NPZ records (same samples as R1) ----------
    rows = read_csv(args.r1_cache / "sample_index.csv")
    row_of = {r["sample_id"]: i for i, r in enumerate(rows)}
    if len(row_of) != len(rows):
        fail("duplicated sample_id in R1 sample_index.csv")
    records = sorted({r["record_id"] for r in rows})
    n_per_record = {}
    for r in rows:
        n_per_record[r["record_id"]] = n_per_record.get(r["record_id"], 0) + 1
    if any(r["split"] not in ("train", "val") for r in rows):
        fail("R1 sample_index.csv contains a split other than train/val")
    print(f"OK: {len(rows)} samples from {len(records)} records (train + val only), device {device}", flush=True)

    # ---------- load and verify all 20 encoders ----------
    enc_index = {r["source_run_id"]: r for r in read_csv(args.a_repo / "reports" / "week2" / "encoder_index.csv")}
    models, norms, enc_info = {}, {}, {}
    for b in BUDGETS:
        for s in SEEDS:
            for m, (tag, ch) in MODS.items():
                run_dir = args.a_runs / f"W2_A_{tag}_{b}_sseed2024_seed{s}_v01"
                shas, row = check_encoder(run_dir, enc_index)
                if row["nominal_budget"] != b or int(row["downstream_seed"]) != s or row["modalities"] != m:
                    fail(f"{run_dir.name}: encoder_index metadata does not match budget/seed/modality")
                model, norm = load_encoder(run_dir / "encoder_export")
                # Note: A's normalization.json names the single channel "EEG Fpz-Cz" even for EOG encoders
                # (build_norm_json uses CHANNEL_NAMES[0] when in_channels == 1), so the modality is taken
                # from the run's config.yaml instead; the reproduction check below confirms the channel.
                run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
                if model.in_channels != 1 or run_cfg.get("modality") != m or int(run_cfg.get("in_channels", 0)) != 1:
                    fail(f"{run_dir.name}: expected a single-channel {tag} encoder (config.yaml says "
                         f"modality={run_cfg.get('modality')}, in_channels={run_cfg.get('in_channels')})")
                if model.training:
                    fail(f"{run_dir.name}: encoder is not in eval mode")
                models[(b, s, m)] = model.to(device)
                norms[(b, s, m)] = (float(norm["channels"][0]["mean"]), float(norm["channels"][0]["std"]))
                enc_info[(b, s, m)] = {"run_id": run_dir.name, **shas,
                                       "norm_mean": norms[(b, s, m)][0], "norm_std": norms[(b, s, m)][1],
                                       "norm_fit_on": norm.get("fit_on", {}).get("labeled_subjects"),
                                       "norm_channel_name_in_json": norm["channels"][0]["name"]}
    print(f"OK: {len(models)} encoders loaded, SHA256 verified, eval mode", flush=True)

    # ---------- forward every record once through all encoders ----------
    feats = {k: np.full((len(rows), 128), np.nan, dtype=np.float32) for k in models}
    n_done = 0
    with torch.no_grad():
        for i_rec, rec in enumerate(records, start=1):
            with np.load(args.data_dir / f"{rec}.npz", allow_pickle=False) as z:
                sig = z["signal"].astype(np.float32)
                ep = z["epoch_index"]
                lab = z["labels"]
            if sig.ndim != 3 or sig.shape[1:] != (2, 3000):
                fail(f"{rec}: unexpected signal shape {sig.shape}")
            ids = [f"{rec}_{int(e):04d}" for e in ep]
            # keep only samples that exist in the R1 index; every R1 sample of this record must be found
            keep = np.array([sid in row_of for sid in ids])
            idx = np.array([row_of[sid] for sid, k in zip(ids, keep) if k])
            if len(idx) != n_per_record[rec]:
                fail(f"{rec}: {len(idx)} samples matched, R1 index has {n_per_record[rec]}")
            # labels in the NPZ must equal R1 y_true (alignment check only; labels are not used)
            if any(int(lab[j]) != int(rows[i]["y_true"]) for j, i in zip(np.where(keep)[0], idx)):
                fail(f"{rec}: NPZ labels differ from R1 sample_index.csv")
            for key, model in models.items():
                ch = MODS[key[2]][1]
                mean, std = norms[key]
                x = torch.from_numpy((sig[keep][:, ch:ch + 1, :] - mean) / std)
                out = [model.encoder(x[b:b + args.batch_size].to(device)).cpu().numpy()
                       for b in range(0, len(x), args.batch_size)]
                feats[key][idx] = np.concatenate(out)
            n_done += len(idx)
            if i_rec % 20 == 0 or i_rec == len(records):
                print(f"  {i_rec}/{len(records)} records, {n_done} samples, {time.time() - t0:.0f}s", flush=True)
    for key, f in feats.items():
        if not np.isfinite(f).all():
            fail(f"{key}: some samples were not filled or contain NaN/Inf")
    print("OK: all samples filled for all encoders, no NaN/Inf", flush=True)

    # ---------- reproduce A's validation predictions with A's own classifier layer ----------
    val_rows = [i for i, r in enumerate(rows) if r["split"] == "val"]
    checks = []
    for key in models:
        b, s, m = key
        run_dir = args.a_runs / enc_info[key]["run_id"]
        ckpt = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=True)
        clf = torch.nn.Linear(128, 5)
        clf.load_state_dict({"weight": ckpt["state_dict"]["classifier.weight"],
                             "bias": ckpt["state_dict"]["classifier.bias"]})
        clf.eval()
        a_pred = {r["sample_id"]: r for r in read_csv(run_dir / "predictions_val.csv")}
        if len(a_pred) != len(val_rows):
            fail(f"{run_dir.name}: A's predictions_val.csv has {len(a_pred)} rows, expected {len(val_rows)}")
        with torch.no_grad():
            p_ours = clf(torch.from_numpy(feats[key][val_rows])).softmax(1).numpy()
        p_a = np.array([[float(a_pred[rows[i]["sample_id"]][c]) for c in PROB] for i in val_rows])
        per_sample_diff = np.abs(p_ours - p_a).max(1)
        max_diff = float(per_sample_diff.max())
        diff_arg = np.where(p_ours.argmax(1) != p_a.argmax(1))[0]
        top2 = np.sort(p_a, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        bad = [i for i in diff_arg if margin[i] > 2 * per_sample_diff[i] + 1e-6]
        checks.append({"budget": b, "seed": s, "modality": m, "run_id": run_dir.name,
                       "max_prob_diff": round(max_diff, 6), "mean_prob_diff": round(float(per_sample_diff.mean()), 8),
                       "n_argmax_diff": len(diff_arg),
                       "max_margin_of_flipped": round(float(margin[diff_arg].max()), 6) if len(diff_arg) else 0.0,
                       "n_argmax_diff_not_explained_by_float_noise": len(bad),
                       "passed": bool(max_diff <= PROB_TOL and not bad)})
    for c in checks:
        print(f"  check {c['run_id']}: max diff {c['max_prob_diff']:.2e}, argmax changes {c['n_argmax_diff']} "
              f"(unexplained {c['n_argmax_diff_not_explained_by_float_noise']}) -> {'PASS' if c['passed'] else 'FAIL'}")
    if not all(c["passed"] for c in checks):
        fail("some encoders do not reproduce A's validation predictions (see lines above)")
    print(f"OK: A's validation predictions reproduced for all {len(checks)} encoders "
          f"(max prob diff {max(c['max_prob_diff'] for c in checks):.2e}, "
          f"total argmax changes {sum(c['n_argmax_diff'] for c in checks)} of {len(val_rows) * len(checks)})", flush=True)

    # ---------- save one cache folder per (budget, seed) ----------
    r1_manifest = json.loads((args.r1_cache / "cache_manifest.json").read_text(encoding="utf-8"))
    for (b, s), d in out_dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.r1_cache / "sample_index.csv", d / "sample_index.csv")
        for m in MODS:
            np.save(d / f"{m}_feat.npy", feats[(b, s, m)])
        with (d / "extraction_check.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(checks[0].keys()))
            w.writeheader()
            w.writerows([c for c in checks if c["budget"] == b and c["seed"] == s])
        manifest = {
            "representation": f"R2_A_cnn_{b}_seed{s}: A's EEG-only + EOG-only SmallCNN encoders (128 + 128), eval mode",
            "budget": b, "downstream_seed": s, "n_samples": len(rows),
            "feature_files": {"eeg": "eeg_feat.npy", "eog": "eog_feat.npy"},
            "dims": {"eeg": 128, "eog": 128}, "dtype": "float32",
            "row_order": "identical to R1 sample_index.csv",
            "r1_sample_index_sha256": r1_manifest["files_sha256"]["sample_index.csv"],
            "encoders": {m: enc_info[(b, s, m)] for m in MODS},
            "encoder_index": str(args.a_repo / "reports" / "week2" / "encoder_index.csv"),
            "files_sha256": {n: sha256(d / n) for n in ("sample_index.csv", "eeg_feat.npy", "eog_feat.npy")},
            "device": device, "torch": str(torch.__version__),
            "minutes_total_build": round((time.time() - t0) / 60, 1),
        }
        if manifest["files_sha256"]["sample_index.csv"] != manifest["r1_sample_index_sha256"]:
            fail(f"{d}: copied sample_index.csv differs from R1")
        (d / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved {d}", flush=True)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
