"""calibrate_threshold_r2.py - Week-2 Experiment 4 (D): internal calibration of p for the R2 (CNN) representation.

Same procedure and the same pre-registered rule as calibrate_threshold.py (R1), but every fold uses
SmallCNN encoders retrained WITHOUT the held-out subject (run_fold_cnns.py), as protocol v2 rule 4 requires:
  for each labeled subject h of the budget:
    - fold EEG-only and EOG-only encoders (trained on labeled minus h, seed 0) give 128-d features
      for the fold-train subjects, for h, and for the validation set (eval mode, no_grad);
    - features are standardized with stats fitted on the fold-train subjects only;
    - three linear teachers (dual 256-d, EEG-only, EOG-only) are trained exactly like the formal teachers
      (ssl_common.train_head, seed 0, validation macro-F1 early stopping);
    - h is pseudo-labeled; h's true labels are used only to measure pseudo-label quality (h is labeled).
  The pooled held-out predictions simulate the unlabeled pool. Candidates p in {0.2, 0.4, 0.6};
  chosen p = largest p with D1 macro pseudo-label precision >= 0.85, otherwise the smallest p.
Unlabeled subjects are never loaded. Validation scores are never used to choose p.

Usage (Ubuntu):
  python -u calibrate_threshold_r2.py --budget label25 --roles .../manifests/v2/labeled_unlabeled_subjects.csv
      --fold-runs <D_local_r2_fold_cnns> --a-repo <team_repo>
      --data-dir <A_processed_npz> --r1-cache <D_local_cache>/R1_eegdm_cond14_pooled_v01
      --output .../calibration/R2_label25_v01
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from calibrate_threshold import CANDIDATE_P, PRECISION_TARGET, TEACHER_SEED, accept_per_class, summarize
from label_guard import LabelLeakError, assert_hidden, hide_labels, load_roles
from ssl_common import STAGES, fit_standardizer, read_csv, sha256, train_head

CH = {"eeg": 0, "eog": 1}


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def load_fold_encoder(run_dir, modality, SmallCNN):
    # Full SmallCNN from A's best.ckpt; only .encoder() is used. Modality is checked from config.yaml.
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if cfg.get("modality") != modality or int(cfg.get("in_channels", 0)) != 1:
        fail(f"{run_dir.name}: config.yaml modality {cfg.get('modality')} != {modality}")
    model = SmallCNN(in_channels=1)
    ckpt = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = json.loads((run_dir / "normalization.json").read_text(encoding="utf-8"))
    mean, std = float(norm["channels"][0]["mean"]), float(norm["channels"][0]["std"])
    fit_on = sorted(norm["fit_on"]["labeled_subjects"])
    return model, mean, std, fit_on, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", choices=["label10", "label25"], required=True)
    ap.add_argument("--roles", type=Path, required=True)
    ap.add_argument("--fold-runs", type=Path, required=True)
    ap.add_argument("--a-repo", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--r1-cache", type=Path, required=True, help="provides sample_index.csv (row order, labels)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if (args.output / "calibration_decision.json").exists():
        fail(f"{args.output} already has a calibration, choose a new folder")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.a_repo / "scripts"))
    from cnn_model import SmallCNN

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()

    # ---------- label hiding; only labeled + validation samples are ever loaded ----------
    rows = read_csv(args.r1_cache / "sample_index.csv")
    labeled, unlabeled = load_roles(args.roles, args.budget)
    sseed = read_csv(args.roles)[0]["sampling_seed"]
    subject = np.array([r["subject_id"] for r in rows])
    split = np.array([r["split"] for r in rows])
    y_all = np.array([int(r["y_true"]) for r in rows], dtype=np.int64)
    try:
        y_train, is_lab, is_unl = hide_labels(y_all, subject, split, labeled, unlabeled)
        assert_hidden(y_train, is_unl, is_lab)
    except LabelLeakError as e:
        fail(str(e))
    del y_all
    keep = np.where(is_lab | (split == "val"))[0]
    subj_k, split_k, y_k = subject[keep], split[keep], y_train[keep]
    sid_k = np.array([rows[i]["sample_id"] for i in keep])
    pos = {sid: j for j, sid in enumerate(sid_k)}
    lab_k, val_k = split_k == "train", split_k == "val"
    y_val = y_k[val_k]

    # ---------- raw signals of labeled + validation records (unlabeled records are not read) ----------
    records = sorted({rows[i]["record_id"] for i in keep})
    sig = np.empty((len(keep), 2, 3000), dtype=np.float32)
    filled = np.zeros(len(keep), dtype=bool)
    for rec in records:
        with np.load(args.data_dir / f"{rec}.npz", allow_pickle=False) as z:
            s, ep = z["signal"].astype(np.float32), z["epoch_index"]
        for j, e in enumerate(ep):
            p = pos.get(f"{rec}_{int(e):04d}")
            if p is not None:
                sig[p] = s[j]
                filled[p] = True
    if not filled.all():
        fail(f"{int((~filled).sum())} labeled/validation samples not found in NPZ files")
    print(f"OK: {args.budget} sseed={sseed}: {len(labeled)} labeled subjects ({int(lab_k.sum())} samples) + "
          f"{int(val_k.sum())} validation samples loaded from {len(records)} records; unlabeled records not read", flush=True)

    folds = [r for r in read_csv(args.fold_runs / "fold_index.csv") if r["budget"] == args.budget]
    held_list = sorted({r["heldout_subject"] for r in folds})
    if held_list != sorted(labeled):
        fail(f"fold_index.csv held-out subjects {held_list} != labeled subjects {sorted(labeled)}")

    pool_parts, fold_rows = [], []
    for fold, held in enumerate(held_list):
        t0 = time.time()
        feats, enc_rows = {}, {}
        for m in ("eeg", "eog"):
            fr = [r for r in folds if r["heldout_subject"] == held and r["modality"] == m]
            if len(fr) != 1:
                fail(f"fold for held-out {held} modality {m} not found exactly once")
            run_dir = args.fold_runs / fr[0]["run_id"]
            model, mean, std, fit_on, cfg = load_fold_encoder(run_dir, m, SmallCNN)
            # the fold encoder must never have seen the held-out subject
            if held in fit_on or held in cfg["labeled_subjects"] or sorted(fit_on) != sorted(set(labeled) - {held}):
                fail(f"{run_dir.name}: fold encoder was trained on the wrong subjects")
            model.to(device)
            out = []
            with torch.no_grad():
                for b in range(0, len(sig), 1024):
                    x = torch.from_numpy((sig[b:b + 1024, CH[m]:CH[m] + 1, :] - mean) / std).to(device)
                    out.append(model.encoder(x).cpu().numpy())
            feats[m] = np.concatenate(out).astype(np.float32)
            if not np.isfinite(feats[m]).all():
                fail(f"{run_dir.name}: NaN/Inf in features")
            enc_rows[m] = {"run_id": run_dir.name, "ckpt_sha256": sha256(run_dir / "best.ckpt"),
                           "val_macro_f1_A_head": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["macro_f1"]}

        tr = lab_k & (subj_k != held)
        ho = lab_k & (subj_k == held)
        stats = {m: fit_standardizer(feats[m], tr) for m in feats}
        z = {m: (feats[m] - stats[m][0]) / stats[m][1] for m in feats}
        inputs = {"dual": np.concatenate([z["eeg"], z["eog"]], axis=1), "eeg": z["eeg"], "eog": z["eog"]}
        probs_ho = {}
        frow = {"fold": fold, "heldout_subject": held, "n_train_subjects": len(labeled) - 1,
                "n_train_samples": int(tr.sum()), "n_heldout_samples": int(ho.sum()),
                "eeg_encoder": enc_rows["eeg"]["run_id"], "eog_encoder": enc_rows["eog"]["run_id"]}
        for name, xi in inputs.items():
            out = train_head(np.ascontiguousarray(xi[tr]), y_k[tr], np.ascontiguousarray(xi[val_k]), y_val, TEACHER_SEED)
            head = torch.nn.Linear(xi.shape[1], 5)
            head.load_state_dict(out["state"])
            head.eval()
            with torch.inference_mode():
                probs_ho[name] = head(torch.from_numpy(np.ascontiguousarray(xi[ho]))).softmax(1).numpy()
            frow[f"{name}_best_epoch"] = out["best_epoch"]
            frow[f"{name}_val_macro_f1"] = round(out["best_val_macro_f1"], 4)
        pred = probs_ho["dual"].argmax(1)
        part = {"sample_id": sid_k[ho], "subject_id": subj_k[ho], "y_true": y_k[ho], "pred": pred,
                "conf": probs_ho["dual"].max(1), "eeg_pred": probs_ho["eeg"].argmax(1), "eog_pred": probs_ho["eog"].argmax(1)}
        frow["heldout_teacher_accuracy"] = round(float((pred == part["y_true"]).mean()), 4)
        fold_rows.append(frow)
        pool_parts.append(part)
        print(f"fold {fold + 1}/{len(held_list)} held={held} n={int(ho.sum())} teacher acc on held-out "
              f"{frow['heldout_teacher_accuracy']:.4f} (dual val {frow['dual_val_macro_f1']:.4f}, {time.time() - t0:.0f}s)", flush=True)

    pool = {k: np.concatenate([p[k] for p in pool_parts]) for k in pool_parts[0]}
    agree = (pool["eeg_pred"] == pool["pred"]) & (pool["eog_pred"] == pool["pred"])
    with (args.output / "heldout_pool_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "subject_id", "y_true", "teacher_pred", "teacher_conf", "eeg_pred", "eog_pred", "modalities_agree"])
        for i in range(len(pool["pred"])):
            w.writerow([pool["sample_id"][i], pool["subject_id"][i], int(pool["y_true"][i]), int(pool["pred"][i]),
                        round(float(pool["conf"][i]), 6), int(pool["eeg_pred"][i]), int(pool["eog_pred"][i]), int(agree[i])])
    with (args.output / "fold_teachers.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        w.writerows(fold_rows)

    cal_rows = [summarize("accept_all", 1.0, np.ones(len(pool["pred"]), dtype=bool), pool["pred"], pool["y_true"],
                          {s: None for s in STAGES})]
    for p in CANDIDATE_P:
        acc1, thr = accept_per_class(pool["pred"], pool["conf"], p)
        cal_rows.append(summarize("D1", p, acc1, pool["pred"], pool["y_true"], thr))
        cal_rows.append(summarize("D2", p, acc1 & agree, pool["pred"], pool["y_true"], thr))
    with (args.output / "threshold_calibration.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)

    print("\n=== pseudo-label quality on the simulated pool (labeled subjects, LOSO, R2 fold encoders) ===")
    print(f"{'method':10s} {'p':>4s} {'cover':>6s} {'macroP':>7s} " + " ".join(f"{'P_' + s:>6s}" for s in STAGES)
          + " " + " ".join(f"{'n_' + s:>6s}" for s in STAGES) + "  N1_captured")
    for r in cal_rows:
        print(f"{r['method']:10s} {r['p']:4.1f} {r['coverage']:6.3f} {r['macro_precision']:7.4f} "
              + " ".join(f"{(r['prec_' + s] if r['prec_' + s] != '' else float('nan')):6.3f}" for s in STAGES) + " "
              + " ".join(f"{r['n_acc_' + s]:6d}" for s in STAGES) + f"  {r['true_N1_captured']:.3f}")

    ok = [r["p"] for r in cal_rows if r["method"] == "D1" and r["macro_precision"] >= PRECISION_TARGET]
    chosen = max(ok) if ok else min(CANDIDATE_P)
    decision = {"representation": "R2", "budget": args.budget, "sampling_seed": sseed, "chosen_p": chosen,
                "rule": f"largest p in {CANDIDATE_P} with D1 macro pseudo-label precision >= {PRECISION_TARGET}; otherwise smallest p",
                "rule_satisfied": bool(ok), "teacher_seed": TEACHER_SEED, "fold_encoder_seed": 0,
                "teacher_early_stopping": "validation macro-F1, patience 10 (same as formal teachers); validation scores NOT used to choose p",
                "fold_encoders": "A's train_cnn_w2.py --subjects (labeled minus held-out), see fold_teachers.csv",
                "n_folds": len(held_list), "n_pool": int(len(pool["pred"])), "roles_sha256": sha256(args.roles),
                "fold_index_sha256": sha256(args.fold_runs / "fold_index.csv"),
                "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
                "minutes": round((time.time() - t_start) / 60, 1)}
    (args.output / "calibration_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"\nDECISION: chosen p = {chosen} (rule satisfied: {bool(ok)})")
    print(f"Saved: {args.output}  ({decision['minutes']} min)")


if __name__ == "__main__":
    main()
