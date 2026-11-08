#run_ssl.py —— 第二周 实验④ 半监督训练入口：D0 / D1 / D2
#D0：只用 5/14 名有标签训练者的真标签训练线性学生头
#D1：双模态教师（= 同种子 D0）给无标签者打伪标签，按类别接收置信度最高的前 p（p 由 calibrate_threshold.py 锁定）
#D2：在 D1 同一候选集合上，再要求 EEG-only、EOG-only 教师预测同一类别才接收
#学生：混合批次，每步 128 真标签 + 384 伪标签；每轮步数与 D0 相同；早停与 D0 相同
#标准化：所有方法都只用有标签训练者拟合；无标签者的 y 通过 label_guard 置为 -1，全程不读取
#每个 run 存六件套 + pseudo_label_trace.csv + pseudo_label_stats.csv（D1/D2）；教师单独存放并登记 teacher_index.csv
#用法（Ubuntu）：python -u run_ssl.py --methods D1 D2 --repr R1 --budget label25 --seeds 0 1 2 3 4
#              --roles 名单.csv --cache 缓存 --runs-root 输出根目录 --calibration calibration_decision.json

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from calibrate_threshold import accept_per_class
from label_guard import LabelLeakError, assert_hidden, hide_labels, load_roles
from ssl_common import (EVALUATE_PATH, STAGES, fit_standardizer, load_official_evaluate, predict_probs,
                        read_csv, save_predictions, sha256, train_head, train_student)

#第一周日志 19：25% 主名单(2024) EEG+EOG 拼接，用于核对 D0（= 双模态教师）是否复现
WEEK1_REFERENCE = {("R1", "label25", "2024", s): v for s, v in enumerate([0.7449, 0.7433, 0.7431, 0.7508, 0.7458])}
SRC = Path(__file__).resolve().parent
MU, LAMBDA = 3, 1.0
DUAL, EEG, EOG = slice(0, 5120), slice(0, 2560), slice(2560, 5120)
REGISTRY_FIELDS = ["run_id", "method", "repr", "budget", "sampling_seed", "seed", "macro_f1", "f1_W", "f1_N1",
                   "f1_N2", "f1_N3", "f1_REM", "n1_precision", "n1_recall", "accuracy", "best_epoch",
                   "n_epochs_run", "n_updates", "n_labeled_samples", "n_pseudo_labels_used", "train_seconds"]
DESCRIPTIONS = {
    "D0": "supervised student head on true labels of labeled subjects only",
    "D1": "self-training: dual-modal teacher pseudo-labels, per-predicted-class top-p confidence",
    "D2": "cross-modal self-training: D1 candidates AND EEG-only and EOG-only teachers predict the same class",
}


def append_csv(path, row, fields):
    #追加一行；已有文件的表头必须一致，否则报错
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        if header != fields:
            print(f"ERROR: header of {path} does not match, expected {fields}")
            sys.exit(1)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in fields})


def write_readme(run_dir, cfg, status, last_step, extra=""):
    text = (f"# {run_dir.name}\n\n"
            f"- status: {status}\n- last successful step: {last_step}\n"
            f"- method: {cfg['method']} ({cfg['method_description']})\n"
            f"- representation: {cfg['representation']}\n"
            f"- budget: {cfg['budget']} ({cfg['n_labeled_subjects']} labeled subjects, "
            f"{cfg['n_labeled_samples']} labeled samples; unlabeled subjects used: {cfg['n_unlabeled_subjects_used']}, "
            f"pseudo labels used: {cfg['n_pseudo_labels_used']})\n"
            f"- sampling_seed: {cfg['sampling_seed']}, downstream seed: {cfg['seed']}\n"
            f"- evaluation: validation set only (8 subjects), official evaluate.py {cfg['evaluate_version']}\n"
            f"{extra}")
    (run_dir / "README.md").write_text(text, encoding="utf-8")


def save_history(path, history):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", choices=["D0", "D1", "D2"], required=True)
    ap.add_argument("--repr", choices=["R1", "R2"], default="R1")
    ap.add_argument("--budget", choices=["label10", "label25"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--roles", type=Path, required=True)
    ap.add_argument("--cache", required=True,
                    help="cache folder; for R2 use a pattern with {seed}, e.g. <D_local_cache>/R2_cnn_label25_seed{seed}_v01")
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, help="calibration_decision.json (required for D1/D2)")
    ap.add_argument("--version", default="v01")
    args = ap.parse_args()
    methods = list(dict.fromkeys(args.methods))
    need_teachers = any(m in ("D1", "D2") for m in methods)

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    ev = load_official_evaluate()
    print(f"Official evaluate.py loaded, version {ev.EVALUATE_VERSION}", flush=True)

    #---------- 名单、校准结果、标签隐藏 ----------
    #R2 features differ per downstream seed (A's encoder of that seed), so the cache path may contain {seed}
    per_seed_cache = "{seed}" in args.cache
    if args.repr == "R2" and not per_seed_cache:
        print("ERROR: R2 needs a per-seed cache pattern containing {seed}")
        sys.exit(1)
    cache_for = lambda s: Path(args.cache.format(seed=s))
    first_cache = cache_for(args.seeds[0])
    manifest = json.loads((first_cache / "cache_manifest.json").read_text(encoding="utf-8"))
    rows = read_csv(first_cache / "sample_index.csv")
    if len(rows) != manifest["n_samples"]:
        print("ERROR: sample_index.csv length does not match cache_manifest.json")
        sys.exit(1)
    role_rows = read_csv(args.roles)
    sseeds = {r["sampling_seed"] for r in role_rows}
    if len(sseeds) != 1:
        print(f"ERROR: roles file has several sampling_seed values {sseeds}")
        sys.exit(1)
    sseed = sseeds.pop()
    roles_sha = sha256(args.roles)
    labeled, unlabeled = load_roles(args.roles, args.budget)

    calib, p_locked = None, None
    if need_teachers:
        if args.calibration is None:
            print("ERROR: --calibration is required for D1/D2")
            sys.exit(1)
        calib = json.loads(args.calibration.read_text(encoding="utf-8"))
        #校准必须来自同一预算、同一名单文件
        if calib.get("representation", "R1") != args.repr:
            print(f"ERROR: calibration is for representation {calib.get('representation', 'R1')}, not {args.repr}")
            sys.exit(1)
        if calib["budget"] != args.budget or calib["sampling_seed"] != sseed or calib["roles_sha256"] != roles_sha:
            print("ERROR: calibration file does not match this budget / sampling seed / roles file")
            sys.exit(1)
        p_locked = float(calib["chosen_p"])
        print(f"OK: calibration locked p = {p_locked} ({args.calibration})", flush=True)

    subject = np.array([r["subject_id"] for r in rows])
    split = np.array([r["split"] for r in rows])
    y_all = np.array([int(r["y_true"]) for r in rows], dtype=np.int64)
    try:
        y_train, is_lab, is_unl = hide_labels(y_all, subject, split, labeled, unlabeled)
        assert_hidden(y_train, is_unl, is_lab)
    except LabelLeakError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    is_val = split == "val"
    del y_all  #之后只用 y_train（无标签者为 -1）和验证集标签
    print(f"OK: {args.budget} sseed={sseed}: {len(labeled)} labeled subjects / {int(is_lab.sum())} samples; "
          f"{len(unlabeled)} unlabeled subjects / {int(is_unl.sum())} samples hidden (-1); "
          f"{int(is_val.sum())} validation samples", flush=True)

    lab_idx = np.where(is_lab)[0]
    val_idx = np.where(is_val)[0]
    unl_idx = np.where(is_unl)[0]
    y_tr = y_train[lab_idx]
    if (y_tr < 0).any():
        print("ERROR: a labeled training sample has a hidden label")
        sys.exit(1)
    val_rows = [rows[i] for i in val_idx]
    y_va = np.array([int(r["y_true"]) for r in val_rows], dtype=np.int64)  # validation labels: evaluation / early stopping only
    unl_ids = [rows[i]["sample_id"] for i in unl_idx]

    def load_features(cache_dir, seed):
        # Read one cache, standardize each modality with stats fitted on labeled training samples only,
        # and return the [N, d_eeg + d_eog] array (EEG columns first). Same procedure for R1 and R2.
        t0 = time.time()
        man = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
        files = man.get("feature_files", {"eeg": "eeg_pooled.npy", "eog": "eog_pooled.npy"})  # R1 manifests have no field
        dims = man.get("dims", {"eeg": 2560, "eog": 2560})
        if sha256(cache_dir / "sample_index.csv") != man["files_sha256"]["sample_index.csv"] or \
                read_csv(cache_dir / "sample_index.csv") != rows:
            print(f"ERROR: {cache_dir} sample_index.csv differs from the first cache")
            sys.exit(1)
        if args.repr == "R2" and (man.get("budget") != args.budget or int(man.get("downstream_seed", -1)) != seed):
            print(f"ERROR: {cache_dir} is for budget {man.get('budget')} seed {man.get('downstream_seed')}, "
                  f"expected {args.budget} seed {seed}")
            sys.exit(1)
        raw = {m: np.load(cache_dir / files[m], mmap_mode="r") for m in ("eeg", "eog")}
        for m, arr in raw.items():
            if arr.shape != (len(rows), dims[m]):
                print(f"ERROR: {files[m]} shape {arr.shape}, expected ({len(rows)}, {dims[m]})")
                sys.exit(1)
        st = {m: fit_standardizer(raw[m], is_lab) for m in raw}
        X = np.empty((len(rows), dims["eeg"] + dims["eog"]), dtype=np.float32)
        starts = {"eeg": 0, "eog": dims["eeg"]}
        for m in ("eeg", "eog"):
            for s0 in range(0, len(rows), 20000):
                X[s0:s0 + 20000, starts[m]:starts[m] + dims[m]] = (np.asarray(raw[m][s0:s0 + 20000], dtype=np.float32)
                                                                    - st[m][0]) / st[m][1]
        del raw
        print(f"Features ready in {time.time() - t0:.1f}s from {cache_dir.name}: all {X.shape}, "
              f"labeled {len(lab_idx)}, val {len(val_idx)}, unlabeled pool {len(unl_idx)}", flush=True)
        return X, st, man, dims

    loaded = None

    code_sha = {p.name: sha256(p) for p in sorted(SRC.glob("*.py"))}
    args.runs_root.mkdir(parents=True, exist_ok=True)
    env = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__),
           "device": "cpu", "torch_threads": 4, "deterministic": True}

    for seed in args.seeds:
        run_ids = {m: f"W2_D_{m}_{args.repr}_{args.budget}_sseed{sseed}_seed{seed}_{args.version}" for m in methods}
        for m, rid in run_ids.items():
            if (args.runs_root / rid).exists():
                print(f"ERROR: {args.runs_root / rid} already exists, use a new --version")
                sys.exit(1)
        print(f"\n===== seed {seed} =====", flush=True)
        cache_dir = cache_for(seed)
        if loaded is None or (per_seed_cache and loaded[0] != cache_dir):
            loaded = (cache_dir, *load_features(cache_dir, seed))
        _, X_all, stats, manifest, dims = loaded
        D = dims["eeg"] + dims["eog"]
        DUAL, EEG, EOG = slice(0, D), slice(0, dims["eeg"]), slice(dims["eeg"], D)
        x_tr, x_va = X_all[lab_idx], X_all[val_idx]

        #---------- 双模态教师（= D0 学生）----------
        t1 = time.time()
        dual = train_head(x_tr, y_tr, x_va, y_va, seed)
        dual_seconds = round(time.time() - t1, 1)
        ref = WEEK1_REFERENCE.get((args.repr, args.budget, sseed, seed))
        if ref is not None:
            same = abs(round(dual["best_val_macro_f1"], 4) - ref) < 1e-4
            print(f"CHECK: dual teacher / D0 val macro-F1 {dual['best_val_macro_f1']:.4f} vs week-1 {ref} -> "
                  f"{'MATCH' if same else 'DIFFERENT'}", flush=True)

        results = {}  #method -> (训练输出, 用到的伪标签数, 额外信息)
        if "D0" in methods:
            results["D0"] = (dual, 0, dual_seconds, None)

        if need_teachers:
            #---------- 单模态教师 + 打伪标签 ----------
            teacher_dir = args.runs_root / "teachers" / f"W2_D_teachers_{args.repr}_{args.budget}_sseed{sseed}_seed{seed}_{args.version}"
            if teacher_dir.exists():
                print(f"ERROR: {teacher_dir} already exists, use a new --version")
                sys.exit(1)
            teacher_dir.mkdir(parents=True)
            teachers = {"dual": (dual, DUAL, dual_seconds)}
            for name, cols in (("eeg", EEG), ("eog", EOG)):
                t1 = time.time()
                out = train_head(np.ascontiguousarray(x_tr[:, cols]), y_tr, np.ascontiguousarray(x_va[:, cols]), y_va, seed)
                teachers[name] = (out, cols, round(time.time() - t1, 1))
            teacher_probs = {}
            for name, (out, cols, secs) in teachers.items():
                tid = f"{teacher_dir.name}_{name}"
                torch.save({"state_dict": out["state"], "best_epoch": out["best_epoch"], "input_columns": [cols.start, cols.stop],
                            "standardizer": {m: {"mean": torch.from_numpy(stats[m][0]), "std": torch.from_numpy(stats[m][1])}
                                             for m in stats}}, teacher_dir / f"teacher_{name}.ckpt")
                save_history(teacher_dir / f"history_{name}.csv", out["history"])
                save_predictions(teacher_dir / f"predictions_val_{name}.csv", val_rows, out["probs"])
                teacher_probs[name] = predict_probs(out["state"], X_all, unl_idx, cols)
                append_csv(args.runs_root / "teacher_index.csv", {
                    "teacher_id": tid, "modality": name, "repr": args.repr, "budget": args.budget,
                    "sampling_seed": sseed, "seed": seed, "n_labeled_samples": len(lab_idx),
                    "best_epoch": out["best_epoch"], "val_macro_f1": round(out["best_val_macro_f1"], 4),
                    "n_updates": out["n_updates"], "train_seconds": secs,
                    "ckpt": str(teacher_dir / f"teacher_{name}.ckpt"), "ckpt_sha256": sha256(teacher_dir / f"teacher_{name}.ckpt"),
                    "standardization": "fitted on labeled training samples only"},
                    ["teacher_id", "modality", "repr", "budget", "sampling_seed", "seed", "n_labeled_samples",
                     "best_epoch", "val_macro_f1", "n_updates", "train_seconds", "ckpt", "ckpt_sha256", "standardization"])
                print(f"teacher {name}: val macro-F1 {out['best_val_macro_f1']:.4f}, best epoch {out['best_epoch']} ({secs}s)", flush=True)

            pseudo = teacher_probs["dual"].argmax(1)
            conf = teacher_probs["dual"].max(1)
            eeg_pred = teacher_probs["eeg"].argmax(1)
            eog_pred = teacher_probs["eog"].argmax(1)
            cand, thr = accept_per_class(pseudo, conf, p_locked)  #阈值只由教师预测分布决定，不用真标签
            agree = (eeg_pred == pseudo) & (eog_pred == pseudo)
            accept_rule = {"D1": cand, "D2": cand & agree}

            for m in ("D1", "D2"):
                if m not in methods:
                    continue
                acc = accept_rule[m]
                t1 = time.time()
                out = train_student(x_tr, y_tr, X_all, unl_idx[acc], pseudo[acc], x_va, y_va, seed, mu=MU, lam=LAMBDA)
                info = {"pseudo": pseudo, "conf": conf, "eeg_pred": eeg_pred, "eog_pred": eog_pred,
                        "accepted": acc, "candidate": cand, "agree": agree, "thresholds": thr,
                        "teacher_dir": teacher_dir}
                results[m] = (out, int(acc.sum()), round(time.time() - t1, 1), info)

        #---------- 保存各方法的 run ----------
        for m in methods:
            out, n_pseudo, secs, info = results[m]
            run_dir = args.runs_root / run_ids[m]
            cfg = {
                "run_id": run_ids[m], "method": m, "method_description": DESCRIPTIONS[m],
                "representation": f"{args.repr}: {manifest['representation']} (EEG {dims['eeg']} + EOG {dims['eog']} = {D}, concat)",
                "budget": args.budget, "sampling_seed": sseed, "seed": seed,
                "n_labeled_subjects": len(labeled), "n_labeled_samples": len(lab_idx),
                "n_unlabeled_subjects_used": 0 if m == "D0" else len(unlabeled),
                "n_unlabeled_pool_samples": 0 if m == "D0" else len(unl_idx),
                "n_pseudo_labels_used": n_pseudo,
                "labeled_subjects": sorted(labeled),
                "standardization": "per-modality mean/std fitted on labeled training samples only",
                "student_head": {"type": f"Linear({D},5)", "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
                                 "labeled_batch": 128, "max_epochs": 50, "early_stop": "val macro-F1, patience 10"},
                "roles_file": str(args.roles), "roles_sha256": roles_sha,
                "cache": str(cache_dir), "cache_files_sha256": manifest["files_sha256"],
                "evaluate_path": str(EVALUATE_PATH), "evaluate_version": ev.EVALUATE_VERSION,
                "code_sha256": code_sha, "env": env,
            }
            if m != "D0":
                cfg["pseudo_labeling"] = {
                    "teacher_dir": str(info["teacher_dir"]), "teacher": "dual-modal head of the same seed (identical to D0)",
                    "rule": "per predicted class, accept top-p by dual-teacher confidence"
                            + ("; plus EEG-only and EOG-only teachers predict the same class" if m == "D2" else ""),
                    "p": p_locked, "calibration_file": str(args.calibration), "calibration_sha256": sha256(args.calibration),
                    "class_thresholds": {k: (round(v, 6) if v is not None else None) for k, v in info["thresholds"].items()},
                    "label_type": "hard", "weight": 1.0, "rounds": 1,
                    "student_mixing": {"pseudo_per_step": MU * 128, "lambda": LAMBDA,
                                       "pseudo_stream_seed": seed + 100003,
                                       "steps_per_epoch": "same as D0 (number of labeled batches)"},
                }
            cfg_text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            run_dir.mkdir()
            (run_dir / "config.yaml").write_text(cfg_text, encoding="utf-8")
            write_readme(run_dir, cfg, "started", "config written")

            save_history(run_dir / "history.csv", out["history"])
            torch.save({"state_dict": out["state"], "best_epoch": out["best_epoch"],
                        "standardizer": {mm: {"mean": torch.from_numpy(stats[mm][0]), "std": torch.from_numpy(stats[mm][1])}
                                         for mm in stats}}, run_dir / "best.ckpt")
            pred_path = run_dir / "predictions_val.csv"
            save_predictions(pred_path, val_rows, out["probs"])

            if m != "D0":
                #伪标签追踪：全部候选与被拒样本都保存；不包含无标签者真实标签
                acc = info["accepted"]
                with (run_dir / "pseudo_label_trace.csv").open("w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["sample_id", "method", "budget", "seed", "teacher_id", "pseudo_label", "confidence",
                                "eeg_prediction", "eog_prediction", "accepted", "weight"])
                    tid = f"{info['teacher_dir'].name}_dual"
                    for i in range(len(unl_ids)):
                        w.writerow([unl_ids[i], m, args.budget, seed, tid, int(info["pseudo"][i]),
                                    round(float(info["conf"][i]), 6), int(info["eeg_pred"][i]), int(info["eog_pred"][i]),
                                    int(acc[i]), 1.0 if acc[i] else 0.0])
                #按（伪）类别统计：候选数、接收数、平均置信度、模态分歧（只用教师输出，不用真标签）
                with (run_dir / "pseudo_label_stats.csv").open("w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["method", "budget", "seed", "pseudo_class", "n_predicted", "n_candidates", "n_accepted",
                                "mean_conf_accepted", "threshold", "frac_eeg_disagree", "frac_eog_disagree",
                                "frac_both_agree_among_candidates"])
                    for c, s in enumerate(STAGES):
                        pc = info["pseudo"] == c
                        ca = pc & info["candidate"]
                        ac = pc & acc
                        w.writerow([m, args.budget, seed, s, int(pc.sum()), int(ca.sum()), int(ac.sum()),
                                    round(float(info["conf"][ac].mean()), 4) if ac.any() else "",
                                    round(info["thresholds"][s], 6) if info["thresholds"][s] is not None else "",
                                    round(float((info["eeg_pred"][pc] != c).mean()), 4) if pc.any() else "",
                                    round(float((info["eog_pred"][pc] != c).mean()), 4) if pc.any() else "",
                                    round(float(info["agree"][ca].mean()), 4) if ca.any() else ""])

            met = ev.evaluate(pred_path)
            met.update({"run_id": run_ids[m], "best_epoch": out["best_epoch"], "n_epochs_run": len(out["history"]),
                        "n_updates": out["n_updates"], "curve_median_val_macro_f1": round(out["curve_median"], 4),
                        "train_seconds": secs, "n_pseudo_labels_used": n_pseudo,
                        "pseudo_samples_seen": out.get("pseudo_samples_seen", 0)})
            with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(met, f, indent=2)
            if abs(met["macro_f1"] - out["best_val_macro_f1"]) > 1e-6:
                print(f"ERROR: official macro-F1 {met['macro_f1']} != training-time {out['best_val_macro_f1']}")
                write_readme(run_dir, cfg, "failed", "metrics mismatch")
                sys.exit(1)
            write_readme(run_dir, cfg, "success", "metrics.json written")

            summary = {"run_id": run_ids[m], "method": m, "repr": args.repr, "budget": args.budget,
                       "sampling_seed": sseed, "seed": seed, "macro_f1": round(met["macro_f1"], 4),
                       **{f"f1_{k}": round(v, 4) for k, v in met["per_class_f1"].items()},
                       "n1_precision": round(met["n1_precision"], 4), "n1_recall": round(met["n1_recall"], 4),
                       "accuracy": round(met["accuracy"], 4), "best_epoch": out["best_epoch"],
                       "n_epochs_run": len(out["history"]), "n_updates": out["n_updates"],
                       "n_labeled_samples": len(lab_idx), "n_pseudo_labels_used": n_pseudo, "train_seconds": secs}
            append_csv(args.runs_root / "run_index.csv", summary, REGISTRY_FIELDS)
            print(f"{m} seed {seed}: macro-F1 {summary['macro_f1']:.4f}  N1 {summary['f1_N1']:.4f} "
                  f"(P {summary['n1_precision']:.3f} R {summary['n1_recall']:.3f})  REM {summary['f1_REM']:.4f}  "
                  f"pseudo used {n_pseudo}  best epoch {out['best_epoch']}  updates {out['n_updates']}  ({secs}s)", flush=True)

    print(f"\nSaved runs under {args.runs_root}")


if __name__ == "__main__":
    main()
