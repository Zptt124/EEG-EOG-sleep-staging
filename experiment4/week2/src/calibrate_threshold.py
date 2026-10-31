#calibrate_threshold.py —— 第二周 实验④ 阶段二：在有标签者内部按人留出，校准伪标签的按类别接收比例 p
#做法（LOSO，留一人交叉）：
#  每一折留出 1 名有标签者，用其余有标签者训练三个教师：EEG+EOG 拼接（双模态）、EEG-only、EOG-only
#  标准化只用该折的训练者拟合；教师早停与正式教师一致（验证 Macro-F1，patience 10）
#  给留出者打伪标签；14 折（10% 为 5 折）的留出者合起来，模拟"无标签池"
#  在模拟池上，按"教师预测类别"分组，每组收置信度最高的前 p%：
#      D1 = 只看双模态教师置信度；D2 = D1 候选 + EEG、EOG 教师都预测同一类
#  留出者的真标签只在这里用来评估伪标签质量（它们本来就是有标签者），真正的无标签者标签从不打开
#选择规则（事先固定，不看验证集分数）：D1 按类平均伪标签精确率 >= 0.85 的最大 p；都不满足取最小 p
#用法（Ubuntu）：python -u calibrate_threshold.py --budget label25 --roles 名单.csv --cache 缓存 --output 输出文件夹

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from label_guard import LabelLeakError, assert_hidden, hide_labels, load_roles
from ssl_common import STAGES, fit_standardizer, read_csv, sha256, train_head

CANDIDATE_P = [0.2, 0.4, 0.6]
PRECISION_TARGET = 0.85
TEACHER_SEED = 0  #内部教师固定用种子 0；选出的 p 对 5 个下游种子统一使用


def accept_per_class(pred, conf, p):
    #按预测类别分组，每组接收置信度最高的前 p（并列时阈值处全部接收）
    acc = np.zeros(len(pred), dtype=bool)
    thresholds = {}
    for c in range(5):
        idx = np.where(pred == c)[0]
        if len(idx) == 0:
            thresholds[STAGES[c]] = None
            continue
        thr = float(np.quantile(conf[idx], 1 - p))
        thresholds[STAGES[c]] = thr
        acc[idx[conf[idx] >= thr]] = True
    return acc, thresholds


def summarize(method, p, accepted, pred, y_true, thresholds):
    #统计一个规则下：接收数、每类接收数、每类伪标签精确率、按类平均精确率、真 N1 被正确收入的比例
    row = {"method": method, "p": p, "n_pool": len(pred), "n_accepted": int(accepted.sum()),
           "coverage": round(float(accepted.mean()), 4)}
    precs = []
    for c, s in enumerate(STAGES):
        sel = accepted & (pred == c)
        n = int(sel.sum())
        prec = float((y_true[sel] == c).mean()) if n else float("nan")
        row[f"n_acc_{s}"] = n
        row[f"prec_{s}"] = round(prec, 4) if n else ""
        row[f"thr_{s}"] = round(thresholds[s], 4) if thresholds[s] is not None else ""
        if n:
            precs.append(prec)
    row["macro_precision"] = round(float(np.mean(precs)), 4)
    row["overall_precision"] = round(float((y_true[accepted] == pred[accepted]).mean()), 4) if accepted.any() else ""
    true_n1 = y_true == 1
    row["true_N1_captured"] = round(float((accepted & (pred == 1) & true_n1).sum() / max(true_n1.sum(), 1)), 4)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", choices=["label10", "label25"], required=True)
    ap.add_argument("--roles", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if (args.output / "calibration_decision.json").exists():
        print(f"ERROR: {args.output} already has a calibration, choose a new folder")
        sys.exit(1)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    t_start = time.time()

    #---------- 标签隐藏：无标签者 y=-1，本脚本完全不读他们的特征 ----------
    rows = read_csv(args.cache / "sample_index.csv")
    labeled, unlabeled = load_roles(args.roles, args.budget)
    sseed = read_csv(args.roles)[0]["sampling_seed"]
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
    del y_all

    keep = is_lab | is_val
    raw = {m: np.load(args.cache / f"{m}_pooled.npy", mmap_mode="r") for m in ("eeg", "eog")}
    x = {m: np.ascontiguousarray(raw[m][keep], dtype=np.float32) for m in raw}
    del raw
    subj_k = subject[keep]
    lab_k = is_lab[keep]
    val_k = is_val[keep]
    y_k = y_train[keep]  #有标签者是真标签；验证集也是真标签（只用于早停）
    sid_k = np.array([r["sample_id"] for r, k in zip(rows, keep) if k])
    y_val = y_k[val_k]
    print(f"OK: {args.budget} sseed={sseed}: {len(labeled)} labeled subjects, LOSO with {len(labeled)} folds; "
          f"unlabeled subjects not loaded", flush=True)

    #---------- LOSO：每折训练三个教师，预测留出者 ----------
    pool_parts, fold_rows = [], []
    for fold, held in enumerate(sorted(labeled)):
        tr = lab_k & (subj_k != held)
        ho = lab_k & (subj_k == held)
        stats = {m: fit_standardizer(x[m], tr) for m in x}
        z = {m: (x[m] - stats[m][0]) / stats[m][1] for m in x}
        inputs = {"dual": np.concatenate([z["eeg"], z["eog"]], axis=1), "eeg": z["eeg"], "eog": z["eog"]}
        probs_ho, frow = {}, {"fold": fold, "heldout_subject": held, "n_train_subjects": len(labeled) - 1,
                              "n_train_samples": int(tr.sum()), "n_heldout_samples": int(ho.sum())}
        t0 = time.time()
        for name, xi in inputs.items():
            out = train_head(xi[tr], y_k[tr], xi[val_k], y_val, TEACHER_SEED)
            model = torch.nn.Linear(xi.shape[1], 5)
            model.load_state_dict(out["state"])
            model.eval()
            with torch.inference_mode():
                probs_ho[name] = model(torch.from_numpy(np.ascontiguousarray(xi[ho]))).softmax(1).numpy()
            frow[f"{name}_best_epoch"] = out["best_epoch"]
            frow[f"{name}_val_macro_f1"] = round(out["best_val_macro_f1"], 4)
        del z, inputs
        pred = probs_ho["dual"].argmax(1)
        part = {"sample_id": sid_k[ho], "subject_id": subj_k[ho], "y_true": y_k[ho], "pred": pred,
                "conf": probs_ho["dual"].max(1), "eeg_pred": probs_ho["eeg"].argmax(1),
                "eog_pred": probs_ho["eog"].argmax(1)}
        frow["heldout_teacher_accuracy"] = round(float((pred == part["y_true"]).mean()), 4)
        fold_rows.append(frow)
        pool_parts.append(part)
        print(f"fold {fold + 1}/{len(labeled)} held={held} n={int(ho.sum())} "
              f"teacher acc on held-out {frow['heldout_teacher_accuracy']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    pool = {k: np.concatenate([p[k] for p in pool_parts]) for k in pool_parts[0]}
    agree = (pool["eeg_pred"] == pool["pred"]) & (pool["eog_pred"] == pool["pred"])

    #保存模拟池逐段结果（全部是有标签者，可以带真标签）
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

    #---------- 各候选 p 下 D1 / D2 的伪标签质量 ----------
    cal_rows = []
    all_acc = np.ones(len(pool["pred"]), dtype=bool)
    cal_rows.append(summarize("accept_all", 1.0, all_acc, pool["pred"], pool["y_true"],
                              {s: None for s in STAGES}))
    for p in CANDIDATE_P:
        acc1, thr = accept_per_class(pool["pred"], pool["conf"], p)
        cal_rows.append(summarize("D1", p, acc1, pool["pred"], pool["y_true"], thr))
        cal_rows.append(summarize("D2", p, acc1 & agree, pool["pred"], pool["y_true"], thr))
    with (args.output / "threshold_calibration.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)

    print("\n=== pseudo-label quality on the simulated pool (labeled subjects, LOSO) ===")
    print(f"{'method':10s} {'p':>4s} {'cover':>6s} {'macroP':>7s} " + " ".join(f"{'P_' + s:>6s}" for s in STAGES)
          + " " + " ".join(f"{'n_' + s:>6s}" for s in STAGES) + "  N1_captured")
    for r in cal_rows:
        print(f"{r['method']:10s} {r['p']:4.1f} {r['coverage']:6.3f} {r['macro_precision']:7.4f} "
              + " ".join(f"{(r['prec_' + s] if r['prec_' + s] != '' else float('nan')):6.3f}" for s in STAGES) + " "
              + " ".join(f"{r['n_acc_' + s]:6d}" for s in STAGES) + f"  {r['true_N1_captured']:.3f}")

    #---------- 按事先固定的规则选 p ----------
    ok = [r["p"] for r in cal_rows if r["method"] == "D1" and r["macro_precision"] >= PRECISION_TARGET]
    chosen = max(ok) if ok else min(CANDIDATE_P)
    decision = {"budget": args.budget, "sampling_seed": sseed, "chosen_p": chosen,
                "rule": f"largest p in {CANDIDATE_P} with D1 macro pseudo-label precision >= {PRECISION_TARGET}; "
                        f"otherwise smallest p",
                "rule_satisfied": bool(ok), "teacher_seed": TEACHER_SEED,
                "teacher_early_stopping": "validation macro-F1, patience 10 (same as formal teachers); "
                                          "validation scores NOT used to choose p",
                "n_folds": len(labeled), "n_pool": int(len(pool["pred"])),
                "roles_sha256": sha256(args.roles), "cache_files_sha256":
                    json.loads((args.cache / "cache_manifest.json").read_text(encoding="utf-8"))["files_sha256"],
                "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
                "minutes": round((time.time() - t_start) / 60, 1)}
    with (args.output / "calibration_decision.json").open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    print(f"\nDECISION: chosen p = {chosen} (rule satisfied: {bool(ok)})")
    print(f"Saved: {args.output}  ({decision['minutes']} min)")


if __name__ == "__main__":
    main()
