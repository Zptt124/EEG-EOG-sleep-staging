"""Step W3-B2: 冻结 EEGDM 双模态池化特征的线性探针(第三周协议:选轮次只用训练侧)。

与第二周 12_eegdm_budget_head.py 同一配方(标准化 → Linear(5120,5),AdamW lr=1e-3 wd=1e-4,batch 128),
区别在选模:第二周按开发集 Macro-F1 早停;第三周开发集只用于最终评估一次。
  1. 预算内训练者按受试者分 K 折(K=min(5,人数);折分配固定 fold_seed=0,与下游 seed 和 pooling 无关,保证配对)。
  2. 每折标准化只拟合折内训练者,训练固定 50 轮,记录每轮对留出受试者的预测。
  3. 拼接各折留出预测,取 Macro-F1 最高的轮次 E*(并列取最小)。
  4. 在全部预算内训练者上重新标准化、同 seed 训练 E* 轮,对 8 名验证者预测一次。
E* 处的训练侧留出预测写入 oof_train.csv,供 N1 分数调整(28_b_n1_score_adjust.py)选系数。

用法:
  python scripts/27_b_w3_linear_probe.py --arm std_t1 --budget label25 --seed 0 \
      --eeg-pooled features/c_pooled_v01/EEG_S0_pooled_154124x2560_unstandardized_v01.npz \
      --eog-pooled features/c_pooled_v01/EOG_C_rebuilt_S0_pooled_154124x2560_unstandardized_v01.npz \
      --eeg-index features/c_pooled_v01/sample_index.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from common import RUNS, SPLIT
import w2_common as w

ev = w.ev
CODE_VERSION = "v01"
FOLD_SEED = 0
MAX_EPOCHS = 50


def subject_folds(subjects, k):
    order = np.random.default_rng(FOLD_SEED).permutation(sorted(subjects))
    return [sorted(f.tolist()) for f in np.array_split(order, k)]


def standardize(x_fit, *xs):
    mean, std = w.fit_standardizer(x_fit)
    return (mean, std), [torch.from_numpy((x - mean) / std) for x in (x_fit, *xs)]


def train_head(x_tr, y_tr, seed, epochs, x_eval=None):
    """固定轮数训练;x_eval 给出时返回每轮的预测概率。与 w.fit 相同的种子顺序与优化器。"""
    w.seed_everything(seed, deterministic=True)
    model = torch.nn.Linear(x_tr.shape[1], 5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y = torch.as_tensor(y_tr, dtype=torch.long)
    per_epoch, updates = [], 0
    for _ in range(epochs):
        model.train()
        for ids in torch.randperm(len(y)).split(128):
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x_tr[ids]), y[ids]).backward()
            opt.step()
            updates += 1
        if x_eval is not None:
            per_epoch.append(w.predict_probs(model, lambda ids: x_eval[ids], len(x_eval)))
    return model, per_epoch, updates


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arm", required=True, help="条件标签,如 std_t1、mean_t1、std_t10;写入 run_id")
    ap.add_argument("--eeg-pooled", required=True)
    ap.add_argument("--eog-pooled", required=True)
    ap.add_argument("--eeg-index", required=True)
    ap.add_argument("--eog-index", help="缺省同 --eeg-index")
    ap.add_argument("--budget", required=True, choices=["label10", "label25"])
    ap.add_argument("--subsets", default=str(SPLIT / "label_subsets_v2.csv"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--runs-root", default=str(RUNS))
    args = ap.parse_args(argv)

    chosen, meta = w.selected_subjects(args.subsets, args.budget)
    if {meta[s]["sampling_seed"] for s in chosen} != {"2024"}:
        sys.exit("错误: 名单 sampling_seed 应为 2024")
    run_id = f"W3_B_{args.arm}_{args.budget}_sseed2024_seed{args.seed}_{CODE_VERSION}"
    run_dir = w.make_run_dir(run_id, args.runs_root)
    eog_index = args.eog_index or args.eeg_index
    train, val, x_tr, x_va = w.load_eegdm_pair(args.eeg_pooled, args.eog_pooled, chosen,
                                               args.eeg_index, eog_index, pooled=True)
    y_tr = np.array([int(r["y_true"]) for r in train])
    subj = np.array([r["subject_id"] for r in train])
    torch.set_num_threads(4)
    t0 = time.perf_counter()

    # 1-3. 训练侧按受试者留出,选轮次
    folds = subject_folds(chosen, min(5, len(chosen)))
    fold_of = np.empty(len(y_tr), dtype=int)
    oof = np.zeros((MAX_EPOCHS, len(y_tr), 5))
    for k, held in enumerate(folds):
        ho = np.isin(subj, held)
        fold_of[ho] = k
        _, (xf, xh) = standardize(x_tr[~ho], x_tr[ho])
        _, per_epoch, _ = train_head(xf, y_tr[~ho], args.seed, MAX_EPOCHS, xh)
        oof[:, ho] = np.stack(per_epoch)
    curve = []
    for e in range(MAX_EPOCHS):
        m = ev.compute_metrics(y_tr, oof[e].argmax(1))
        curve.append(dict(epoch=e + 1, oof_macro_f1=m["macro_f1"], oof_n1_f1=m["n1_f1"]))
    best = max(curve, key=lambda r: (r["oof_macro_f1"], -r["epoch"]))
    e_star = best["epoch"]

    # 4. 全部预算内训练者重训 E* 轮,开发集预测一次
    (mean, std), (xa, xv) = standardize(x_tr, x_va)
    model, _, updates = train_head(xa, y_tr, args.seed, e_star)
    probs = w.predict_probs(model, lambda ids: xv[ids], len(xv))
    elapsed = time.perf_counter() - t0

    w.write_prediction_table(run_dir / "predictions_val.csv", val, probs)
    met = ev.evaluate(run_dir / "predictions_val.csv", reference=w.VAL_REFERENCE)
    met.update(run_id=run_id, selected_epoch=e_star, oof_macro_f1_at_selected=best["oof_macro_f1"])
    (run_dir / "metrics.json").write_text(json.dumps(met, indent=2, ensure_ascii=False), encoding="utf-8")
    ev.per_subject_table(ev.load_predictions(run_dir / "predictions_val.csv")).to_csv(run_dir / "by_subject.csv", index=False)
    pd.DataFrame(curve).to_csv(run_dir / "inner_cv.csv", index=False)
    o = pd.DataFrame({k: [r[k] for r in train] for k in ("sample_id", "subject_id", "record_id", "epoch_index")})
    o["y_true"], o["fold"], o["y_pred"] = y_tr, fold_of, oof[e_star - 1].argmax(1)
    for j, c in enumerate(w.PROB_COLS):
        o[c] = oof[e_star - 1][:, j]
    o.to_csv(run_dir / "oof_train.csv", index=False, float_format="%.8g")
    torch.save(dict(state=model.state_dict(), epoch=e_star, mean=mean, std=std), run_dir / "final.ckpt")
    np.savez(run_dir / "normalization.npz", mean=mean, std=std)
    pd.DataFrame({"subject_id": chosen}).to_csv(run_dir / "labeled_subjects.csv", index=False)

    config = dict(experiment="W3 实验B 线性读出(冻结 EEGDM,EEG+EOG)", arm=args.arm, run_id=run_id,
                  nominal_budget=args.budget, labeled_subjects=len(chosen), sampling_seed=2024, downstream_seed=args.seed,
                  subset_sha256=w.sha256_file(args.subsets), train_samples=len(y_tr), val_samples=len(val),
                  head="Linear(5120,5)", optimizer="AdamW lr=1e-3 wd=1e-4", batch_size=128,
                  standardization="fit on training-side subjects only (fold-train in inner CV, all budget subjects for final)",
                  selection=f"inner subject-wise CV on budget subjects, K={len(folds)}, fold_seed={FOLD_SEED}, "
                            f"fixed {MAX_EPOCHS} epochs per fold, epoch = argmax pooled OOF macro-F1 (ties -> earliest)",
                  folds=folds, selected_epoch=e_star, dev_used_for="final evaluation only",
                  eeg_source=args.eeg_pooled, eog_source=args.eog_pooled,
                  eeg_source_sha256=w.sha256_file(args.eeg_pooled), eog_source_sha256=w.sha256_file(args.eog_pooled),
                  eeg_index_sha256=w.sha256_file(args.eeg_index), eog_index_sha256=w.sha256_file(eog_index),
                  torch=str(torch.__version__), device="cpu", test_used=False, code_commit=w.code_version(),
                  evaluate_version=ev.EVALUATE_VERSION, val_reference_sha256=w.sha256_file(w.VAL_REFERENCE),
                  predictions_sha256=w.sha256_file(run_dir / "predictions_val.csv"),
                  final_updates=updates, elapsed_seconds=round(elapsed, 2))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"{run_id}: E*={e_star} (OOF macro-F1 {best['oof_macro_f1']:.4f})  dev macro-F1 {met['macro_f1']:.4f}  "
          f"N1 P/R/F1 {met['n1_precision']:.3f}/{met['n1_recall']:.3f}/{met['n1_f1']:.3f}  {elapsed:.0f}s")


if __name__ == "__main__":
    main()
