"""Step W2-B4: 原始冻结 EEGDM 双模态(EEG 2560 + EOG 2560 = 5120 维)的监督线性头,按人数预算训练。

配方照搬 D 的 train_linear_concat.py / multiseed_budget.py,以便 100% 与 25% 的旧结果在严格匹配时可复用:
  时间均值池化 → 标准化(只拟合"预算内有标签者"的中心样本)→ Linear(5120,5)
  AdamW lr=1e-3 wd=1e-4、batch 128、最多 50 轮、按验证 Macro-F1 选最佳权重、patience=10。
数据权限:只读预算内训练者与 8 名验证者的特征行;其余训练者、测试者的特征行不读。

用法(特征目录在团队存储,见 C/D 的 external_artifacts):
  python scripts/12_eegdm_budget_head.py --eeg-dir <eeg_cond14_v01> --eog-dir <eog_cond14_v01> \
      --budget label10 --seed 0
"""
import argparse
import sys

import numpy as np
import torch

from common import RUNS, SPLIT
import w2_common as w

CODE_VERSION = "v01"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eeg-dir", required=True)
    ap.add_argument("--eog-dir", required=True)
    ap.add_argument("--budget", required=True, choices=list(w.BUDGET_SIZES))
    ap.add_argument("--subsets", default=str(SPLIT / "label_subsets_v2.csv"))
    ap.add_argument("--sampling-seed", type=int, default=2024, help="必须与名单文件一致(v2 仅含 2024)")
    ap.add_argument("--seed", type=int, required=True, help="下游训练种子,正式 0-4")
    ap.add_argument("--device", default="cpu", help="与 D 一致默认 cpu;线性头很小")
    ap.add_argument("--runs-root", default=str(RUNS))
    args = ap.parse_args(argv)

    chosen, meta = w.selected_subjects(args.subsets, args.budget)
    if {meta[s]["sampling_seed"] for s in chosen} != {str(args.sampling_seed)}:
        sys.exit("错误: --sampling-seed 与名单文件不一致")
    run_id = f"W2_B_EEGDMhead_{args.budget}_sseed{args.sampling_seed}_seed{args.seed}_{CODE_VERSION}"
    run_dir = w.make_run_dir(run_id, args.runs_root)

    train, val, x_tr, x_va = w.load_eegdm_pair(args.eeg_dir, args.eog_dir, chosen)
    y_tr = np.array([int(r["y_true"]) for r in train])
    y_va = np.array([int(r["y_true"]) for r in val])
    mean, std = w.fit_standardizer(x_tr)           # 只用预算内有标签者
    x_tr_t = torch.from_numpy((x_tr - mean) / std)
    x_va_t = torch.from_numpy((x_va - mean) / std)

    torch.set_num_threads(4)
    model = torch.nn.Linear(2 * w.POOLED_DIM, 5)
    result = w.fit(model, lambda ids: x_tr_t[ids], y_tr, lambda ids: x_va_t[ids], y_va,
                   seed=args.seed, device=args.device)
    probs = w.predict_probs(model, lambda ids: x_va_t[ids], len(y_va), args.device)

    config = dict(experiment="W2 实验② 监督预算曲线(原始冻结 EEGDM 头)", representation="eegdm_raw_frozen_eegeog_5120",
                  modalities="EEG+EOG", context_length=1, nominal_budget=args.budget,
                  labeled_subjects=len(chosen), unlabeled_signal_subjects=0,
                  subset_file=args.subsets, subset_sha256=w.sha256_file(args.subsets),
                  sampling_seed=args.sampling_seed, downstream_seed=args.seed,
                  train_samples=len(y_tr), val_samples=len(y_va), standardization="fit on labeled train subjects only",
                  head="Linear(5120,5)", optimizer="AdamW lr=1e-3 wd=1e-4", batch_size=128, max_epochs=50,
                  patience=10, selection="val macro-F1", eeg_dir=str(args.eeg_dir), eog_dir=str(args.eog_dir),
                  eeg_index_sha256=w.sha256_file(f"{args.eeg_dir}/feature_index.csv"),
                  eog_index_sha256=w.sha256_file(f"{args.eog_dir}/feature_index.csv"),
                  device=args.device, torch=str(torch.__version__), test_used=False)
    m = w.finish_run(run_dir, run_id, config, result, val, probs, chosen, (mean, std),
                     "原始冻结 EEGDM 特征;本 run 只用预算内有标签者的信号与标签,其余训练者未读取。")
    print(f"{run_id}: macro-F1 {m['macro_f1']:.4f}  N1 P/R/F1 "
          f"{m['n1_precision']:.3f}/{m['n1_recall']:.3f}/{m['n1_f1']:.3f}  -> {run_dir}")


if __name__ == "__main__":
    main()
