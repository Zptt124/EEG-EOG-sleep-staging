"""Step W2-B5: 25% 预算下的时间上下文头(长度 1 对照 vs 长度 3),两种逐段表征。

固定结构(锁定于 reports/week2/context_config.yaml,不在 8 人验证集上搜索):
  标准化 → Linear(D,128)+ReLU+Dropout(0.2) → 1 层 BiGRU(hidden 64) → 取中心步 → Linear(128,5)
长度 1 与长度 3 用同一个头,只把序列长度从 1 改为 3(前一段、当前段、后一段,预测中心段)。
表征:
  eegdm  原始冻结 EEGDM 双模态 5120 维(--eeg-dir/--eog-dir)
  cnn    A 的 25% 双通道 CNN 编码器逐段特征(--cnn-features,npz:sample_id[str]、x[N,F];eval 模式导出)

数据权限:训练窗口只含 label25 选中的 14 人,邻居取自同人同夜真实片段或中心复制填充(window_index.csv);
不输入邻居真标签;其余 40 名训练者的信号不进入。验证窗口邻居只用验证者自己的信号。
所有设置共用同一中心样本集合(window_index.csv 的 center 集合),与特征索引不一致即报错。

用法:
  python scripts/13_context_head.py --repr eegdm --eeg-dir <eeg> --eog-dir <eog> --context 3 --seed 0
  python scripts/13_context_head.py --repr eegdm --eeg-pooled e.npz --eog-pooled o.npz --eeg-index idx.csv --context 3 --seed 0
  python scripts/13_context_head.py --repr cnn --cnn-features <npz> --context 1 --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from common import PROCESSED, RUNS, SPLIT
import w2_common as w

CODE_VERSION = "v01"
WINDOW_INDEX = Path(__file__).resolve().parent.parent / "reports" / "week2" / "window_index.csv"
PROJ, HIDDEN, DROPOUT = 128, 64, 0.2


class ContextHead(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.Linear(dim, PROJ), torch.nn.ReLU(), torch.nn.Dropout(DROPOUT))
        self.gru = torch.nn.GRU(PROJ, HIDDEN, num_layers=1, batch_first=True, bidirectional=True)
        self.out = torch.nn.Linear(2 * HIDDEN, 5)

    def forward(self, x):                      # x: [B, L, D];L 为奇数,中心步 = L // 2
        h, _ = self.gru(self.proj(x))
        return self.out(h[:, x.shape[1] // 2])


def window_lookup(path, chosen):
    rows = w.read_csv_rows(path)
    return {r["center_sample_id"]: r for r in rows
            if (r["split"] == "train" and r["subject_id"] in chosen) or r["split"] == "val"}


def labels_from_processed(centers, processed):
    """cnn 表征没有特征索引:真标签只读选中训练者与验证者记录的 NPZ labels/epoch_index。"""
    by_record = {}
    for r in centers:
        by_record.setdefault(r["record_id"], []).append(r)
    out = {}
    for rec in by_record:
        with np.load(Path(processed) / f"{rec}.npz", allow_pickle=False) as d:
            out.update({w_sid: int(y) for w_sid, y in
                        ((f"{rec}_{int(i):04d}", y) for i, y in zip(d["epoch_index"], d["labels"]))})
    return out


def build_inputs(args, chosen, windows):
    """返回 (train_rows, val_rows, x_train[N,D], x_val[M,D]);行序与窗口索引一致性已校验。"""
    if args.repr == "eegdm":
        eeg_src, eog_src, eeg_idx, eog_idx, pooled = w.feature_inputs(args)
        train, val, x_tr, x_va = w.load_eegdm_pair(eeg_src, eog_src, chosen, eeg_idx, eog_idx, pooled)
    else:
        centers = list(windows.values())
        labels = labels_from_processed(centers, args.processed)
        make = lambda part: [dict(sample_id=r["center_sample_id"], subject_id=r["subject_id"], record_id=r["record_id"],
                                  epoch_index=r["epoch_index"], split=part, y_true=str(labels[r["center_sample_id"]]))
                             for r in centers if r["split"] == part]
        train, val = make("train"), make("val")
        x_tr = w.load_encoder_features(args.cnn_features, train)
        x_va = w.load_encoder_features(args.cnn_features, val)
    ids = {r["sample_id"] for r in train + val}
    if ids != set(windows):
        raise ValueError(f"特征中心样本与 window_index 不一致: 特征独有 {len(ids - set(windows))}, "
                         f"窗口独有 {len(set(windows) - ids)}")
    return train, val, x_tr, x_va


def window_ids(rows, windows, pos, length):
    """每个中心对应的行位置数组 [N, L];邻居必须落在同一集合(同人同夜)内。"""
    cols = [[pos[windows[r["sample_id"]]["previous_sample_id"]] for r in rows],
            [pos[r["sample_id"]] for r in rows],
            [pos[windows[r["sample_id"]]["next_sample_id"]] for r in rows]]
    arr = np.stack(cols, axis=1)
    return arr if length == 3 else arr[:, 1:2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repr", required=True, choices=["eegdm", "cnn"])
    w.add_feature_args(ap)
    ap.add_argument("--cnn-features")
    ap.add_argument("--context", type=int, required=True, choices=[1, 3])
    ap.add_argument("--budget", default="label25", choices=["label25"])
    ap.add_argument("--subsets", default=str(SPLIT / "label_subsets_v2.csv"))
    ap.add_argument("--window-index", default=str(WINDOW_INDEX))
    ap.add_argument("--processed", default=str(PROCESSED))
    ap.add_argument("--sampling-seed", type=int, default=2024)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs-root", default=str(RUNS))
    args = ap.parse_args(argv)
    if args.repr == "eegdm":
        try:
            w.feature_inputs(args)
        except ValueError as e:
            sys.exit(f"错误: {e}")
    if args.repr == "cnn" and not args.cnn_features:
        sys.exit("错误: cnn 需要 --cnn-features")

    chosen, _ = w.selected_subjects(args.subsets, args.budget)
    windows = window_lookup(args.window_index, set(chosen))
    tag = "EEGDM" if args.repr == "eegdm" else "CNN"
    run_id = f"W2_B_CTX{args.context}_{tag}_{args.budget}_sseed{args.sampling_seed}_seed{args.seed}_{CODE_VERSION}"
    run_dir = w.make_run_dir(run_id, args.runs_root)

    train, val, x_tr, x_va = build_inputs(args, chosen, windows)
    mean, std = w.fit_standardizer(x_tr)           # 只用 14 名有标签者的中心样本
    x_tr_t = torch.from_numpy((x_tr - mean) / std)
    x_va_t = torch.from_numpy((x_va - mean) / std)
    pos_tr = {r["sample_id"]: i for i, r in enumerate(train)}
    pos_va = {r["sample_id"]: i for i, r in enumerate(val)}
    idx_tr = torch.from_numpy(window_ids(train, windows, pos_tr, args.context))
    idx_va = torch.from_numpy(window_ids(val, windows, pos_va, args.context))
    y_tr = np.array([int(r["y_true"]) for r in train])
    y_va = np.array([int(r["y_true"]) for r in val])

    device = w.pick_device(args.device)
    w.seed_everything(args.seed)                           # 先设种子再建模型
    model = ContextHead(x_tr.shape[1])
    f_tr = lambda ids: x_tr_t[idx_tr[ids]]
    f_va = lambda ids: x_va_t[idx_va[ids]]
    result = w.fit(model, f_tr, y_tr, f_va, y_va, device=device)
    probs = w.predict_probs(model, f_va, len(y_va), device)

    pad_tr = sum(windows[r["sample_id"]]["boundary_rule"] != "real" for r in train)
    config = dict(experiment="W2 实验② 时间上下文(25%)", representation=("eegdm_raw_frozen_eegeog_5120" if args.repr == "eegdm"
                                                                       else "cnn_dual_channel_label25_encoder"),
                  context_length=args.context, nominal_budget=args.budget, labeled_subjects=len(chosen),
                  unlabeled_signal_subjects=0, sampling_seed=args.sampling_seed, downstream_seed=args.seed,
                  window_index=str(args.window_index), window_index_sha256=w.sha256_file(args.window_index),
                  subset_sha256=w.sha256_file(args.subsets),
                  head=f"proj{PROJ}+ReLU+Dropout{DROPOUT} -> BiGRU(h{HIDDEN},1 layer) -> center step -> Linear(128,5)",
                  optimizer="AdamW lr=1e-3 wd=1e-4", batch_size=128, max_epochs=50, patience=10,
                  selection="val macro-F1", padded_train_windows=int(pad_tr), train_windows=len(train),
                  val_windows=len(val), neighbor_labels_used=False, test_used=False, device=device,
                  torch=str(torch.__version__), deterministic_algorithms=False,
                  encoder_features=args.cnn_features, eeg_source=args.eeg_pooled or args.eeg_dir,
                  eog_source=args.eog_pooled or args.eog_dir,
                  feature_input="pooled_npz" if args.eeg_pooled else ("raw_npy_dir" if args.eeg_dir else "NA"))
    m = w.finish_run(run_dir, run_id, config, result, val, probs, chosen, (mean, std),
                     "上下文对照:长度 1 与长度 3 共用同一头、同一中心样本集合;邻居缺失处中心复制(window_index.csv 记录 mask)。")
    print(f"{run_id}: macro-F1 {m['macro_f1']:.4f}  N1 F1 {m['n1_f1']:.3f}  -> {run_dir}")


if __name__ == "__main__":
    main()
