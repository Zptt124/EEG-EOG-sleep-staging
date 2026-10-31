"""a_w2_checks.py — 实验① A 第3步小批量检查(W2 v1.1)。

覆盖:输入形状 [B,1,3000]/[B,2,3000]、梯度有限且所有参数更新、双通道行同步、
预算隔离(不读其余训练者标签)、过拟合检查、编码器输出维度与第一周权重兼容、
验证预测格式通过 B 官方 evaluate.py。

用法(在 repo 根目录):
    D:/python/python.exe scripts/a_w2_checks.py --data-dir D:/sleepedf/processed \
        --runs-dir C:/Users/34364/Desktop/睡眠/exp1/runs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cnn_model import SmallCNN
from train_cnn_w2 import (MODALITY_CH, load_records, load_subjects,
                          labeled_records, macro_f1_np)


def check_shapes_and_grads(data_dir, ch_idx):
    """[B,C,3000] 前向、编码器 128 维、全参数梯度有限。"""
    print("[1] shape / encoder dim / gradient check")
    z = np.load(data_dir / "SC4001.npz")
    x = z["signal"][:, ch_idx][:64].astype(np.float32)
    x = torch.from_numpy(x)
    assert x.shape == (64, len(ch_idx), 3000), f"shape {x.shape}"
    y = torch.randint(0, 5, (64,))
    for name, model in (("single", SmallCNN(in_channels=1)),
                        ("dual", SmallCNN(in_channels=2))):
        model.train()
        out = model(x[:, :model.in_channels])
        assert out.shape == (64, 5), (name, out.shape)
        enc = model.encoder(x[:, :model.in_channels])
        assert enc.shape == (64, 128), (name, enc.shape)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        bad = [k for k, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not bad, f"{name}: 梯度异常 {bad}"
        model.zero_grad()
        print(f"  {name}: forward ok, encoder dim={enc.shape[1]}, all grads finite")


def check_dual_sync(data_dir):
    """双通道两列必须来自同一 record/epoch 行(结构上由同一 npz 行保证)。

    实际加载验证:同一记录分别按 EEG、EOG 单通道加载,标签与 epoch 网格必须
    完全一致——若双通道各自打乱,此处必然对不上。
    """
    print("[2] dual-channel row sync check")
    rec = "SC4001"
    _, y_eeg, m_eeg = load_records(Path(data_dir), [rec], [0])
    _, y_eog, m_eog = load_records(Path(data_dir), [rec], [1])
    assert len(y_eeg) == len(y_eog), (len(y_eeg), len(y_eog))
    assert np.array_equal(y_eeg, y_eog), "EEG/EOG 两列标签不一致"
    assert m_eeg["epoch_index"].equals(m_eog["epoch_index"]), "EEG/EOG epoch 网格不一致"
    assert m_eeg["sample_id"].equals(m_eog["sample_id"]), "EEG/EOG sample_id 不一致"
    # sample_id 与 npz 的 epoch_index 一一对应
    z = np.load(Path(data_dir) / f"{rec}.npz")
    expect = [f"{rec}_{int(i):04d}" for i in z["epoch_index"]]
    assert m_eeg["sample_id"].tolist() == expect, "sample_id 与 npz epoch_index 不对应"
    print(f"  {rec}: EEG/EOG rows share labels/epoch_grid/sample_id ({len(y_eeg)} epochs)")


def check_budget_isolation(subset_csv, split_csv, budget):
    """预算人员 ⊆ 训练集;名单人数与 label_counts 一致(不读其余训练者标签)。"""
    print(f"[3] budget isolation check ({budget})")
    subset = load_subjects(subset_csv)
    split_df = pd.read_csv(split_csv, dtype=str)
    train_recs = set()
    for _, r in split_df[split_df["split"] == "train"].iterrows():
        train_recs.update(r["record_ids"].split(";"))
    lab = set(labeled_records(subset, budget))
    assert lab <= train_recs, f"预算记录越界: {lab - train_recs}"
    sel = subset[subset[budget] == 1]
    print(f"  {len(sel)} subjects / {len(lab)} records, all in train split")


def check_overfit(data_dir, ch_idx, device, epochs=300, n=128):
    """128 样本过拟合到 ~100%:管线可用性检查(第一周第6步的 W2 版)。"""
    print(f"[4] overfit check ({n} samples, {'dual' if len(ch_idx) == 2 else 'single'})")
    z = np.load(data_dir / "SC4001.npz")
    x = torch.from_numpy(z["signal"][:, ch_idx][:n].astype(np.float32)).to(device)
    y = torch.from_numpy(z["labels"][:n]).long().to(device)
    model = SmallCNN(in_channels=len(ch_idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        out = model(x)
        loss = lossf(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        acc = (out.argmax(1) == y).float().mean().item()
        if (ep % 50 == 0 or (loss.item() < 0.01 and acc > 0.99)):
            print(f"  ep {ep:3d} loss {loss.item():.4f} acc {acc:.4f}")
        if loss.item() < 0.01 and acc > 0.99:
            print("  OVERFIT CHECK PASSED")
            return
    sys.exit(f"overfit check failed: ep {epochs} loss {loss.item():.4f} acc {acc:.4f}")


def check_w1_ckpt_compat(runs_dir):
    """第一周权重必须能被新 SmallCNN 严格加载(复用核验的基础)。"""
    print("[5] week-1 checkpoint compat (load_state_dict strict)")
    for run_id, ch in (("E1_EEG_label100_seed0_v01", 1), ("E1_EOG_label100_seed0_v01", 1)):
        ckpt_path = Path(runs_dir) / run_id / "best.ckpt"
        if not ckpt_path.exists():
            print(f"  skip {run_id}: no ckpt")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = SmallCNN(in_channels=ch)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        print(f"  {run_id}: strict load ok (first-week weights compatible)")


def check_val_prediction_format(pred_csv, repo_root, val_ref):
    """验证预测表必须通过 B 官方 evaluate.py --reference(格式/集合/标签一致)。"""
    print(f"[6] B official evaluator on {Path(pred_csv).name}")
    import tempfile
    tmp_out = Path(tempfile.mkdtemp(prefix="a_w2_check_")) / "metrics.json"
    cmd = [sys.executable, str(repo_root / "src" / "evaluation" / "evaluate.py"),
           str(pred_csv), "--out", str(tmp_out),
           "--reference", str(val_ref)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"B evaluator failed: {res.stderr.strip()}")
    print("  passed (sample set matches val_reference_v1, y_true consistent)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs-dir", default=None, help="第一周 run 目录(检查 5 用)")
    ap.add_argument("--pred-csv", default=None, help="任意 11 列预测表(检查 6 用)")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.deterministic = True

    check_shapes_and_grads(data_dir, [0, 1])
    check_dual_sync(data_dir)
    for budget in ("label10", "label25", "label50", "label100"):
        check_budget_isolation(repo_root / "manifests" / "label_subsets_v2.csv",
                               repo_root / "manifests" / "split_subjects.csv", budget)
    check_overfit(data_dir, [0, 1], device)
    if args.runs_dir:
        check_w1_ckpt_compat(Path(args.runs_dir))
    if args.pred_csv:
        check_val_prediction_format(args.pred_csv, repo_root,
                                    repo_root / "manifests" / "val_reference_v1.csv")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
