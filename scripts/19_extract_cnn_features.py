"""Step W2-B11: 用 A 的 25% 双通道编码器提取逐段特征,供 13_context_head.py --repr cnn 使用。

每个下游 seed 用同 seed 的编码器;范围是 window_index.csv 的全部中心样本(14 名有标签者 + 8 名验证者),
其余 40 名训练者不进入。eval + no_grad,标准化用编码器随附的 normalization.json。

提取前逐项校验:权重与标准化的 SHA256 须与 reports/week2/encoder_index.csv 一致,否则报错。
提取后用 best.ckpt 的分类层接这些特征,在验证集上还原 A 的 predictions_val.csv:
概率最大偏差须不超过 PROB_TOL;预测类别不同的段只允许出现在 A 的前两类概率差小于 TIE_MARGIN 的近似打平段,
并如实计入 extraction_check.csv。任何其他不一致都报错,说明特征提取与 A 的前向不一致。

输出:<out-dir>/cnn_label25_seed<k>.npz(sample_id[str]、x[N,128],float32)与 extraction_check.csv。

用法:
  python scripts/19_extract_cnn_features.py --runs-dir features/a_w2_v01/extracted --out-dir features/a_w2_v01/cnn_label25
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cnn_model import load_encoder
from common import BASE, PROCESSED
import w2_common as w

PROB_TOL = 5e-3     # CPU 与 A 的 GPU 前向存在浮点差异
TIE_MARGIN = 1e-3   # 前两类概率差小于它才算近似打平
ENCODER_INDEX = BASE / "reports" / "week2" / "encoder_index.csv"
WINDOW_INDEX = BASE / "reports" / "week2" / "window_index.csv"


def records_in_order(window_index):
    rows = w.read_csv_rows(window_index)
    return list(dict.fromkeys(r["record_id"] for r in rows)), {r["center_sample_id"] for r in rows}


def check_sha(run_id, enc_dir, index):
    row = index[run_id]
    for name, key in (("encoder_weights.pt", "weights_sha256"), ("normalization.json", "normalization_sha256")):
        if w.sha256_file(enc_dir / name) != row[key]:
            raise ValueError(f"{run_id}/{name} 的 SHA256 与 encoder_index.csv 不一致")


@torch.no_grad()
def extract(model, norm, records, processed, batch_size=512):
    mean = torch.tensor([c["mean"] for c in norm["channels"]], dtype=torch.float32)[None, :, None]
    std = torch.tensor([c["std"] for c in norm["channels"]], dtype=torch.float32)[None, :, None]
    feats, ids = [], []
    for rec in records:
        with np.load(Path(processed) / f"{rec}.npz", allow_pickle=False) as d:
            x, epochs = torch.from_numpy(d["signal"].astype(np.float32)), d["epoch_index"]
        x = (x - mean) / std
        feats += [model.encoder(x[b:b + batch_size]).numpy() for b in range(0, len(x), batch_size)]
        ids += [f"{rec}_{int(i):04d}" for i in epochs]
    return np.array(ids), np.concatenate(feats).astype(np.float32)


@torch.no_grad()
def check_against_predictions(run_dir, ids, x):
    """分类层接提取的特征,与 A 的验证预测逐段比对。"""
    ckpt = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=True)
    weight, bias = ckpt["state_dict"]["classifier.weight"], ckpt["state_dict"]["classifier.bias"]
    pred = pd.read_csv(run_dir / "predictions_val.csv", dtype={"sample_id": str})
    pos = {s: i for i, s in enumerate(ids)}
    feats = torch.from_numpy(x[[pos[s] for s in pred["sample_id"]]])
    probs = torch.softmax(feats @ weight.T + bias, dim=1).numpy()
    theirs = pred[["p_W", "p_N1", "p_N2", "p_N3", "p_REM"]].to_numpy(dtype=np.float64)
    differ = probs.argmax(1) != pred["y_pred"].to_numpy()
    top2 = np.sort(theirs, axis=1)[:, -2:]
    real_diff = int((differ & (top2[:, 1] - top2[:, 0] >= TIE_MARGIN)).sum())
    max_dev = float(np.abs(probs - theirs).max())
    if real_diff or max_dev > PROB_TOL:
        raise ValueError(f"{run_dir.name}: 无法还原 A 的预测(非打平段类别不同 {real_diff},概率最大偏差 {max_dev:.2e})")
    return len(pred), int(differ.sum()), max_dev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs-dir", required=True, help="解压后的 w2_a_runs_v01 目录")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--processed", default=str(PROCESSED))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)

    index = {r["source_run_id"]: r for r in w.read_csv_rows(ENCODER_INDEX)}
    records, centers = records_in_order(WINDOW_INDEX)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    for seed in args.seeds:
        run_id = f"W2_A_EEGEOG_label25_sseed2024_seed{seed}_v01"
        run_dir = Path(args.runs_dir) / run_id
        enc_dir = run_dir / "encoder_export"
        check_sha(run_id, enc_dir, index)
        model, norm = load_encoder(enc_dir)
        if model.in_channels != 2 or [c["name"] for c in norm["channels"]] != ["EEG Fpz-Cz", "EOG horizontal"]:
            raise ValueError(f"{run_id}: 不是 [EEG, EOG] 双通道编码器")
        ids, x = extract(model, norm, records, args.processed)
        if set(ids) != centers or len(ids) != len(centers) or not np.isfinite(x).all():
            raise ValueError(f"{run_id}: 提取的样本与 window_index.csv 不一致或含非有限值")
        n, diff, max_dev = check_against_predictions(run_dir, ids, x)
        path = out / f"cnn_label25_seed{seed}.npz"
        np.savez(path, sample_id=ids.astype(str), x=x)
        checks.append(dict(run_id=run_id, n_samples=len(ids), feature_dim=x.shape[1], val_epochs_checked=n,
                           near_tie_pred_mismatch=diff, max_prob_dev=f"{max_dev:.2e}", feature_sha256=w.sha256_file(path)))
        print(f"seed{seed}: {x.shape} 验证 {n} 段,打平段类别不同 {diff},概率最大偏差 {max_dev:.2e}")
    with (out / "extraction_check.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(checks[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(checks)


if __name__ == "__main__":
    main()
