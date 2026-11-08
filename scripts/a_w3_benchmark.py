"""a_w3_benchmark.py — 第三周实验A:同设备同 batch 推理吞吐与资源对照(W3 v01)。

计划要求"用同一设备和 batch size 测推理吞吐,并记录训练时间、峰值显存和参数量"。
本脚本在同一会话内测:
  - SmallCNN dual(W2 权重复用)val 集推理吞吐(epoch/s,batch=128,eval+no_grad)
  - ContraWR dual(W3 权重)同口径吞吐
  - 两者峰值显存、参数量;训练时间取各自 resource.json
输出 reports/week3/inference_throughput.csv 与 stdout 表。

用法:
    D:/python/python.exe scripts/a_w3_benchmark.py ^
        --runs-dir C:/Users/34364/Desktop/睡眠/exp1/runs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cnn_model import SmallCNN, apply_normalization
from contrawr_model import ContraWRSleepNet
from train_cnn_w2 import load_records

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "week3" / "inference_throughput.csv"


def bench_forward(model, X: np.ndarray, device: str, batch_size: int,
                  transform=None):
    """eval+no_grad 全 val 集前向,返回 (epochs/s, peak MiB)。transform 负责标准化/单位转换。"""
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[b:b + batch_size]).to(device)
            if transform is not None:
                xb = transform(xb)
            model(xb)
    secs = time.time() - t0
    peak = int(torch.cuda.max_memory_allocated() / 2 ** 20)
    return round(len(X) / secs, 1), peak


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = 128

    # val 数据(与训练入口同一加载)
    split_df = pd.read_csv(REPO / "manifests" / "split_subjects.csv", dtype=str)
    rec_split = {}
    for _, r in split_df.iterrows():
        for rec in r["record_ids"].split(";"):
            rec_split[rec] = r["split"]
    val_recs = sorted(r for r, s in rec_split.items() if s == "val")
    Xva, yva, mva = load_records(Path(args.data_dir), val_recs, [0, 1])
    print(f"val epochs: {len(Xva)}  batch: {bs}  device: {device}")

    rows = []

    # ---- SmallCNN dual(W2 复用权重) ----
    cnn_dir = runs / "W2_A_EEGEOG_label100_sseed2024_seed0_v01"
    if not cnn_dir.exists():
        sys.exit(f"缺少 W2 CNN run: {cnn_dir}")
    cnn = SmallCNN(in_channels=2).to(device)
    ckpt = torch.load(cnn_dir / "best.ckpt", map_location=device, weights_only=True)
    cnn.load_state_dict(ckpt["state_dict"])
    norm = json.loads((cnn_dir / "normalization.json").read_text(encoding="utf-8"))
    cnn_params = sum(p.numel() for p in cnn.parameters())
    eps, peak = bench_forward(
        cnn, Xva, device, bs,
        transform=lambda xb: apply_normalization(xb, norm))
    rows.append({
        "model": "SmallCNN dual (W2)", "run_id": cnn_dir.name,
        "batch_size": bs, "device": device,
        "parameters": cnn_params,
        "val_epochs_per_second": eps,
        "inference_peak_memory_mib": peak,
        "train_elapsed_seconds": json.loads((cnn_dir / "resource.json").read_text(encoding="utf-8"))["elapsed_seconds"],
    })
    del cnn, ckpt
    torch.cuda.empty_cache()

    # ---- ContraWR dual(W3 权重) ----
    cw_dir = runs / "W3_A_CONTRAWR_label100_sseed2024_seed0_v01"
    if not cw_dir.exists():
        print(f"跳过 ContraWR 基准:缺少 {cw_dir}(label100 seed0 完成后再跑一次本脚本)")
        rows.append({"model": "ContraWR dual (W3)", "run_id": "W3_A_CONTRAWR_label100_sseed2024_seed0_v01",
                     "batch_size": bs, "device": device, "parameters": "",
                     "val_epochs_per_second": "", "inference_peak_memory_mib": "",
                     "train_elapsed_seconds": ""})
    else:
        cw = ContraWRSleepNet().to(device)
        ckpt = torch.load(cw_dir / "best.ckpt", map_location=device, weights_only=True)
        cw.load_state_dict(ckpt["state_dict"], strict=False)   # norm 缓冲随后从 JSON 注入
        cw_norm = json.loads((cw_dir / "normalization.json").read_text(encoding="utf-8"))
        means = torch.tensor(cw_norm["bins"]["means"], dtype=torch.float32, device=device)
        stds = torch.tensor(cw_norm["bins"]["stds"], dtype=torch.float32, device=device)
        cw.set_norm(means, stds)
        cw_params = sum(p.numel() for p in cw.parameters())
        eps, peak = bench_forward(
            cw, (Xva * 1e-6).astype(np.float32), device, bs)
        rows.append({
            "model": "ContraWR dual (W3)", "run_id": cw_dir.name,
            "batch_size": bs, "device": device,
            "parameters": cw_params,
            "val_epochs_per_second": eps,
            "inference_peak_memory_mib": peak,
            "train_elapsed_seconds": json.loads((cw_dir / "resource.json").read_text(encoding="utf-8"))["elapsed_seconds"],
        })
        del cw, ckpt

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")
    for r in rows:
        print(" | ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
