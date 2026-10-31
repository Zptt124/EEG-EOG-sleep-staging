"""load_encoder_example.py — A 交付编码器的加载例子(B/D 照此使用)。

演示:加载编码器 -> eval 模式 -> 标准化 -> 批量前向 -> 128 维逐段特征。
实际运行(B/D 设备,NPZ 路径换成自己的数据目录):

    python scripts/load_encoder_example.py ^
        --encoder-dir <runs>/W2_A_EEG_label25_sseed2024_seed0_v01/encoder_export ^
        --data-dir <A已核验NPZ目录> --records SC4061,SC4062

输出逐段特征及其 sample_id 索引(保存为 out_feats 指定的 npz)。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cnn_model import load_encoder


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--encoder-dir", required=True, help="含 encoder_weights.pt 与 normalization.json")
    ap.add_argument("--data-dir", required=True, help="A 已核验 NPZ 目录(信号单位 µV)")
    ap.add_argument("--records", required=True, help="逗号分隔 record_id")
    ap.add_argument("--out-feats", default="features.npz")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 加载编码器(eval 模式,含标准化统计)
    model, norm = load_encoder(args.encoder_dir)
    model.to(device)
    print(f"encoder loaded: in_channels={model.in_channels}, "
          f"output_dim={model.encoder(torch.zeros(2, model.in_channels, 3000, device=device)).shape[1]}")
    print(f"normalization: {[(c['name'], c['mean'], c['std']) for c in norm['channels']]}")

    # 2) 读取原始 NPZ,按编码器通道取列并标准化(不修改原始数据)
    feats, ids = [], []
    with torch.no_grad():
        for rec in args.records.split(","):
            rec = rec.strip()
            z = np.load(Path(args.data_dir) / f"{rec}.npz")
            if model.in_channels == 1:
                ch_idx = 0 if norm["channels"][0]["name"].startswith("EEG") else 1
                x = z["signal"][:, ch_idx][:, None, :]
            else:
                x = z["signal"]                      # (N, 2, 3000),两列同一行同步
            x = torch.from_numpy(x.astype(np.float32))
            means = torch.tensor([c["mean"] for c in norm["channels"]], dtype=torch.float32)
            stds = torch.tensor([c["std"] for c in norm["channels"]], dtype=torch.float32)
            x = (x - means[None, :, None]) / stds[None, :, None]
            f = []
            for b in range(0, len(x), args.batch_size):
                f.append(model.encoder(x[b:b + args.batch_size].to(device)).cpu().numpy())
            feats.append(np.concatenate(f))
            ids.extend([f"{rec}_{int(i):04d}" for i in z["epoch_index"]])

    F = np.concatenate(feats).astype(np.float32)
    np.savez(args.out_feats, features=F, sample_id=np.array(ids))
    print(f"features {F.shape} -> {args.out_feats}  (eval mode, no BN update)")


if __name__ == "__main__":
    main()
