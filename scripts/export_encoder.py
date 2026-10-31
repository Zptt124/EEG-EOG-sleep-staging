"""export_encoder.py — 实验① A:从已训练 run 导出编码器与逐段特征(W2 v1.1)。

导出内容(在 run 文件夹下 encoder_export/):
    encoder_weights.pt   编码器(features,去分类层)state_dict + 元信息
    normalization.json   每通道标准化统计(复制自 run)
    SHA256SUMS.txt       上述两文件的 sha256
    features_<...>.npz   可选:按记录批量前向的逐段特征 + sample 索引
                         (eval 模式 + no_grad,BatchNorm 不更新)

encoder_id 默认取 run_id 加 _enc 后缀(如 W2_A_EEG_label25_sseed2024_seed0_v01_enc),
供 reports/week2/encoder_index.csv 登记。

用法:
    D:/python/python.exe scripts/export_encoder.py --run-dir <runs>/W2_A_EEG_label25_sseed2024_seed0_v01
    D:/python/python.exe scripts/export_encoder.py --run-dir ... --data-dir D:/sleepedf/processed \
        --extract-records SC4061 --out-feats val_feats.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cnn_model import SmallCNN


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_encoder(run_dir: Path, encoder_id: str | None = None) -> Path:
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "best.ckpt"
    norm_path = run_dir / "normalization.json"
    cfg_path = run_dir / "config.yaml"
    for p in (ckpt_path, norm_path, cfg_path):
        if not p.exists():
            sys.exit(f"run 不完整,缺少 {p}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = SmallCNN(in_channels=cfg["in_channels"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    out = run_dir / "encoder_export"
    out.mkdir(exist_ok=True)
    encoder_id = encoder_id or f"{cfg['run_id']}_enc"
    torch.save({
        "encoder_id": encoder_id,
        "source_run_id": cfg["run_id"],
        "model": "SmallCNN.features (classifier removed)",
        "in_channels": cfg["in_channels"],
        "channel_names": cfg["channel_names"],
        "input_shape": cfg["input_shape"],
        "output_dim": 128,
        "n_classes": cfg["n_classes"],
        "nominal_budget": cfg["nominal_budget"],
        "labeled_subjects": cfg["labeled_subjects"],
        "subset_version": cfg["subset_version"],
        "sampling_seed": cfg["sampling_seed"],
        "downstream_seed": cfg["downstream_seed"],
        "selection_rule": cfg["selection_rule"],
        "best_epoch": ckpt["epoch"],
        "val_macro_f1": ckpt["val_macro_f1"],
        "code_commit": cfg.get("code_commit"),
        "code_version": cfg.get("code_version"),
        "normalization_sha256": sha256(norm_path),
        "state_dict": model.encoder_state_dict(),
    }, out / "encoder_weights.pt")

    shutil.copy2(norm_path, out / "normalization.json")

    sums = [f"{sha256(out / 'encoder_weights.pt')}  encoder_weights.pt",
            f"{sha256(out / 'normalization.json')}  normalization.json"]
    # 显式 LF 换行,保证 sha256sum -c 在任意平台可校验
    with open(out / "SHA256SUMS.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sums) + "\n")
    print(f"exported encoder -> {out}")
    for line in sums:
        print(" ", line)
    print(f"  output_dim=128  in_channels={cfg['in_channels']}  "
          f"budget={cfg['nominal_budget']}  seed={cfg['downstream_seed']}")
    return out


def extract_features(run_dir: Path, data_dir: Path, records: list[str],
                     out_feats: Path, batch_size: int = 256):
    """逐段特征接口:加载编码器,对给定记录做批量前向,输出特征与 sample 索引。"""
    run_dir = Path(run_dir)
    enc_dir = run_dir / "encoder_export"
    ckpt = torch.load(enc_dir / "encoder_weights.pt", map_location="cpu", weights_only=True)
    with open(enc_dir / "normalization.json", encoding="utf-8") as f:
        norm = json.load(f)
    model = SmallCNN(in_channels=ckpt["in_channels"])
    sd = {k[len("features."):]: v for k, v in ckpt["state_dict"].items()}
    model.features.load_state_dict(sd, strict=True)
    model.eval()                                  # 无标签前向必须 eval,防 BN 更新
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    means = [ch["mean"] for ch in norm["channels"]]
    stds = [ch["std"] for ch in norm["channels"]]
    feats, ids, subs, recs_out, eis = [], [], [], [], []
    with torch.no_grad():
        for rec in sorted(records):
            z = np.load(data_dir / f"{rec}.npz")
            if ckpt["in_channels"] == 1:
                ch_idx = 0 if ckpt["channel_names"][0].startswith("EEG") else 1
                x = z["signal"][:, ch_idx]
            else:
                x = z["signal"]                        # 双通道:两列同一行,天然同步
            x = x.astype(np.float32)
            if x.ndim == 2:                      # 单模态 (N,3000) -> (N,1,3000)
                x = x[:, None, :]
            # 统计显式 float32,避免 numpy 提升为 float64 导致模型前向 dtype 不匹配
            x = ((x - np.array(means, dtype=np.float32)[None, :, None])
                 / np.array(stds, dtype=np.float32)[None, :, None]).astype(np.float32)
            f = []
            for b in range(0, len(x), batch_size):
                xb = torch.from_numpy(x[b:b + batch_size]).to(device)
                f.append(model.encoder(xb).cpu().numpy())
            feats.append(np.concatenate(f))
            ei = z["epoch_index"]
            ids.extend([f"{rec}_{int(i):04d}" for i in ei])
            subs.extend([rec[:5]] * len(ei))
            recs_out.extend([rec] * len(ei))
            eis.extend(ei.astype(int))
    np.savez(out_feats,
             features=np.concatenate(feats).astype(np.float32),
             sample_id=np.array(ids), subject_id=np.array(subs),
             record_id=np.array(recs_out), epoch_index=np.array(eis),
             encoder_id=ckpt["encoder_id"], source_run_id=ckpt["source_run_id"],
             output_dim=ckpt["output_dim"])
    print(f"features ({np.concatenate(feats).shape}) -> {out_feats}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--encoder-id", default=None)
    ap.add_argument("--data-dir", default=None, help="NPZ 目录(仅提取特征时需要)")
    ap.add_argument("--extract-records", default=None,
                    help="逗号分隔 record_id,或 'val'(全部 8 验证人)")
    ap.add_argument("--out-feats", default=None)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    export_encoder(run_dir, args.encoder_id)

    if args.extract_records:
        if not args.data_dir or not args.out_feats:
            sys.exit("--extract-records 需要同时给 --data-dir 与 --out-feats")
        data_dir = Path(args.data_dir)
        if args.extract_records == "val":
            split_df = pd.read_csv(Path(args.repo_root) / "manifests" / "split_subjects.csv",
                                   dtype=str)
            records = sorted(r for r in [";".join(split_df[split_df["split"] == "val"]["record_ids"])
                              .split(";")])
        else:
            records = [r.strip() for r in args.extract_records.split(",")]
        extract_features(run_dir, data_dir, records, Path(args.out_feats))


if __name__ == "__main__":
    main()
