"""cnn_model.py — 实验① A 公平监督基线:唯一 SmallCNN 实现(W2 v1.1)。

单模态输入 [B,1,3000](通道 0=EEG Fpz-Cz,通道 1=EOG horizontal,与 A 第一周
预处理 NPZ 的 signal[:, [0,1], :] 一致),双通道输入 [B,2,3000](通道顺序
[EEG, EOG])。双通道只把第一层输入通道数从 1 改为 2,层数、卷积结构、上下文
均不增加(第二周工作文档 三.实验①)。

结构沿用第一周团队文档参考结构:
    block(1,32,50,6) -> block(32,64,8,1) -> block(64,128,8,1) -> GAP -> Linear(128,5)

编码器 = features(去掉分类层),逐段输出 128 维(以实际代码检查为准,
见 docs/week2/cnn_interface.md)。两单模态编码器拼接(256 维)与双通道编码器
(128 维)是不同表征,不能混称。

用法:
    model = SmallCNN(in_channels=1)          # EEG 或 EOG 单模态
    model = SmallCNN(in_channels=2)          # EEG+EOG 双通道
    feats = model.encoder(x)                 # (B, 128),训练/推理均可用

编码器导出/加载(权重与标准化分离保存,标准化统计只拟合本预算有标签者):
    export_encoder(run_dir, out_dir)         # 见 scripts/export_encoder.py
    model, norm = load_encoder(enc_dir)      # 返回 eval 模式模型与标准化字典
加载例子见 scripts/load_encoder_example.py。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

N_CLASSES = 5                     # 0=W 1=N1 2=N2 3=N3 4=REM
ENCODER_DIM = 128                 # GAP 前的通道数,即编码器输出维度
EPOCH_PTS = 3000                  # 30 s x 100 Hz
CHANNEL_NAMES = ["EEG Fpz-Cz", "EOG horizontal"]
CHANNEL_UNITS = "uV"              # A 第一周预处理后的信号单位


class SmallCNN(nn.Module):
    """团队文档参考结构;双通道只改首层输入通道数。"""

    def __init__(self, in_channels: int = 1, n_classes: int = N_CLASSES):
        super().__init__()
        if in_channels not in (1, 2):
            raise ValueError(f"in_channels 必须为 1(单模态)或 2(双通道),got {in_channels}")

        def block(cin, cout, k, s):
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, stride=s, padding=k // 2),
                nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(4))

        # 命名与第一周 05_train_cnn.py 完全一致,保证旧权重可直接 load_state_dict
        self.features = nn.Sequential(
            block(in_channels, 32, 50, 6),   # 大卷积核,抓低频慢波
            block(32, 64, 8, 1),
            block(64, 128, 8, 1),
            nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(ENCODER_DIM, n_classes)
        self.in_channels = in_channels

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """去掉分类层的逐段特征接口,x: (B, C, 3000) -> (B, 128)。"""
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

    def encoder_state_dict(self) -> dict:
        """仅编码器(features)部分的 state_dict,供导出与交接。"""
        return {k: v.detach().cpu().clone()
                for k, v in self.state_dict().items() if k.startswith("features.")}


def load_encoder(enc_dir) -> tuple[nn.Module, dict]:
    """加载已导出编码器。返回 (model, norm);model 已置 eval 模式。

    enc_dir 需包含:
        encoder_weights.pt   torch.save 的编码器权重及元信息
        normalization.json   每通道标准化统计(拟合对象记录在内)
    权重只含 features(去分类层),分类头不恢复——请用 model.encoder(x)
    取特征,不要用 model(x) 当分类器。
    """
    enc_dir = Path(enc_dir)
    ckpt = torch.load(enc_dir / "encoder_weights.pt", map_location="cpu", weights_only=True)
    model = SmallCNN(in_channels=ckpt["in_channels"])
    # state_dict 键带 "features." 前缀,去前缀后严格加载到 features 子模块
    sd = {k[len("features."):]: v for k, v in ckpt["state_dict"].items()}
    model.features.load_state_dict(sd, strict=True)
    model.eval()
    with open(enc_dir / "normalization.json", encoding="utf-8") as f:
        norm = json.load(f)
    return model, norm


def apply_normalization(x, norm: dict):
    """按 normalization.json 做每通道 z-score。

    x 可为 torch.Tensor 或 numpy 数组,形状 (..., C, 3000);返回同类型:
    Tensor 返回 Tensor(原 dtype/device),数组返回 float32 numpy 数组。
    """
    was_tensor = torch.is_tensor(x)
    x = torch.as_tensor(x, dtype=torch.float32)
    means = [ch["mean"] for ch in norm["channels"]]
    stds = [ch["std"] for ch in norm["channels"]]
    mean_t = torch.as_tensor(means, dtype=torch.float32, device=x.device).view(1, -1, 1)
    std_t = torch.as_tensor(stds, dtype=torch.float32, device=x.device).view(1, -1, 1)
    out = (x - mean_t) / std_t
    return out if was_tensor else out.numpy()


def build_norm_json(means: list[float], stds: list[float], fit_on: dict,
                    in_channels: int, version: str = "v01") -> dict:
    """生成标准化参数文件内容。fit_on 须写明预算、有标签者及片段数。"""
    if len(means) != in_channels or len(stds) != in_channels:
        raise ValueError(f"means/stds 数量({len(means)})必须等于通道数({in_channels})")
    return {
        "method": "per-channel z-score (scalar per channel, fit on labeled subjects only)",
        "input_shape": [in_channels, EPOCH_PTS],
        "sampling_rate_hz": 100,
        "channel_units": CHANNEL_UNITS,
        "channels": [
            {"index": i, "name": CHANNEL_NAMES[i], "mean": float(means[i]), "std": float(stds[i])}
            for i in range(in_channels)
        ],
        "fit_on": fit_on,
        "version": version,
    }
