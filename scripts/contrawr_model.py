"""contrawr_model.py — 实验A 第三周额外基线:ContraWR 监督架构移植(W3 v01)。

架构来源(按团队规则核查许可并保留来源):
  - 论文:Chaoqi Yang, Danica Xiao, M. Brandon Westover, Jimeng Sun,
    "Self-Supervised Electroencephalogram Representation Learning for Automatic
    Sleep Staging: Model Development and Evaluation Study",
    JMIR AI 2023;2(1):e46769(arXiv:2110.15278)。
  - 上游仓库:https://github.com/ycq091044/ContraWR
    固定 commit 217753a5711e1d753d7295b4eb9f35701f2739c8(2023-06-02)。
  - 上游仓库无 LICENSE 文件(已核查,见 reports/week3/a_baseline_selection.md),
    因此本文件是按其 CNNEncoder2D_SLEEP 架构的**重实现**(非逐字复制),
    结构与维度严格对应上游:conv1(4->6) + ResBlock(6->8,stride2)
    + ResBlock(8->16,stride2+pool) + ResBlock(16->32,stride2+pool)
    -> flatten(128) -> Linear(128,32) -> Linear(32,5)。
    上游 ResBlock 在残差+池化后固定 Dropout(0.5),训练时生效,这里原样保留。

输入:双通道原始信号 [B, 2, 3000](100 Hz,30 s)。上游 Sleep-EDF 用
Fpz-Cz + Pz-Oz 双 EEG;本组已核验 NPZ 为 [EEG Fpz-Cz, EOG horizontal],
按第三周"同样的分期输入"要求以组内 dual 代入(形状不变,模型无改动),
该模态差异记录于 reports/week3/a_baseline_selection.md。V 单位(上游经 mne
读取为 V;本仓库 NPZ 为 µV,训练脚本内 ×1e-6 转回 V 再喂入)。前向内部做:
    torch.stft(n_fft=256, hop_length=64, center=False, onesided=True)
    -> log(|real|+1e-8) / log(|imag|+1e-8) 按通道拼接 -> [B, 4, 129, 47]
    -> 每 (通道分量, 频率 bin) 标量 z-score(统计只拟合本预算有标签者,
       是协议"每通道标量 z-score"在谱图表征上的对应扩展,记录于 config/normalization)
    -> 2D CNN -> 5 类 logits。

本模型只用于监督比较(自监督预训练不跑),sup 头即上游 sup 分支;
上游 fc/byol 分支未移植。
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 5                     # 0=W 1=N1 2=N2 3=N3 4=REM
N_FFT = 256                       # 上游 torch.stft 固定参数
HOP = 256 // 4                    # = 64
STFT_FREQ = N_FFT // 2 + 1        # 129
STFT_TIME = 43                    # floor((3000-256)/64)+1(center=False 帧数)
SPECTROGRAM_SHAPE = (4, STFT_FREQ, STFT_TIME)   # [EEG_real, EOG_real, EEG_imag, EOG_imag]
LOG_EPS = 1e-8                    # 上游 log(|x|+1e-8)


class ResBlock(nn.Module):
    """上游 ResBlock 重实现:conv3x3+BN+ELU -> conv3x3+BN -> 残差(stride2 时
    downsample)-> 可选 MaxPool2d(2) -> Dropout(0.5)。上游所有块均 downsample=True。
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels))
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)      # 上游固定值,训练时生效

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class ContraWRSleepNet(nn.Module):
    """ContraWR(监督分支)移植:输入 [B,2,3000] V 单位 -> 5 类 logits。

    norm_means/norm_stds:(4,129) float32 缓冲,谱图每 (分量,频率 bin) 标量
    z-score 统计;None 表示不做 z-score(仅用于统计拟合阶段)。
    """

    def __init__(self, norm_means: torch.Tensor | None = None,
                 norm_stds: torch.Tensor | None = None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True))
        self.conv2 = ResBlock(6, 8, 2, True, False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 32, 2, True, True)
        self.sup = nn.Sequential(
            nn.Linear(128, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, N_CLASSES, bias=True))
        if norm_means is not None and norm_stds is not None:
            self.register_buffer("norm_means", norm_means.view(1, 4, 129, 1).float())
            self.register_buffer("norm_stds", norm_stds.view(1, 4, 129, 1).float())
        # 否则不注册缓冲;由 set_norm() 惰性注册(避免普通属性占用注册名)

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,2,3000) V 单位 -> (B,4,129,43) log|real|/log|imag| 谱图(上游 torch_stft)。

        矩形窗(显式传 window=torch.ones)与上游默认一致;return_complex=False
        在新 torch 已弃用,改用 view_as_real 得到相同的 (...,2) 实/虚张量。
        """
        window = torch.ones(N_FFT, device=x.device, dtype=x.dtype)
        parts = []
        for s in range(x.shape[1]):
            spec = torch.stft(x[:, s, :], n_fft=N_FFT, hop_length=HOP,
                              center=False, onesided=True, window=window,
                              return_complex=True)
            parts.append(torch.view_as_real(spec))  # (B,129,43,2)
        real = torch.stack([p[..., 0] for p in parts], dim=1)   # (B,C,129,43)
        imag = torch.stack([p[..., 1] for p in parts], dim=1)
        assert tuple(real.shape[1:]) == (2, STFT_FREQ, STFT_TIME), real.shape
        return torch.cat([torch.log(torch.abs(real) + LOG_EPS),
                          torch.log(torch.abs(imag) + LOG_EPS)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectrogram(x)
        if hasattr(self, "norm_means"):
            x = (x - self.norm_means) / self.norm_stds
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.shape[0], -1)
        assert x.shape[1] == 128, f"展平维度 {x.shape[1]} != 128(上游 sup 头输入)"
        return self.sup(x)

    def set_norm(self, means: torch.Tensor, stds: torch.Tensor):
        """训练脚本拟合谱图统计后注入;与构造参数等价,便于确定性流程。"""
        self.register_buffer("norm_means", means.view(1, 4, STFT_FREQ, 1).float())
        self.register_buffer("norm_stds", stds.view(1, 4, STFT_FREQ, 1).float())
