# 环境

实验④用了两个环境：

| 环境 | 系统 | 用途 | 依赖记录 |
|---|---|---|---|
| `eegdm`（conda，Python 3.11） | Ubuntu 24.04（WSL2） | 特征提取、线性头训练、波形图；入库后的全部分析脚本也在这里运行 | `requirements_eegdm_wsl.txt` |
| `exp4_env`（venv） | Windows | 第一周本地分析（对齐、融合、翻转、混淆矩阵、官方评估、按人统计、挑片段） | `requirements_exp4_windows.txt` |

`eegdm` 环境按 C 的说明配置（PyTorch 2.7.1 + CUDA 12.6，见 `experiment3/environment/`）。两份依赖清单都是用 `pip freeze` 导出的实际版本。

只复现分析（不训练）只需要：`numpy`、`pandas`、`scikit-learn`、`matplotlib`（画图时）。
