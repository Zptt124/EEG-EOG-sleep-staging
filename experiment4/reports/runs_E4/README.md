# 实验④（D）运行记录索引

每个子文件夹是一次运行的**小文件副本**：单次运行有逐轮记录（`history.json`）、指标、汇总，用 C 的脚本训练的还有配置（`run_config.json`）和训练受试者名单；多种子运行有每一次的结果（`summary_all_runs.csv`）和均值 ± 标准差（`summary_mean_std.csv`）。完整运行（含分类头权重 `*.pt` 和逐段预测）保存在 D 本地的 `C_experiment3/runs/`，不入库；正式使用的那几份逐段预测在 `../predictions_E4/`。

所有运行都在 Ubuntu（WSL2）的 `eegdm` 环境里完成，CPU 训练分类头；只用训练集和验证集，测试集未使用。

| 运行 | 做了什么 | 代码 | 关键结果（验证集 Macro-F1） |
|---|---|---|---|
| `D_EOG_label100_seed0_cond14_v01` | EOG 特征（条件 14）+ 线性头，100% 标签，种子 0 | C 的 `train_linear100.py` | 0.6835 |
| `D_EOG_label100_seed0_cond10_v01` | 同上，条件 10 | 同上 | 单次；与 14 的差距后被多种子证明是早停偶然 |
| `D_EEGEOG_label100_seed0_cond14_v01` | EEG + EOG 特征拼接（5120 维）+ 线性头，种子 0 | `scripts/train_linear_concat.py` | 0.7766（最佳第 20 轮） |
| `D_multiseed_cond14_v01` | 种子 0–4 × {EEG, EOG, 拼接, 概率平均}，100% 标签 | `scripts/multiseed_heads.py` | EEG 0.7458±0.0037；EOG 0.6921±0.0063；平均 0.7615±0.0030；拼接 0.7740±0.0019 |
| `D_multiseed_eogcond10_v01` | 同上，EOG 用条件 10 | 同上 | EOG 0.696±0.007；拼接 0.774±0.003 |
| `D_multiseed_eogcond18_v01` | 同上，EOG 用条件 18 | 同上 | EOG 0.692±0.007；拼接 0.775±0.002 |
| `D_multiseed_label25_cond14_v01` | B 的 3 份 25% 名单 × 种子 0–4 × {EEG, EOG, 拼接, 平均} | `scripts/multiseed_budget.py` | EEG 0.7089±0.0111；EOG 0.6595±0.0072；平均 0.7331±0.0100；拼接 0.7452±0.0044 |

## 特征提取（不是训练运行，但结果都依赖它）

| 特征 | 做了什么 | 状态 |
|---|---|---|
| `eog_cond14_pilot` | EOG 试跑，SC4001 一晚，检查数值（`scripts/check_features.py`） | 通过：无 NaN/Inf、无全零样本 |
| `eog_cond14_v01` | 全量 EOG，条件 14，122 晚 / 154,124 段，约 23.67 GB | 完成；元数据在 `../../manifests/eog_cond14_v01/` |
| `eog_cond10_v01`、`eog_cond18_v01` | 全量 EOG，条件 10 / 18 | 完成；三条件无差别后特征已删除，元数据保留在 D 本地 |

提取用的是 C 的 `experiment3/scripts/extract_features.py`（数据下标 1 = EOG，其余设置与 C 的 EEG 提取相同：200 Hz、µV/100、20 层、1 秒窗口、每段 30 个窗口）。

## 复现

特征准备好之后（EEG 特征来自 C，EOG 特征按上面重新提取），在仓库根目录：

```bash
python experiment4/scripts/multiseed_heads.py --eeg <EEG特征文件夹> --eog <EOG特征文件夹> --output <新输出文件夹> --seeds 0 1 2 3 4
python experiment4/scripts/multiseed_budget.py --eeg <EEG特征文件夹> --eog <EOG特征文件夹> --output <新输出文件夹> \
    --subsets manifests/label_subsets_v1.csv manifests/label_subsets_v1-s2025.csv manifests/label_subsets_v1-s2026.csv --seeds 0 1 2 3 4
```

名单文件在 B 的分支 `feat/b-evaluation` 的 `manifests/` 下。输出文件夹里已经有结果时脚本会拒绝运行，防止覆盖。
