# 实验④：跨模态与融合（EOG 路线）

D 负责。第一周已完成：CNN 层面的概率融合、冻结 EEGDM 处理 EOG、EEGDM 层面的特征拼接融合、全组 N1 错误对比、按受试者分析和典型片段，并写出 6 条带证据强度的假设。

全部结果来自验证集（8 人，21,186 段），**测试集未使用**；指标用 B 的官方 `evaluate.py` v1 计算，按受试者 bootstrap 1000 次。

## 结果入口

| 验证集 Macro-F1 | 只用 EEG | 只用 EOG | 概率平均 | **特征拼接** |
|---|---:|---:|---:|---:|
| EEGDM，100% 标签（种子 0–4） | 0.746 ± 0.004 | 0.692 ± 0.006 | 0.762 ± 0.003 | **0.774 ± 0.002** |
| EEGDM，25% 标签（3 份名单 × 种子 0–4） | 0.709 ± 0.011 | 0.660 ± 0.007 | 0.733 ± 0.010 | **0.745 ± 0.004** |
| CNN，100% 标签（A，单次） | 0.765 | 0.683 | 0.756 | — |
| CNN，25% 标签（B，名单 2024，单次） | 0.717 | 0.656 | 0.734 | — |

- **EOG 拼接让 EEGDM 稳定变好**，标签越少越明显，主要补 REM。官方配对差：100% +2.7 [+1.0, +4.0]，25% +4.8 [+2.5, +7.9]。
- **25% 标签 + EOG 与 100% 标签只用 EEG 无法区分**：−0.5 [−2.5, +1.5]。
- 都用上 EOG 后，**EEGDM 是否比 CNN 更省标签，证据不足**：25% 下 +1.05 [−1.7, +4.1]。
- 冻结 EEGDM 能处理 EOG；条件编号 10 / 14 / 18 没有差别。
- 概率平均在某人 EOG 质量差时会被拖垮（SC462），拼接更稳（个案）。

详细论证、证据强度和下一步见 [reports/hypotheses.md](reports/hypotheses.md)。全部模型的运行编号与分数见 [reports/run_index.csv](reports/run_index.csv)。

## 文件结构

```text
experiment4/
├── README.md
├── configs/fusion_config.yaml     三种融合方法的输入、参数、种子、代码
├── environment/                   两个环境的说明与依赖版本
├── manifests/
│   ├── eog_cond14_v01/            EOG 特征的提取设置、逐段索引、资源记录、汇总
│   └── external_artifacts.md      不入库的大文件（特征 23.67 GB 等）在哪、怎么重新得到
├── scripts/                       全部代码（见下表）
└── reports/
    ├── hypotheses.md              6 条假设与证据
    ├── run_index.csv              15 个模型：负责人、运行编号、预测表位置、官方分数
    ├── paired_predictions_val.csv 文档要求的交付文件：A 的 EEG / EOG 预测逐段对齐
    ├── fusion_changes.csv         文档要求：翻转统计（以 EEG、EOG 为参照，写明分母）
    ├── by_subject_stage.csv       文档要求：按受试者 × 阶段的翻转
    ├── n1_confusion_all_models.csv 文档要求：全组 14 个模型的 N1 去向和误报来源
    ├── models_summary.csv、confusion/  总表、每个模型的 5×5 混淆矩阵
    ├── official/                  官方指标（含 95% 区间）、11 组配对比较
    ├── subject/                   12 个模型 × 8 人的指标与翻转
    ├── cases/                     4 类 20 个典型片段与 EEG / EOG 波形图
    ├── predictions_E4/            D 自己的 9 份逐段预测
    └── runs_E4/                   7 次运行的小文件记录
```

文档要求的 6 个交付文件：`configs/fusion_config.yaml`、`reports/paired_predictions_val.csv`、`reports/fusion_changes.csv`、`reports/by_subject_stage.csv`、`reports/n1_confusion_all_models.csv`、`reports/hypotheses.md`。

## 代码

| 脚本 | 作用 | 在哪运行 |
|---|---|---|
| `prepare_inputs.py` | 把 A、B、C 的预测表和 B 的 `evaluate.py` 从各自的提交里取出来，连同 D 的预测表一起放进 `real_data/`，核对格式并记录来源 commit 与 SHA256 | Ubuntu |
| `align.py` → `fuse.py` → `flips.py` | A 的两个 CNN：按 `sample_id` 对齐 → 概率平均 → 翻转统计（总体、按人 × 阶段） | 任意 |
| `fuse_avg.py` | 通用概率平均：任意两份预测表 → 一份标准 11 列预测表 | 任意 |
| `confusion.py` | 14 个模型的混淆矩阵、各阶段 F1、N1 去向 | 任意 |
| `official_eval_all.py` | 用官方 `evaluate.py` 重算 14 个模型并做配对比较；同时核对与 `confusion.py` 一致 | 任意 |
| `subject_breakdown.py` | 按受试者的指标和翻转 | 任意 |
| `pick_cases.py` | 挑 4 类典型片段 | 任意 |
| `plot_cases.py` | 读 A 的 npz 原始信号，画典型片段波形 | Ubuntu（需要 A 的预处理数据） |
| `check_features.py` | 检查一份 EEGDM 特征文件的形状、NaN/Inf、全零样本 | Ubuntu |
| `train_linear_concat.py` | EEG + EOG 特征拼接后训练线性头（单个种子） | Ubuntu（需要特征） |
| `multiseed_heads.py` | 多种子 × {EEG, EOG, 拼接, 平均}，100% 标签 | Ubuntu（需要特征） |
| `multiseed_budget.py` | 同上，按 B 的少标签名单训练 | Ubuntu（需要特征） |
| `compare_with_reports.py` | 把重新跑出的结果和 `reports/` 里提交的逐格比较 | 任意 |

所有脚本出错时都会打印原因并以非零状态退出，不会留下半成品。分析脚本读写 `experiment4/real_data/`（已被忽略，不入库）。

## 只复现分析（不需要特征和 GPU）

在仓库根目录，先 `git fetch origin` 拿到 B、C 的分支，然后：

```bash
python experiment4/scripts/prepare_inputs.py
python experiment4/scripts/align.py
python experiment4/scripts/fuse.py
python experiment4/scripts/flips.py
python experiment4/scripts/confusion.py
python experiment4/scripts/official_eval_all.py
python experiment4/scripts/subject_breakdown.py
python experiment4/scripts/pick_cases.py
python experiment4/scripts/compare_with_reports.py
```

最后一步应输出 `DONE: all regenerated results match reports/`。B、C 的分支合并到 main 后，`prepare_inputs.py` 会直接从工作区读取，不再需要分支。

**一处已知差异**：`reports/cases/typical_cases.csv` 是第一周用旧版 `pick_cases.py` 挑的，当时的排序在分数相同时顺序不固定。B 类（25% 下 EOG 救回的 REM）有很多候选的把握都是 1.0，入库版本改用稳定排序后，B 类会挑出另一组同样符合条件的片段，其余三类完全一致。`compare_with_reports.py` 会把这一项标为 `KNOWN`。波形图和假设 6 的描述对应的是入库的这份表。

## 与第一周检查项的对照

| WEEK1 检查项 | 状态 |
|---|---|
| 用假预测验证 sample_id 对齐、概率平均、翻转和分组统计 | **未用假数据做**：A 的真实预测到得早，直接用了真实数据。替代证据：`align.py`、`fuse_avg.py` 逐段核对 `sample_id`、真实标签和每行概率和；官方 `evaluate.py` 重算 14 个模型，与自写代码结果完全一致 |
| 相对 EEG / EOG 分别统计纠正与新增错误及分母 | 完成：`fusion_changes.csv`、`by_subject_stage.csv`、`subject/per_subject_flips.csv` |
| 探索 EOG 是否可用；保存输入设置及失败记录 | 完成：条件 10 / 14 / 18 全量提取，各 5 个种子；设置在 `manifests/eog_cond14_v01/` |
| 拼接 EEG / EOG 特征训练线性分类器；与相同条件的 EEG-only 比较 | 完成：100% 和 25%，多种子 |
| 汇总全组 N1 错误，按阶段 / 受试者分析，写 1–2 个待验证假设 | 完成：写了 6 条，标明证据强度 |
| C 受阻时先完成 CNN 融合 | 不适用（C 未受阻） |

## 局限与下一步

- CNN 只有单次训练、只有晚融合；与 EEGDM 拼接（多种子、早融合）不对等。需要 CNN 多种子和 EEG+EOG 双通道 CNN 后，才能比较 EEGDM 与 CNN 谁更省标签。
- 只有 8 个验证者；按人的现象（SC462）属于个案。
- 下一步候选：50% 标签档、可学习权重的晚融合、EOG 客观质量指标、多种子的按人统计。

## 交接

- 输入：A `E1_{EEG,EOG}_label100_seed0_v01`；B `E2_{EEG,EOG}_label25_sseed2024_seed0_v01`、`evaluate.py` v1、`label_subsets_v1*.csv`；C `E3_EEG_label100_seed0_provisional_v01`、`E3_EEG_label25_sseed2024_seed0_v01`、EEG 特征 `eeg_cond14_v01`。具体 commit 由 `prepare_inputs.py` 记在 `real_data/_upstream/provenance.csv`。
- 输出：本文件夹全部内容；大文件见 `manifests/external_artifacts.md`。
