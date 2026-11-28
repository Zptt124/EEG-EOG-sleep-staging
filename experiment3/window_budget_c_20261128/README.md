# 实验 C｜不依赖真实分期的窗口

Sleep-EDF 开发复验完成 26 份可配对名单、130 次配对抽块和 390 次新窗口拟合。关灯窗口下，「新人新夜 − 原人原夜」的点估计为正，两种预算口径的 95% 配对区间均跨零。

| 分析 | 名单 / 配对抽块 | 旧窗口：新人减原夜 | 关灯窗口：新人减原夜 | 关灯窗口减旧窗口的差之差 |
|---|---:|---:|---:|---:|
| 每组实际追加 600 个五类标签 | 19 / 43 | +1.78 [+0.08, +3.49] | +0.24 [−1.40, +1.78] | −1.54 [−3.20, +0.03] |
| 每组尝试追加 600 段 | 26 / 130 | +1.64 [+0.50, +2.83] | +0.55 [−0.67, +1.79] | −1.10 [−2.10, −0.12] |

单位为五类 Macro-F1 的百分点，方括号为 95% 配对区间。严格等预算分析仅保留三组都取得足额标签的配对。固定尝试量分析纳入所有可配对抽块，各组实际取得的标签数可以不同。四份原名单因原人原夜组在指定时段无可用块而退出窗口比较。

## 窗口与预算

已经标注的起点沿用原开发复验的 5 人、1,200 段。对比两个窗口下三种追加方式：原人原夜、原人另一夜、新人新夜。每组尝试标注 10 个连续 30 分钟块，每块 60 段。新候选池仅用 PSG 记录起点及长度和 LightsOff 元数据，候选锁定后才读取新选片段的分期。

- 从关灯后第一个完整的 30 秒 epoch 起，将接下来 9 小时分为三个 3 小时时段，各时段划分为连续、不重叠的完整 30 分钟块。去掉超出记录或窗口的块。
- 两种窗口沿用相同人员、原夜/另一夜/新人记录、追加时段配额和开发评估片段。以时间索引剔除和起点重叠的候选块。任一组、记录、时段无可用块时，整份名单退出对照，并在清单中保留原因。
- 每名单做五次三组配对抽块。旧窗口读取同一名单和次序的原归档预测，并按原评估索引重新核对混淆计数。新窗口复用 W4 30k EMA 的 384 维冻结特征及原线性头，固定 30 轮、种子 0。
- 主分析要求三组每次都实际取得 600 个五类标签。另报告固定尝试 600 段的全部抽块，无法评分的段屏蔽，不替换选块。各名单等权，通过同一次抽样重抽名单、名单内抽块和两个开发评估队列内的受试者，计算 95% 配对区间。
- 主要对照为“新人新夜减原人原夜”的五类 Macro-F1 差值，以及新窗口减旧窗口的差之差，单位为百分点。结论适用于沿用已标注起点的追加阶段。开发复验原有的两个 8 人评估队列参与评分，最终封存 16 人不进入流程。

## 候选池接口

[窗口规则](results/protocol.json) 保存规则和文件散列，[可用名单](results/paired_rosters.csv) 给出原名单号和退出原因。[候选块](results/candidate_blocks.csv) 的坐标为 `record_id`、`first_epoch`、`last_epoch`，边界含首尾，另有名单、受试者、分配方式和时段字段。使用 `paired_roster=1` 的候选及[共同起点](results/anchor_blocks.csv)。[冻结选块](results/selected_blocks.csv) 给出每名单五次追加抽块。

## 复算

从仓库根目录运行下列命令，使用 Python 3.11、NumPy、PyTorch 和 scikit-learn。复算校验锁定文件、历史混淆计数、逐次得分和两种口径的配对区间。完整输出见[汇总](results/summary.json)、[逐拟合评分](results/scores.csv)、[逐次配对差](results/paired_deltas.csv)和[混淆计数](results/paired_confusions.npz)。

```bash
python experiment3/window_budget_c_20261128/verify_results.py
```

## 重新提取和拟合

1. 安装 Python 3.11、NumPy、CUDA 版 PyTorch、MNE、SciPy、scikit-learn、einops、xlrd 及 W4 模型依赖。将 [D 的全量冻结缓存](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w6-shared-cache-v01-20261126) 解压到本地，并设置 `BUDGET_C_BUNDLE` 指向 `full` 目录。
2. 先按 [Sleep-EDF 开发复验](../week3_budget_replication_20261128/README.md#结果与复算) 重新拟合原窗口，输出到 `output/budget_replication_20261128`，或用 `BUDGET_C_OLD` 指向已有的完整运行目录。旧窗口拟合需要逐名单预测与原评估索引。
3. 设置 `BUDGET_C_CHECKPOINT` 指向 [D 的 W4 30k 发布](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-pretrain-w4-main30k-v01-20261117)内的 `ckpt_step30000.pt`，设置 `BUDGET_C_W4_SOURCE` 和 `EEGDM_VENDOR` 分别指向 [W4 提特征源码](https://github.com/Zptt124/EEG-EOG-sleep-staging/tree/6590ec90a9fcc4ea6e1d7118556191f618117810/experiment4/week4/src)和[官方 EEGDM 源码](https://github.com/jhpuah/EEGDM/tree/003b5a2a016b2f02788ea5693d8b7cd00b6bd969)。
4. 从仓库根目录运行下列命令。发布版调整了输入目录定位，原运行脚本散列与日志在[归档目录](results/)。

```bash
source_dir=output/budget_unlabeled_window_c_20261128/source
mkdir -p "$source_dir"
for file in SC-subjects.xls SHA256SUMS.txt; do
  curl -fsSL "https://physionet.org/files/sleep-edfx/1.0.0/$file" -o "$source_dir/$file"
done
python experiment3/window_budget_c_20261128/plan.py
python experiment3/window_budget_c_20261128/download.py
python experiment3/window_budget_c_20261128/download.py --hypnograms
python -u experiment3/window_budget_c_20261128/extract.py
python -u experiment3/window_budget_c_20261128/fit_compare.py
```

数据来源 [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)。
