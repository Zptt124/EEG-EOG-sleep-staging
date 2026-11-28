# Ying 外部复验｜优先标新人

外部队列来自 [Ying 等的多夜睡眠库](https://doi.org/10.1038/s41597-025-05202-6) 及其 [OSF 归档](https://osf.io/gx36m/)。纳入 20 位健康青年的前两夜，共 38 夜、39,505 个评估用 epoch。每个评估折内训练和评价分类头时按人隔离，冻结表示沿用在 Sleep-EDF 上训练的 W4 30k EMA。4 个固定评估折，每折 8 份训练名单，每名单 5 次配对抽样、三臂共 480 次拟合。

协议中的 `same-amplifier Days 1 and 2` 指放大器型号。Ying 等报告前两晚均使用 ICSMS（成都，中国），第三晚使用 actiCHamp（Brain Products，德国）。出处为数据论文 Methods 的 “Data recording equipment” 段，PDF 第 2 页表 1 下方。[Ying 等，Scientific Data 12, 946 (2025)](https://www.nature.com/articles/s41597-025-05202-6.pdf#page=2)

主比较「新人新夜 − 原人原夜」为 **+1.79 个 Macro-F1 百分点**，95% 配对区间为 **[+0.58, +3.07]**。4 折均值同向，32 份名单中 24 份同向。按外部预设规则，点估计至少 +1、区间下限大于 0、至少 3/4 折方向为正，结果达到推进门槛。区间仍包含小于 +1 的效应。完整三组比较见 [并列表](../week3_budget_replication_20261128/README.md)。

## 窗口与候选块

Ying 以与专家标注对齐后的记录长度为基准，将每晚等分为三个时段，并按块中点归入相应时段。抽样前，只保留块内 60 个连续的 30 秒片段全部可评分的候选块。**+1.79 个 Macro-F1 百分点**支持在这一筛选条件下「新人新夜优于原人原夜」。关灯窗口下「新人是否优于混合随机」应依据实验 D 的直接比较。[对齐代码](extract.py#L47-L73) · [抽块代码](confirm.py#L47-L70) · [实验 D 关灯窗口对照](https://github.com/Zptt124/EEG-EOG-sleep-staging/tree/7df91959fd589746574e8ebc2fa170a47f2238dd/experiment4/week4/random_arm_w3)

## 结果与复算

B 核对协议修订时间线所需的三版原件、文件系统时间、终端记录摘录、原始日志和实际运行版提取器，见 [C → B 原始运行证据](results/run_evidence/README.md)。

归档含 [冻结协议](results/protocol.json)、[配对块](results/blocks.csv)、[输入指纹](results/data_lock.json)、[汇总](results/summary.json) 以及各折按评估者汇总的混淆计数。两份协议快照保留对齐和训练资格核查前的版本。运行下列命令复算两个队列的点估计、区间与判读，使用 Python 3.11、NumPy、PyTorch、scikit-learn 即可。

```bash
python experiment3/week4_budget_external_ying/verify_results.py
```

## 重新拟合

1. 从仓库根目录运行 `python experiment3/week4_budget_external_ying/download.py`，脚本从 OSF 下载并校验压缩包。原始信号、专家逐段标签和特征存放在本机运行目录。
2. 从 [D 的 W4 30k 交付](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-pretrain-w4-main30k-v01-20261117) 取 `w4_d_main_ckpt_0to30k_v01.zip` 内的 `ckpt_step30000.pt`，放到 `output/budget_external_ying/`。使用 [D 的 W4 源码 6590ec9](https://github.com/Zptt124/EEG-EOG-sleep-staging/tree/6590ec90a9fcc4ea6e1d7118556191f618117810/experiment4/week4/src) 和 [官方 EEGDM 源码 003b5a2](https://github.com/jhpuah/EEGDM/tree/003b5a2a016b2f02788ea5693d8b7cd00b6bd969)。
3. 在带 CUDA 版 PyTorch、MNE、SciPy 及上述源码依赖的环境运行以下命令。发布版提特征脚本把两个源码目录做成命令行参数，其余数据处理逻辑沿用实际运行版。实际运行版 `extract.py` 的 SHA-256 是 `018da6f7b2a999e16ec9e04f52a338873a05a0a0e76ec4d138d0aa34dd524924`。

```bash
python experiment3/week4_budget_external_ying/extract.py output/budget_external_ying \
  --w4-source /path/to/W4/experiment4/week4/src --eegdm-vendor /path/to/EEGDM
python experiment3/week4_budget_external_ying/confirm.py output/budget_external_ying --freeze
python experiment3/week4_budget_external_ying/confirm.py output/budget_external_ying
```
