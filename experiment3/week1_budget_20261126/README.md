# 实验 C｜加人和加深的标注预算配对

2026 年 11 月 26 日启动。C 运行 01–05，D 运行 06–10。每份名单先拟合共同起点，再用相同的 600 段新增标签比较加人与加深。36 人训练池内评估为主表，固定 dev 8 人单独评分。

[下载交接包](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/c-budget-week1-v01-20261126/c_budget_week1_handoff_v01.zip) · [下载校验值](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-budget-week1-v01-20261126) · [06 号名单](rosters/06/blocks.csv)

## 名单与选块

先生成 06，再生成 01 和其余名单。名单、评分对象和源文件 SHA-256 均登记在 `rosters/`。源文件来自组内 W4 无分期标签的逐段索引、组内受试者划分、W4 预训练划分及固定 dev 样本参考。选块只读取训练池索引的受试者、记录号和原始 epoch 序号。

任务书未规定具体抽样规则。本次用 `20261126 + 名单号` 定名单种子，用 0 定分类头种子。连续区间从首个 epoch 开始铺不重叠的 60 段候选块，余数舍弃。用块中点在该夜首末可用 epoch 之间的位置分前、中、后三段。每位加深对象与一位加人对象配对抽取相同的两个第三段。加深对象再从原夜剩余候选块抽四块给共同起点。46 人都有至少一夜符合抽样条件，合计 90 夜。每份名单的三组都从评分中排除同一个 10 人并集。

`rosters/06/blocks.csv` 是 D 的首份名单，`rosters/01/blocks.csv` 是 C 的联调及 loss 检查名单。块的起止 epoch 都包含在内。对应评分对象见同目录的 `evaluation_subjects.csv`。

## 运行

用带 NumPy、PyTorch 和 scikit-learn 的 Python 3.11 或更新版本。在交接包根目录运行以下命令。从仓库直接运行脚本时，`--data` 指向解压交接包后得到的 `data` 目录，`--plan` 指向本目录的 `rosters`。

收到 D 的共享特征后，将 D 的发布回执与任务书约定核对，确认是均匀 timestep 训练的 W4 30k EMA、5 秒窗、读出 t=1、10–12 层、每秒 std 后在 30 秒内平均。完整缓存覆盖 54 名训练池受试者和 dev 8 人。训练入口沿用 04 的文件格式，目录包含 `features_info.json`、`index.csv`、`eeg_std.npy`、`eog_std.npy`。索引与两个数组行序相同，数组各为 192 维 float32。回执列出 step、weights、pool_sec、dim_per_modality、n_samples、index_sha256 和 files 下两数组的 sha256。使用独立取得的回执 SHA-256。训练器核验回执、索引、数组及完整样本键。W4 held-out 8 人只在缓存和索引对齐中出现。

```text
python train.py --plan rosters --data data --bundle /path/to/D/cache --output runs --feature-info-sha256 D_RECEIPT_SHA256 --roster 01 --fit-only
```

01 起点训练结束后，查看 `runs/01/loss_review/base/loss.csv` 与 `runs/01/loss_review/summary.json`，记录最后五轮斜率和是否仍明显下降。两人锁定轮数后，D 使用同一份 01 loss-only 输出，或用相同输入在本地复跑该步骤。批准 30 轮时，用已复核 loss 曲线的 SHA-256 启动三组。若确需改轮数，先登记并由 C/D 统一版本后再运行。

```text
python train.py --plan rosters --data data --bundle /path/to/D/cache --output runs --feature-info-sha256 D_RECEIPT_SHA256 --roster 06 --reviewed-loss-sha256 REVIEWED_LOSS_SHA256
```

脚本沿用 34 号 probe 经 27 号 probe 的五类线性头、训练块内标准化、交叉熵和 AdamW 设置。固定末轮，batch 128 保留尾批。每组保存训练损失、标准化参数、模型、更新步数和代码及特征指纹。每份名单另存两套同键的逐段真值和五类概率，以及各组 Macro-F1 和配对差（pp）。原始 Macro-F1 对五类等权。
