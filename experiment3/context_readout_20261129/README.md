# 睡眠分期 C、D 交接｜物理邻段与 BiGRU

C 的正式主比较是同一份选块的 K5（前后各 5 段）减 K1（前后各 1 段）。D 的正式主比较是信号覆盖 V 减混合随机 R，两臂都用 K1。两项沿用关灯网格、固定 30 轮以及两个 8 人开发队列的 41,498 个评估中心。[`protocol.json`](protocol.json) 登记输入规则、模型、种子和重抽方法，供双方核对。

## D 可以先做什么

1. 使用 `anchor_blocks.csv`、`condition_A_RP_draws_1_5.csv` 和原版配额锁定选块。原 `add_people` 对应 P，`random` 对应 R。V/U 和 R/P 第 6–10 次由 D 锁定。
2. 准备已有冻结特征和本轮 `physical_context.npz`。后者提供同口径开发评估和 C 已补提的特征。下载后按各 Release 的 SHA-256 核对。[已有冻结特征](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w6-shared-cache-v01-20261126) · [本轮物理邻段](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/cd-context-readout-preflight-20261129)
3. 选块锁定后，用 `make_d_input.py` 组装训练输入。脚本核对同 PSG 物理 ±1。真实 PSG 边界夹到第一或最后一个完整 epoch，记录内缺行需要补提。

```bash
python experiment3/context_readout_20261129/make_d_input.py \
  --blocks D_blocks.csv --selection-sha256 "$D_SELECTION_SHA256" \
  --anchor experiment3/context_readout_20261129/anchor_blocks.csv \
  --labels D_locked_center_labels.csv --bundle /path/to/shared/full \
  --c-cache /path/to/physical_context.npz \
  --metadata experiment3/context_readout_20261129/recording_metadata.csv \
  --missing-output D_missing_feature_keys.csv --output D_k1_input.npz
python experiment3/context_readout_20261129/train_bigru.py \
  --experiment D --input D_k1_input.npz --timing
```

`D_blocks.csv` 每块一行，需要 `roster,draw,strategy,record_id,first_epoch`，可附 `last_epoch`（含末端）。标签表需要 `record_id,epoch_index,y_true`，W/N1/N2/N3/REM 为 0/1/2/3/4，不可评分为 -1。缺行清单可按 W4 30k EMA、原本 5 秒编码与 1 秒标准差池化补提，再用 `--supplement` 提供带 `feature_record,feature_epoch,x` 的 NPZ，`x` 为 384 维 float32，EEG 在前。

U 的五个线性头分别按起点的 20 个已标块做有放回 bootstrap。重复抽中的块保留训练权重，仍只用可评分的起点中心。双方签出前核对 U 的重抽种子和 D 的选块散列。

### 实验二复用 R/P 第 1–5 次

C 已交付 260 个共用 K1 拟合。使用 `import_shared_k1.py` 导入 D 的 BiGRU 输出目录，然后把同一目录传给 `train_bigru.py --experiment D`。导入器核验共同锁、缓存等价回执、模型和归档，保留 C 的原拟合缓存散列。D 训练入口会从剩余 806 个单元继续。已另行拟合的文件需先由 D 核对，导入器会保留它们。[共用 K1 Release](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/cd-context-c-k1-20261129)

```bash
python experiment3/context_readout_20261129/import_shared_k1.py \
  --archive c_1_seed0_fits.zip --archive-sha256 "$C_K1_ARCHIVE_SHA256" \
  --input-audit experiment3/context_readout_20261129/shared_k1_input_audit.json \
  --lock experiment3/context_readout_20261129/C_D_signed.json \
  --d-cache cd_context_D_k1_input.npz --output D_bigru
```

## C 的正式训练和出分

在双方签出后使用下面命令。可用一个或多个 `--variant` 先跑 K1/K5。默认覆盖 K1、K5、K10、K5+B 与 K1/K5 的 seed 1、2。每个单元保存模型、逐开发评估者的 5×5 混淆计数、概率和预测，`--csv` 同步导出标准 11 列预测表。D 使用相同入口，并设置 `--experiment D`。

```bash
python experiment3/context_readout_20261129/train_bigru.py \
  --experiment C --input physical_context.npz --joint-lock C_D_signed.json \
  --output C_fits --variant K1 --variant K5 --csv
```

`C_D_signed.json` 需要 `approved_by` 内 C、D 各自的签出记录、`tau_pp` 数值 1 及 `sha256` 对象。散列项包括 `protocol`、`physical_neighbor_rule`、`rosters`、`seeds`、`evaluation_index`、`selections_C`、`selections_D`、`cache_C`、`cache_D`、`analysis.py`、`train_bigru.py`、`13_context_head.py`。双方核对各自散列。共用评估索引 SHA-256 为 `2ba2bcb8c4883f0af9dad658ed95ab49e53efe706c9924987b928e802f1a93a5`。

`analysis.py` 读取含 `counts` 和 `anchor_members` 的 NPZ。计数轴依次为名单、选块策略、重复、比较臂、评估者、真值、预测。C 的策略轴为 P/R，臂轴为 K1/K5。D 的策略轴长度为 1，臂轴为 R/V。脚本给出主比较、双队列重抽区间、多重归属敏感性区间和逐一排除起点受试者的结果。

## 归档基线

旧 K1 训练按物理位置取邻段，评估从共享索引取邻段，缺位时复制中心。此次统一为训练和评估使用同一物理规则。历史条件 A 的 260 个单元只归档了预测和混淆计数，统一评估需要重新拟合 K1。实验二沿用的 R/P 第 1–5 次使用同一批新版 K1 拟合，两边按名单、重复、选块和输入散列对齐。`legacy_replay.json` 保存 C 机按旧规则复跑的 6 个单元。`prepare.py` 和 `replay_legacy.py` 依赖 C 工作区的原始实验脚本及数据，用于追溯缓存构建及复跑过程。D 使用本目录的输入构建和训练脚本即可。
