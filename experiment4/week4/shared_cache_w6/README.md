# D · 新一轮第 1 周共享特征缓存（384 维，W4 30k EMA，第 10–12 层）

2026-11-26。供预算线（C、D）和时序线（A、B）共用。按[新一轮第 1 周安排](../../../docs/budget-temporal-20261126/README.md)的统一口径生成。

## 口径

| 项目 | 设置 |
| --- | --- |
| 权重 | W4 主实验 `W4_D_main_m64x12_seed0_v01` 的 `ckpt_step30000.pt`，`ema_model`（SHA256 `21c8514e…`，与 Release `d-pretrain-w4-main30k-v01-20261117` 相同） |
| 预训练设置 | 均匀 timestep、5 秒窗、46 人训练名单（W4 held-out 8 人不参与） |
| 读出 | t=1、不加噪、query=gate、空条件；只取第 10–12 层（全层 768 列中的第 576–768 列）；每 1 秒求 std（correction=1），再在 30 个位置上平均 |
| 维度 | `eeg_std.npy`、`eog_std.npy` 各 [N, 192]，float32；下游按 EEG 在前、EOG 在后拼成 384 维 |
| 范围 | 训练池 54 人（106 晚，132,938 段，含 W4 held-out 8 人，只缓存）+ 固定 dev 8 人（16 晚，21,186 段），共 154,124 段。测试集 16 人未读取 |
| 行序 | 训练池按 `pretrain_epoch_index.csv`（SHA `e3632bce…`）的行序，后接 dev 按记录号、epoch 排序；`index.csv` 与两个数组同行序 |

`index.csv` 列：`sample_id, subject_id, record_id, epoch_index, split, role, y_true`。`role` 为 `pretrain_train`（46 人）、`heldout_val`（W4 held-out 8 人）或 `dev`。`y_true` 来自 A 的 NPZ，dev 部分已与 `val_reference_v1.csv` 逐段核对。

## 核对

- 每份记录的 epoch 与冻结索引逐个对上，否则脚本报错退出；样本键与 `pretrain_epoch_index.csv` + `val_reference_v1.csv` 完全相同。
- 与 W4 已发布的 30k 全层特征（14 + 8 人，53,496 段）重叠部分逐位相等（取第 576–768 列）。结果写在 `features_info.json` 的 `check_against_w4_all_layers`。
- 计算代码复用 `04_extract_features.py` 的 `gate_activations` / `pooled`，入口是 `src/09_extract_shared_cache.py`。

## 文件与 SHA-256

见 Release 页面和 `../registry/artifact_index_D.csv`。C 的 `train.py` 需要的是 `features_info.json` 的 SHA-256。

## 复现

```bash
python -u 09_extract_shared_cache.py \
  --heldout-csv ../../../manifests/pretrain_split_w4_v01.csv \
  --val-reference ../../../manifests/val_reference_v1.csv \
  --out-name W6_D_main30k_l10_12_v01 \
  --check-against <W4 30k 全层特征目录>
```

RTX 4060 Laptop 上约 50 段/秒，全量约 52 分钟，显存峰值约 2.3 GB。
