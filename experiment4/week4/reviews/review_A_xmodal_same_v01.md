# D 对 A 同窗 EEG 30k 实现的 review（v01，2026-11-19）

对象：PR #24（分支 `feat/a-week4-pooling-cv`，HEAD `ee4bbd3`）中的同窗 EEG 30k run `W4_A_xmodal_same_30k_seed0_v01`，以及 Release `a-w4-xmodal-same-30k-v01-20261119`。

## 结论

- **训练实现：通过。** A 直接使用公共预训练脚本，没有改代码；设置与无 EEG 组、错时组一致，只有跨模态模式不同。
- **提特征：有一处偏差，但影响可以忽略，不需要重提。** 以后统一用 `src/04_extract_features.py`。
- **val MSE：** 配方和 seed 与全组相同；噪声 SHA256 不同，原因是 CPU 浮点末位差异。D 已用全组统一的噪声重算（见 `analysis/val_recompute/`）。

## 逐项核对

| 项目 | 结果 |
|---|---|
| 训练代码 | `src/w4d_common.py`、`w4d_model.py`、`w4d_train.py`、`01_pretrain.py` 与 D 分支逐字节相同 |
| 跨模态设置 | `same_window`，`cond_dropout` 0.2，只有 EOG 目标窗接收 EEG 条件，评估主口径不给 EEG，符合 C 的 `cross_modal_protocol.md` |
| held-out 名单 | `pretrain_split_w4_v01.csv`，SHA256 `470c23f7…`，8 人与全组相同 |
| 训练窗、缓存 | `train_units_sha256` `d710f5bb…`、`cache_sha256` `baa935f7…`，与无 EEG 组、错时组相同 |
| 模型与优化 | m64x12，backbone seed 0，batch 108，30k 步，EMA 设置相同 |
| checkpoint | Release 中 `ckpt_step30000.pt` 的 SHA256 前缀 `ba81448a`，与 A 的 `checkpoints.csv`、`stop_decision.json` 一致 |
| 训练过程 | 0.47–0.51 s/step，显存峰值 6,040 MiB，没有溢出迹象；step 200 的 loss 0.2328，另外两组为 0.2332 / 0.2329（不同 GPU，不要求逐位相同） |
| val MSE 噪声 | 配方 `torch.Generator(cpu).manual_seed(20261114); torch.randn((2048, 10, 1, 1000))` 相同；噪声 SHA256 为 `89812e14…`（全组参考 `3465eadc…`）。step 0 的 stop_metric 两边都是 0.772786 |

## 提特征的偏差

A 的 `scripts/39_a_w4_eog_extract.py` 使用官方 `LatentActivityExtractor`，脚本说明里认为"`xc_flag` 是零初始化，空条件加 0，与官方前向等价"。这只在训练开始时成立：

- 训练后 `xc_flag.weight` 不再为 0（A 的 30k EMA 权重中范数为 0.099）。
- 训练时的空条件分支会加上这个 flag 嵌入，A 的提特征没有加，所以特征和训练时的空条件前向不完全一致。

影响大小（D 测量）：
- 同一个 checkpoint，在训练池中随机抽 64 段 EOG，加与不加 flag 的 EOG std 特征，相对 L2 差异平均 0.34%（0.27%–0.43%）。
- 下游 probe：C 用 A 原特征得到的同窗 vs 无 EEG 为 label10 −0.56、label25 +0.37；D 用 04 号脚本重新提取后为 −0.63、+0.29（`analysis/w4_xmodal_recheck/`），差别不到 0.1 个百分点。

建议：A 的结论不需要改；以后所有跨模态 / self-cond 模型的特征都用 04 号脚本提取，保证与训练时的前向一致。

## 其他

- A 提的特征以 float16 存储，D 的 04 号脚本以 float32 存储。在本次结果上看不出影响，正式汇总以 04 号脚本的特征为准。
- pooling 长度：A 在 PR #24 推荐 2 秒（针对 LFT 提速）；C 已锁定三组线性探针用 1 秒（11-19）。
- val MSE 统一重算结果：A 的同窗 30k（EMA，不给 EEG）stop_metric 0.054065，与 A 本机记录在 6 位小数上完全相同（给 EEG 时 0.053881，同样相同）。噪声 SHA 不同对 val MSE 没有可见影响。
