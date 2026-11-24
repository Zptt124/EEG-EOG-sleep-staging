# W5 C 四格实验

## 实验设置

- 使用普通 30k EMA 和第 9 层 self-conditioning 的 30k EMA。每个 backbone 分别读出第 1–12 层和第 10–12 层，固定 EEG+EOG、diffusion timestep `t=1`、无输入加噪，以及每秒 `std` pooling 后在 30 秒 epoch 内平均。
- 5 人和 14 人两档标签预算各训练 3 个 linear head seed（0、1、2）。用训练受试者的 5 折交叉验证选训练轮数。复用普通模型的 12 个已完成 probe，新跑 self-cond 的 12 个 probe。两种 checkpoint 各对应一个 backbone seed。
- D 的特征压缩包通过 SHA256 核验，53,496 段样本索引与普通模型逐字节一致。24 个 probe 的 dev 预测样本、标签和各自预算内的 OOF folds 对齐。每个 probe 评估同一批 8 名 dev 受试者的 21,186 个 epoch。封存的 16 人测试集未使用。[D 发布的特征](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w4-selfcond30k-feat-v01-20261124)
- 将 6 个 self-cond 全层 probe 与 D 的公开基准核对，Macro-F1、N1 F1 和 N1 AP 的 18 个数值最大绝对差为 `8.7e-7`。[D 的基准明细](https://raw.githubusercontent.com/Zptt124/EEG-EOG-sleep-staging/bdbb973ac599e7499098b10f02116de59b5aff26/experiment4/week4/analysis/w4_selfcond_30k/d_metrics.csv)

## Dev 指标

表中为 3 个 head seed 分数的算术平均。F1 用默认 argmax 决策。另有一份在训练 OOF 上调 N1 阈值后的结果，保存在汇总表中。

| 标签预算 | Backbone | 读出层 | Macro-F1 | N1 F1 | N1 AP |
| --- | --- | --- | ---: | ---: | ---: |
| 5 人 | 普通 | 全 12 层 | 0.7155 | 0.2982 | 0.3294 |
| 5 人 | 普通 | 10–12 | 0.7137 | 0.3359 | 0.3436 |
| 5 人 | Self-cond | 全 12 层 | 0.7141 | 0.3046 | 0.3379 |
| 5 人 | Self-cond | 10–12 | 0.7086 | 0.3179 | 0.3224 |
| 14 人 | 普通 | 全 12 层 | 0.7336 | 0.3464 | 0.3819 |
| 14 人 | 普通 | 10–12 | 0.7389 | 0.4139 | 0.3903 |
| 14 人 | Self-cond | 全 12 层 | 0.7360 | 0.3504 | 0.3927 |
| 14 人 | Self-cond | 10–12 | 0.7372 | 0.3981 | 0.3774 |

## 配对比较与开销

配对效应合并 3 个 seed 的预测后计算。95% 区间以 dev 受试者为单位做 1,000 次配对 bootstrap，四格共用同一批抽样。区间反映当前 8 名 dev 受试者上的不确定性，未纳入不同 backbone seed 的变异。下文差值单位为百分点。

- Self-cond 内读后三层相对于全层，Macro-F1 在 5 人档变化 −0.63，95% CI [−1.91, +1.06]，在 14 人档变化 +0.11，95% CI [−0.99, +1.11]。点估计接近，区间仍允许小幅下降或提升。
- 固定只读后三层，self-cond 相对于普通模型的 Macro-F1 在 5 人档变化 −0.51，95% CI [−1.84, +0.84]，在 14 人档变化 −0.17，95% CI [−1.00, +0.73]。普通模型也能用后三层取得接近全层的 Macro-F1。这次对照没有看到 self-cond 在后三层读出上的额外收益。
- 在 14 人档，self-cond 只读后三层时，默认 N1 F1 增加 4.74，95% CI [+0.14, +8.29]。N1 AP 同时下降 1.45，95% CI [−3.67, −0.20]。用 OOF 调整 N1 决策后，后三层相对全层的 N1 F1 差值为 −2.28，95% CI [−5.10, −0.20]。
- 两个 backbone 的读出宽度都从 1,536 减到 384 维，缓存从约 333 MB 减到 87 MB。Self-cond 的 CPU linear head 前向中位数从 560 µs 降到 155 µs（4 线程、512 样本、特征已在内存）。冻结 backbone 仍需完整计算第 1–12 层。本轮计时只覆盖 head 前向，后几层 LFT 的训练开销需要单独测量。

[指标汇总](summary.csv) · [配对差值及区间](paired_contrasts.csv) · [逐 seed 差值](paired_by_seed.csv) · [读出开销](readout_cost.csv) · [数据及预测核验](verification.json)
