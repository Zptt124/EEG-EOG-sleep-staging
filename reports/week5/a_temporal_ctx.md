# W5 · A：Temporal context 对照（2026-11-24）

按 [W5 实验安排](../../WEEK5.md) A 线：固定 W4 普通 30k backbone（无 EEG 条件），在 5 人 / 14 人
两档标签预算下比较三种下游读出——单 epoch linear、±2 邻居直接拼接、小 BiGRU——看 temporal
context 能否缓解 5 人档直接拼接的下降并保留 14 人档的收益。

**一句话结论**：待实验结果填入。

## 1. 输入与核对（周一任务，完成）

- 特征：D 的 04 号 30k 普通（无 EEG 条件）双模态特征
  `W4_D_main_m64x12_seed0_v01/step30000_ema_model_pool1s/`
  （eeg_std / eog_std [53496,768] float32 + index.csv，checkpoint SHA `21c8514e…`）。
- 核对结果：43 记录 / 22 人（14 训练 + 8 dev），53,496 段；sample_id = `{record}_{epoch:04d}`，
  同记录内行序与 epoch 序一致；13 条记录有缺窗（共 55 个 epoch，最大缺 12）；y_true 齐全；
  与 npy 行序逐行对齐抽检通过。
- 交叉验证：41 号 single 臂与 34 号 `--modalities eeg+eog` 在 label10 seed0 上
  **预测表逐位一致**（同引擎、同协议、同线程数）。

## 2. 协议（与 34 号完全一致）

- 预算内按人 5 折（fold_seed=0，固定 50 轮），每折标准化只拟合折内训练者，OOF Macro-F1 选 E*
  （并列取最早）；全名单重训 E* 轮；8 人 dev 只最终评分一次；α 在 E* 的 OOF 上按 28 号网格选。
- 三臂（每 epoch 1536 维 = eeg_std|eog_std）：
  - single：Linear(1536,5)
  - cat：当前 epoch ±2（5×1536=7680 维）拼接 → Linear(7680,5)
  - bgru：小 BiGRU（hidden 64，双向，1 层）读 5 个 epoch，取中心时刻输出 → Linear(128,5)
- 窗口规则（统一处理缺窗与记录边界）：邻居限定同一受试者、同一记录；缺窗/边界取该方向上
  最近可用 epoch（可用序列位置钳制，边界复制）；只预测中心 epoch。
- 标准化口径：窗口臂在展开矩阵 [N*5,1536] 上拟合（每个位置共用同一 mean/std），与单 epoch 臂
  完全同构。head seed / backbone seed 均注明；backbone 只有一个（30k 普通）。

## 3. 结果

（实验完成后填入：两档预算 × 三臂的 OOF / dev Macro-F1、N1 AP、N1 F1 原始与调整后；
seed 合并的配对差与逐 seed 方向。）

## 4. 结论与建议

（实验完成后填入：BiGRU 是否缓解 5 人档下降并保留 14 人档收益；下一轮是否保留 temporal
context；是否值得补 240k 或折内降维拼接。）

## 附：资源与脚本

- `scripts/41_a_w5_temporal_compare.py`（三臂对照）、`scripts/42_a_w5_temporal_paired.py`（配对）
- run 目录 `runs/W5_A_tctx_*`；逐 run 小文件与预测随分支提交
