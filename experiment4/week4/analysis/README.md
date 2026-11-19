# W4 D 下游 probe 结果（初版，11-19）

口径：B 第三周协议（27/28/29 号脚本，由 `src/05_probes.py` 调用），标签预算 label25（14 人）/ label10（5 人），sseed 2024，head seed 0–2，dev 8 人只评一次，测试集未使用。特征一律由 `src/04_extract_features.py` 在 D 的 RTX 4060 Laptop 上提取（EMA 权重，t=1，12 层，1 秒 std pooling）。**预训练只有 1 个 backbone seed（seed 0）**，下面的区间只反映 head seed 和受试者抽样的波动，不反映预训练 seed 的波动。

Δ 单位为百分点，括号内为按受试者 bootstrap（1000 次）的 95% 区间。

## 1. 主实验各 checkpoint（`w4_main_ckpts/`、`w4_main_ckpts_pairwise/`）

EEG+EOG std 特征，dev Macro-F1（3 个 head seed 平均）：

| checkpoint | label25 | label10 |
|---|---|---|
| 随机初始化（step 0） | 0.714 | 0.670 |
| 15k | 0.731 | 0.703 |
| 30k | 0.734 | 0.716 |
| 60k | 0.738 | 0.723 |
| 120k | 0.742 | 0.720 |
| 240k | 0.742 | 0.723 |

- 各点相对随机初始化：Macro-F1 全部提升，区间都不含 0（label25 +1.7 至 +2.8，label10 +3.3 至 +5.4）。
- 各点之间（`w4_main_ckpts_pairwise/d_paired.csv`）：
  - 30k vs 15k：label10 +1.30 [+0.57, +1.77]，label25 +0.22 [−0.16, +0.47]。
  - 60k / 120k / 240k vs 30k：Macro-F1 全部在 +0.35 至 +0.90 之间，**区间都含 0**。
  - 240k vs 60k、240k vs 120k：Macro-F1 区间都含 0。
- 读法：15k→30k 在 5 人档有可测的提升；30k 之后继续训练，下游 Macro-F1 的变化在单 seed 下测不出来。held-out val MSE 同期仍在缓慢下降（见 `runs/W4_D_main_m64x12_seed0_v01/README.md`）。
- 只用 EOG 特征（30k vs 随机初始化）：Macro-F1 label25 +2.11 [+1.14, +3.45]，label10 +3.05 [−0.12, +5.98]；N1 F1 未提升（−1.94 / −0.95，区间含 0）。

## 2. self-conditioning 30k（`w4_selfcond_30k/`）

`W4_D_selfcond_l9_mean_30k_v01`（反馈层 m=9，由 `analysis/sc_profile_choice.json` 的 2k 步短跑选出；三个候选的 val 差距 <1%）。对照为主实验 30k。

| 对比 | 预算 | Macro-F1 | N1 F1 | N1 AP |
|---|---|---|---|---|
| self-cond 全层 std 特征 | label25 | +0.26 [−0.24, +0.73] | +0.43 [−1.46, +2.18] | +1.00 [+0.15, +1.62] |
| self-cond 全层 std 特征 | label10 | −0.14 [−0.79, +0.32] | +0.64 [−1.11, +2.01] | +0.84 [−0.97, +2.14] |
| feedback 特征 | label25 | −7.59 [−12.82, −3.39] | −14.81 | −10.80 |
| feedback 特征 | label10 | −7.41 [−8.57, −5.83] | −10.63 | −8.37 |

- 全层特征与主实验 30k 打平；label25 的 N1 AP 区间不含 0，但只有单 seed 且 label10 不成立，不作为有效证据。
- 单独的 feedback 特征明显更差，不能替代全层读出。

## 3. 错时 EEG 组 EOG-only（D 侧核对，`w4_main_ckpts_pairwise/`）

`b_shift030k_eog`（B 的错时组 30k，D 代跑）vs `d_pre030k_eog`（无 EEG 30k），同一套特征提取与 probe：Macro-F1 label25 +0.12 [−0.43, +0.67]、label10 −0.29 [−0.99, +0.78]；N1 AP 区间均含 0。三组正式汇总以 B 的结果为准。

## 文件

- `d_summary.csv`：各 arm × 预算的均值与 min/max；`d_paired.csv`：配对比较；`d_n1_adjust.csv`：N1 α 调整；`d_eval_registry.csv`：预测表位置（逐 run 的预测表未入库）。
