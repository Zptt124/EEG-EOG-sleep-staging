# W6 · A：timestep sampling 筛查（logitnormal vs 均匀）· 负面结果（2026-11-26）

按 W6 执行指南：把训练时 diffusion timestep 抽样从均匀改成偏向中小 t 的 logitnormal
（μ=−0.5, σ=1，t = floor(sigmoid(u)×50)），其余全部不变，训 30k 步，与 W4 主实验 30k
（uniform）做同口径 linear probe 对照。**只跑 1 个 backbone seed（0），3 个 head seed（0/1/2），
8 人 dev 只最终评分一次，测试集（16 人）未使用。**

**一句话结论：有害，停止。** 5 人档 N1 AP 配对差 −3pp，95% 区间整体为负（原始
[−0.06, −0.01]、α 调整后 [−0.05, −0.01]），Macro-F1 点估计也为负（5 人档调整后
−2pp，区间整体为负）；14 人档无差异。logitnormal 抽样不加入下一轮框架，不补 backbone seed。

## 1. 训练与核对

- 代码：D 的 `feat/d-week6-tsampling`（`--t-sampling logitnormal`，默认 uniform 不变），A 未改任何参数。
- 冒烟：VAL step0 stop_metric 0.77105 ✓；loss step50 0.53 → step200 0.09 ✓；config 记录
  `t_sampling: {mode: logitnormal, logit_mean: -0.5, logit_std: 1.0}` ✓。
- 正式训练 `W6_A_tlogit_m64x12_seed0_30k_v01`：起点 VAL step0 stop_metric 0.77279 与 W4
  主实验逐位一致 ✓；train subjects 46、units 1,368,492 ✓；0.50–0.52 s/step（中途两次暂停续训，
  从 latest.pt 恢复）。30k 完成：final loss 0.0546，val stop_metric 0.05372@30k（best=30k）。
  注意：val MSE 不作判定（大 t 段按新分布必然变差），只作记录。
- checkpoint（30k EMA）SHA256 `2cdef40bb8605940806d72002c2b935c7b5c64d4a7cb953905af273278429d05`。
- 提特征（04 号，t=1、12 层、1s std、30 位置平均，53,496 段，EOG-only 验收通过）。
- 基线特征 = W4 主实验 30k（zip SHA 9072c803…）；转换核对：w6_base_all 开发集 Macro 三 seed
  均值 label10 0.7155 / label25 0.7336，与 W4 报告（≈0.716 / ≈0.734）一致 ✓。

## 2. 绝对分数（开发集 21,186 段，3 head seed 均值）

| 预算 | 臂 | Macro-F1 | N1 AP | N1 F1 原始 | N1 F1 调整后 |
| --- | --- | --- | --- | --- | --- |
| 5 人 | base all（uniform） | 0.7155 | 0.329 | 0.298 | 0.391 |
| 5 人 | tlogit all | 0.7092 | 0.300 | 0.262 | 0.374 |
| 5 人 | base l10 | 0.7137 | 0.344 | 0.336 | 0.365 |
| 5 人 | tlogit l10 | 0.6916 | 0.293 | 0.232 | 0.361 |
| 14 人 | base all（uniform） | 0.7336 | 0.382 | 0.346 | 0.455 |
| 14 人 | tlogit all | 0.7280 | 0.382 | 0.335 | 0.450 |
| 14 人 | base l10 | 0.7389 | 0.390 | 0.414 | 0.443 |
| 14 人 | tlogit l10 | 0.7223 | 0.367 | 0.342 | 0.429 |

训练侧 OOF Macro-F1（E* 轮，3 seed 均值）：base all 5 人 0.710 / 14 人 0.731；tlogit all
5 人 0.692（−1.8pp）/ 14 人 0.732——OOF 方向与开发集一致（5 人档为负、14 人档无差异）。

## 3. 配对差（tlogit − base，百分点；按 8 名 dev 受试者成对 bootstrap 1000，rng seed 0）

**主比较 all（12 层）**

| 预算 | 口径 | ΔMacro-F1 | ΔN1 AP | Δ原始 N1 F1 |
| --- | --- | --- | --- | --- |
| 5 人 | 原始 | −0.01 [−0.01, +0.00] | **−0.03 [−0.06, −0.01]** | **−0.04 [−0.07, −0.02]** |
| 5 人 | α 调整后 | **−0.02 [−0.03, −0.02]** | **−0.03 [−0.05, −0.01]** | **−0.02 [−0.03, −0.01]** |
| 14 人 | 原始 | −0.01 [−0.01, +0.00] | −0.00 [−0.04, +0.02] | −0.01 [−0.04, +0.02] |
| 14 人 | α 调整后 | −0.01 [−0.01, −0.00] | +0.00 [−0.02, +0.02] | −0.01 [−0.02, +0.01] |

**次级比较 l10（第 10–12 层）**：无正信号（5 人原始 N1 F1 −0.10 [−0.20, −0.02] 整体为负，
调整后与 14 人各格均跨 0）。

## 4. 判定（按指南第 5 节）

5 人档 N1 AP 区间整体小于 0（原始与调整后两套口径）→ **有害**。Macro-F1 点估计全为负、
5 人档调整后区间整体为负。按指南：停止，作为负面结果记录，不再投入算力（不补 backbone
seed 1/2，不加入下一轮完整实验框架）。

## 5. 共用结果表行（团队 W6 表建立后粘贴）

| 负责人 | 实验对照 | 关键数字 | 一句话结论 | 结果链接 |
| --- | --- | --- | --- | --- |
| A | logitnormal vs uniform timestep 抽样（30k，1 backbone seed，3 head seed） | 5 人档 ΔN1 AP −3pp [−0.06, −0.01]（原始）与 [−0.05, −0.01]（调整后）均整体为负；14 人档无差异 | 有害，停止；不补 seed、不进下一轮框架 | [A 报告](reports/week6/a_tsampling.md) |

## 附：资源

- run 小文件：`experiment4/week4/runs/W6_A_tlogit_m64x12_seed0_30k_v01/`（config、history、
  val_mse、checkpoints、stop_decision）
- 24 个 probe run 的 11 列预测、n1adj、metrics、config：`reports/week6/a_tsampling/runs/`
- 配对评估输出：`reports/week6/a_tsampling/eval/`
- 大文件（checkpoint、特征、probe 权重）：Release `a-w6-tlogit-30k-v01-20261126`
