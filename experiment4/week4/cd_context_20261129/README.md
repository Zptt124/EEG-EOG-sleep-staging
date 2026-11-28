# 成员 D · C/D 上下文实验（2026-11-29 修订版）· 实验二选块锁定

对应 C 的方案 `睡眠分期课题_CD双人实验方案_上下文读出与选块规则_修订版_20261129` 和交接分支 `feat/cd-context-readout-20261129`。本目录是 D 在**签出前**交付的部分：实验二全部选块、精度预估和 D 的脚本。新得分尚未生成。

## 1. 已锁定的选块

`selection_lock/D_blocks.csv`，SHA-256 `e8597867156cb01e401a2b65e1ec87ee84597f9ce6a31cb8f9579f1c422bbe17`（即签出文件的 `selections_D`）。

| 策略名 | 方案中的臂 | 每份名单 | 说明 |
| --- | --- | --- | --- |
| `random` | R 混合随机 | 10 次 | 第 1–5 次与 C 交接的 `condition_A_RP_draws_1_5.csv` 逐行一致（脚本内核对）；第 6–10 次用同一共享排序（SEED 2026112903）生成 |
| `add_people` | P 新人 | 10 次 | 同上 |
| `coverage` | V 信号覆盖（主比较） | 10 次 | epoch 级 min-sum 覆盖度，按块贪心，每步从前 3 名按种子取 1 块 |
| `coverage_det` | V 确定性版 | 1 次（draw 1） | 每步取第 1 名，只作敏感性分析 |
| `disagreement` | U 多头分歧（探索） | 10 次 | 5 个线性头各按起点 20 块有放回 bootstrap 训练；BALD 式分歧的块均值，按时段从前 3 名取 |

种子与全部输入、代码的 SHA-256 见 `selection_lock/lock.json`。核对项：每单元恰好 10 块；时段配额与 `mixed_quotas.csv` 一致；只用条件 A；与起点无交集；**全程不读取候选块的分期标签或不可评分标记**（U 只用起点块标签）。V 的增量轨迹见 `v_trace.csv`，U 的块得分见 `u_scores.csv`。

**来源构成**（26 份 × 各次均值，块数 / 10）：

| 策略 | 原人原夜 | 原人另一夜 | 新人 | 覆盖人数 | 覆盖夜数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 2.44 | 3.78 | 3.78 | 6.61 | 7.67 |
| add_people | 0 | 0 | 10 | 5.00 | 5.00 |
| coverage | 0.71 | 3.54 | 5.75 | 7.49 | 7.79 |
| coverage_det | 0.73 | 3.77 | 5.50 | 7.69 | 7.96 |
| disagreement | 1.43 | 2.27 | 6.30 | 3.93 | 4.13 |

覆盖法比混合随机多选约 2 个新人块，所以 V − P 和来源匹配对照对解释很重要。

## 2. 候选池特征补提

V/U 需要整个候选池（26 份名单的 1,290 个唯一 A 块，77,400 段）的冻结特征。`01_extract_candidate_features.py` 按 W4 30k EMA、第 10–12 层、1 秒 std 池化补提缓存缺失的段（不解析 hypnogram 标签，只核对起始时钟）。与共享缓存重叠段的重放误差最大 2e-4（容差 atol 2e-4 / rtol 5e-3）。补提文件较大，签出时随 D 的输入缓存一起放 Release。

## 3. 精度预估（开跑前）

`precision/precision_estimate.json`。用第一轮归档的条件 A 分数，以 P 代理 V，只重抽名单：

| 模型 | 5 次：半宽 / 最小可检测差 | 10 次：半宽 / 最小可检测差 |
| --- | --- | --- |
| BiGRU K1 | 0.69 / 0.98 pp | 0.56 / 0.80 pp |
| 线性头 | 1.04 / 1.49 pp | 0.99 / 1.41 pp |

P 和 R 共用排序，相关性高于 V 和 R，所以 V − R 的实际区间可能更宽；评估者重抽也会让区间略宽。

## 4. 签出后的流程（D）

1. `04_center_labels.py`：选块锁定后才导出中心标签表。
2. C 的 `make_d_input.py` 组装 `D_k1_input.npz`（需要 C 的 Release `cd-context-readout-preflight-20261129` 里的 `physical_context.npz`），缺行按同一 checkpoint 补提。
3. C 的 `train_bigru.py --experiment D`：BiGRU K1，5 种策略共 26 × 41 = 1,066 个单元。
4. `05_linear_fits.py`：线性头辅助模型，同一批单元。
5. `06_analysis_d.py`：调用 C 的 `analysis.py`。主比较 BiGRU coverage − random；解释性比较 V − P、U − R、P − R、coverage_det − R，线性头同样四项。

## 5. D 还需要 C 提供

- Release `cd-context-readout-preflight-20261129`（截至本提交在 GitHub 上还看不到）。
- 共同签出文件 `C_D_signed.json`。D 这边的 `selections_D` 已在上面给出；`cache_D` 等输入组装完成后补上。

D 同意 `protocol.json` 中的邻段规则、重抽方法、τ = 1 和两个主比较，也同意 C 已写入的两条补充（U 臂 bootstrap；K1 重训时 R/P 第 1–5 次一并重跑）。
