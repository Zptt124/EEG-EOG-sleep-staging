# B 第三周状态（实验 B 线性读出与统一评估）

方案见 [WEEK3.md](../../WEEK3.md#b-线性读出与统一评估)。开发集 8 人反复使用，结果只用于第三周内部比较；测试集封存。

## 选模协议变更：开发集只用于最终评估

第二周 `12_eegdm_budget_head.py` 按开发集 Macro-F1 早停，不符合第三周“选参与 checkpoint 选择用训练侧数据”。第三周改用 [27_b_w3_linear_probe.py](../../scripts/27_b_w3_linear_probe.py)：

- 预算内训练者按受试者分 K=min(5,人数) 折（折分配 fold_seed=0，std/mean 共用），每折标准化只拟合折内训练者，固定训练 50 轮。
- 取拼接留出预测 Macro-F1 最高的轮次 E\*，在全部预算内训练者上重训 E\* 轮，对开发集只预测一次。
- 其余配方与第二周相同（Linear(5120,5)、AdamW lr=1e-3 wd=1e-4、batch 128）。重写的训练循环与第二周逐位一致：label25 seed0 恰好也选到第 11 轮，开发集预测与第二周逐段相同。

std 一组因此按新协议重跑，第二周结果只作对照。两者的配对差（新 − 旧，seed 0–2 合并，按受试者 bootstrap 95% 区间）：

| 预算 | ΔMacro-F1 | ΔN1 F1 | ΔN1 AP |
| --- | --- | --- | --- |
| 5 人（label10） | −0.032 [−0.049, −0.012] | −0.120 [−0.176, −0.076] | −0.072 [−0.122, −0.031] |
| 14 人（label25） | −0.004 [−0.010, +0.005] | −0.027 [−0.060, −0.005] | −0.016 [−0.040, +0.008] |

第二周按开发集选轮次的结果偏乐观，5 人预算下尤其明显。第三周所有线性读出（含 mean、t 扫描及 D 的探针）都应使用同一选模协议。

## 进度

| 项目 | 状态 | 结果位置 |
| --- | --- | --- |
| std、t=1，5/14 人 × seed 0–2（6 格） | 完成 | `runs/W3_B_std_t1_*`，[b_w3_summary.csv](b_w3_summary.csv) |
| mean、t=1 特征提取 | 脚本已合并（PR #14），待 C 在 CUDA 上运行 | [b_w3_extraction.md](b_w3_extraction.md) |
| mean、t=1，5/14 人 × seed 0–2（6 格） | 待特征 | |
| t 扫描（std，t=1/10/25，14 人，seed 0） | t=1 完成，t=10/25 待特征 | |
| N1 分数调整（14 人，预先固定 seed 0） | std 完成，mean 待特征 | [b_n1_adjust_summary.csv](b_n1_adjust_summary.csv) |
| 统一评估脚本 | 完成，可登记 A/C/D 的预测 | [29_b_w3_evaluate.py](../../scripts/29_b_w3_evaluate.py) |

## 当前 std 结果（新协议，seed 0–2 均值）

| 预算 | Macro-F1 | N1 F1 | N1 P / R | N1 AP | N1→W / N1→N2 |
| --- | --- | --- | --- | --- | --- |
| 5 人 | 0.701 | 0.269 | 0.335 / 0.229 | 0.275 | 0.330 / 0.367 |
| 14 人 | 0.741 | 0.383 | 0.425 / 0.357 | 0.383 | 0.249 / 0.323 |

N1 分数调整（std，14 人，seed 0）：训练侧留出选出 α=3.52（p_N1 乘 α 后取 argmax）。开发集 N1 P/R/F1 由 0.441/0.309/0.363 变为 0.335/0.580/0.424，Macro-F1 由 0.7449 变为 0.7391。N1 漏报主要可通过移动判定门槛减少，代价是误报增加；原始分数 AP 为 0.373，不受 α 影响。

## A 第三周交付的独立复评（PR #15，2026-11-09）

登记 21 份预测（[b_a_w3_registry.csv](b_a_w3_registry.csv)）：ContraWR dual 9 份（Release `a-contrawr-w3-v01-20261108`）、复用的 W2 CNN dual 9 份与 CNN EEG-only 14 人 3 份。全部通过 `val_reference_v1.csv` 的 sample_id 与 y_true 核对。复算的 18 格均值与 [a_w3_results.md](a_w3_results.md) 逐位一致。

seed 合并后按受试者 bootstrap（A 原表为逐 seed 区间，结论方向一致）：

| 比较 | 预算 | ΔMacro-F1 | ΔN1 F1 |
| --- | --- | --- | --- |
| ContraWR − CNN | 5 人 | −0.017 [−0.043, +0.016] | −0.095 [−0.167, +0.025] |
| ContraWR − CNN | 14 人 | −0.000 [−0.024, +0.027] | +0.034 [−0.056, +0.131] |
| ContraWR − CNN | 54 人 | −0.008 [−0.021, +0.002] | −0.046 [−0.089, −0.019] |
| CNN dual − EEG-only | 14 人 | +0.023 [+0.011, +0.042] | −0.011 [−0.058, +0.045] |

**选模口径不一致：** A 的 CNN 与 ContraWR 都按开发集 Macro-F1 早停（patience=10），B 第三周改为训练侧选模。B 线性读出与 A 基线直接相减会偏向 A，5 人预算尤甚：

| 预算 | B std（训练侧选模）− A CNN | B std（W2 开发集选模）− A CNN |
| --- | --- | --- |
| 5 人 Macro-F1 / N1 F1 | −0.017 / −0.108 | +0.016 / +0.016 |
| 14 人 Macro-F1 / N1 F1 | −0.001 / +0.021 | +0.002 / +0.047 |

5 人预算下“EEGDM 线性读出与 CNN 谁更好”的结论随选模口径翻转，需要 A 按训练侧选模重跑，或全组约定一个统一口径后再比较。B 的 [27_b_w3_linear_probe.py](../../scripts/27_b_w3_linear_probe.py) 内层按受试者选轮次的做法可供参考。

## 统一评估用法

```
python scripts/29_b_w3_evaluate.py --extra <A/C/D 登记 CSV> --pairs mean_t1:std_t1 std_t10:std_t1 std_t25:std_t1
```

登记 CSV 列为 `run_id,group,arm,budget,seed,predictions`，`predictions` 指向 11 列 `predictions_val.csv`。每份预测先按 `manifests/val_reference_v1.csv` 核对 sample_id 与 y_true。配对只在同 budget、同 seed 之间进行。
