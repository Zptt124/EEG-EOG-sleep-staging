# 实验 D｜候选片段筛选与标注策略（C、D 双人任务书，2026-11-28）

D 在任务书中的工作：
- 共享随机排序与选块锁定；
- 精度估算；
- 跨机抽查；
- 对 C 的 780 次线性头主拟合和 520 次 BiGRU 拟合做独立复算。

| 内容 | 位置 |
| --- | --- |
| 条件 B 的睡眠边界（与旧窗口裁剪规则相同） | `01_sleep_boundaries.py`，`prep/sleep_boundaries.csv` |
| 调试版抽样（已作废，仅保留脚本） | `02_shared_order_sampler.py` |
| 精度估算 | `03_precision_estimate.py`，`prep/precision_estimate.csv`，`prep/D_prep_results_20261128.md` |
| **锁定选块包 v01**（基于 C 的正式交接表） | `04_lock_selection.py`，`selection_lock_v01/`（说明见 `README_D_to_C.md`） |
| **独立复算与跨机抽查** | `05_independent_recompute.py`，`recompute_v01/`（报告见 `D_verification_report_20261128.md`） |

主要结果（固定尝试 600 段，26 份名单 × 5 次重复，16 位开发评估者，head seed 0）：

- 跨机抽查 6/6 通过，逐段预测 100% 一致；
- 线性头 ΔB−ΔA = −0.21 [−0.60, +0.15] pp，ΔB−ΔC = −0.05 [−0.43, +0.36] pp，两者都落在 ±1 pp 实践带内；
- BiGRU ΔB−ΔA = −0.01 [−0.63, +0.62] pp。

在同一关灯网格上单独加入分期边界筛选，对"新人 − 混合随机"的影响在实践带以内。

`D_spotcheck_predictions.npz` 和锁定包 zip 放在 Release `d-filter-exp-v01-20261128`。C 的原始结果归档见 Release `cd-screening-handoff-v01-20261128`。历史预留 8 人与封存 16 人均未使用。
