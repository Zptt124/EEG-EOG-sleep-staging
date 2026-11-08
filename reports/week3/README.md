# reports/week3 — 第三周实验A 结果与登记

- 负责人:A;日期:2026-11-08 起。
- 指标口径:一律 B 官方 `src/evaluation/evaluate.py --reference manifests/val_reference_v1.csv`;
  8 人开发集反复使用,按受试者配对 bootstrap(官方 --compare,n=1000,95%)。
- 各 CSV 全列按文本存储;`NA` 表示不适用,不留默认值;`sample_set_sha256` 必须一致才能配对比较。

## 文件

| 文件 | 内容 |
| --- | --- |
| [a_baseline_selection.md](a_baseline_selection.md) | ContraWR / ST-Transformer 核查、选择依据、适配决定、文献协议对照 |
| [a_w3_reuse_check.md](a_w3_reuse_check.md) / .csv | 12 个 W2 CNN run 复用逐项核查(官方重跑比对) |
| [a_w3_main_grid.csv](a_w3_main_grid.csv) | 18 格主矩阵:方法×预算(5/14/54人)×seed(0-2),Macro-F1/N1 F1/各类 F1/N1 P/R |
| [a_w3_paired_deltas.csv](a_w3_paired_deltas.csv) | 配对差 ContraWR − CNN(逐 seed + 按人配对 CI) |
| [a_w3_eog_contrast.csv](a_w3_eog_contrast.csv) | EOG 对照 dual − eeg(label25,seed0-2) |
| [a_w3_n1_rem_errors.csv](a_w3_n1_rem_errors.csv) | N1/REM 错误去向(逐 seed 混淆行占比,主矩阵与 EOG 对照两侧) |
| [by_subject/](by_subject/) | 每个比较的逐人 Macro-F1 与差值 |
| [budget_curves_w3.png](budget_curves_w3.png) | 5/14/54 人 Macro-F1 与 N1 F1 曲线(3 seed 均值,竖线=seed 范围) |
| [run_registry.csv](run_registry.csv) | 9 个 W3 ContraWR run(completed)+ 12 个复用 W2 run(reused);列同 reports/week2/run_registry.csv |
| [metrics_summary.csv](metrics_summary.csv) | 同口径指标汇总(列同 reports/week2/metrics_summary.csv) |
| [resource_profile.csv](resource_profile.csv) | 资源记录(参数/耗时/显存/吞吐;列同 reports/week2/resource_profile.csv) |
| [paired_comparison.csv](paired_comparison.csv) | 配对比较登记(comparison − reference,uncertainty_unit=subject) |
| [a_w3_results.md](a_w3_results.md) | 结果叙述、种子波动与限制(汇总脚本产出后补写) |

## run_id 约定(第三周 A)

- 新训练:`W3_A_CONTRAWR_<budget>_sseed2024_seed<N>_v01`(modalities=dual;
  model 名替代 W2 的 modality 槽位,因第三周同模态存在两个模型,须在 id 中区分)
- 复用:沿用 W2 run_id(W2_A_EEGEOG_* / W2_A_EEG_*),status=reused,
  reuse_check_path=reports/week3/a_w3_reuse_check.md

## 实验B 文件(B,2026-11-07 起)

状态与结论见 [b_status_w3.md](b_status_w3.md)。B 第三周线性读出改为训练侧选模(预算内按受试者内层交叉验证选轮次),开发集只用于最终评估。

| 文件 | 内容 |
| --- | --- |
| [b_status_w3.md](b_status_w3.md) | 进度、选模协议变更、std 结果、N1 分数调整、A 交付复评与选模口径提醒 |
| [b_w3_extraction.md](b_w3_extraction.md) | mean pooling 与 t 扫描特征提取说明(C 在 CUDA 机器上运行) |
| [b_eval_registry_w3.csv](b_eval_registry_w3.csv) | 统一评估登记(B 自动登记 + 下列外部登记) |
| [b_a_w3_registry.csv](b_a_w3_registry.csv) | A 的 21 份预测登记;ContraWR 预测来自 Release `a-contrawr-w3-v01-20261108`,解压到本地 `features/a_w3_v01/extracted/` |
| [b_w2_devselected_registry.csv](b_w2_devselected_registry.csv) | 第二周按开发集选模的 std 线性头(对照用) |
| [b_w3_metrics.csv](b_w3_metrics.csv) | 每个 run:Macro-F1、各阶段 F1、N1 P/R/F1、N1 AP/AUROC、N1 去向与来源 |
| [b_w3_summary.csv](b_w3_summary.csv) | group×arm×budget 的 seed 数与均值/最小/最大 |
| [b_w3_paired.csv](b_w3_paired.csv) / [b_w3_paired_by_subject.csv](b_w3_paired_by_subject.csv) | 配对差(seed 合并、按受试者 bootstrap)与逐人差 |
| [b_n1_adjust_summary.csv](b_n1_adjust_summary.csv) / [b_n1_adjust_by_subject.csv](b_n1_adjust_by_subject.csv) | N1 分数调整对照(14 人,seed 0) |

脚本:`scripts/26_b_extract_eegdm_w3.py`(提取)、`27_b_w3_linear_probe.py`(线性探针)、`28_b_n1_score_adjust.py`(N1 分数调整)、`29_b_w3_evaluate.py`(统一评估)。B 的 run 为 `W3_B_<arm>_<budget>_sseed2024_seed<N>_v01`,arm 如 `std_t1`、`mean_t1`、`std_t10`。
