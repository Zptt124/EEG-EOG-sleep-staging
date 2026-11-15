# 实验A 第三周:W2 CNN run 复用逐项核查(主矩阵 CNN 侧 + EOG 对照)

> **2026-11-09 修订注:** 本核查用于 2026-11-08 初版方案(开发集早停选轮)下的复用。
> 第三周选模协议改为训练侧按人 5 折留出选轮后,开发集只用于最终评估一次;
> W2 run 是开发集口径,不再复用于第三周主矩阵,CNN 侧改为 W3 新训练
> (W3_A_EEGEOG_* / W3_A_EEG_*)。本文件保留为 W2 交付核验记录。

- 负责人/日期:A / 2026-11-08
- 核查对象:12 个 W2 run(发布包 w2_a_runs_v01.zip;
  主矩阵 dual label10/25/100 × seed0-2 共 9 格;EOG 对照 eeg label25 × seed0-2
  共 3 格;dual 一侧由主矩阵复用,EOG 对照合计 6 格)
- 依据:协议v2 旧 run 复用须核对名单、标准化、编码器与特征SHA、头、种子、
  更新预算、选模及评价样本;指标一律 B 官方 evaluate.py --reference 重跑比对
- 官方评估器:src\evaluation\evaluate.py(main 当前版本);
  参考样本集 manifests/val_reference_v1.csv
- 结论:**全部 12/12 通过**
- 明细:[a_w3_reuse_check.csv](a_w3_reuse_check.csv)

## 逐项结果

| run_id | 用途 | 官方重跑 | 指标与zip一致 | 指标与W2汇总表一致 | 样本集一致 | 名单一致 | 预算/种子/超参 | 标准化 |
|---|---|---|---|---|---|---|---|---|
| W2_A_EEGEOG_label10_sseed2024_seed0_v01 | main_grid:dual:label10:seed0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEGEOG_label10_sseed2024_seed1_v01 | main_grid:dual:label10:seed1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEGEOG_label10_sseed2024_seed2_v01 | main_grid:dual:label10:seed2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEGEOG_label25_sseed2024_seed0_v01 | main_grid:dual:label25:seed0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEGEOG_label25_sseed2024_seed1_v01 | main_grid:dual:label25:seed1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEGEOG_label25_sseed2024_seed2_v01 | main_grid:dual:label25:seed2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEGEOG_label100_sseed2024_seed0_v01 | main_grid:dual:label100:seed0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEGEOG_label100_sseed2024_seed1_v01 | main_grid:dual:label100:seed1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEGEOG_label100_sseed2024_seed2_v01 | main_grid:dual:label100:seed2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEG_label25_sseed2024_seed0_v01 | eog_contrast:eeg:label25:seed0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEG_label25_sseed2024_seed1_v01 | eog_contrast:eeg:label25:seed1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W2_A_EEG_label25_sseed2024_seed2_v01 | eog_contrast:eeg:label25:seed2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

指标口径:官方重跑 Macro-F1/N1 F1 与 zip 内 metrics.json 及 reports/week2/a_supervised_comparison.csv 逐位一致(阈值 1e-12);全部 run 的 sample_set_sha256 一致(8 人验证集 21,186 段)。
