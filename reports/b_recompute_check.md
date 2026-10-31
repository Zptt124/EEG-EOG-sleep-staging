# B 官方 evaluate.py 对 A 的 E1 结果的重算核对

- 日期:2026-10-22 · 脚本:`src/evaluation/evaluate.py` v1
- 输入:交付包 `3_验证集预测_给D\predictions_val_{EEG,EOG}.csv`(sha256 见 `run_index.md`)
- 输出:`metrics_official_E1_{EEG,EOG}_label100_seed0_v01.json`(含按受试者 bootstrap)

| run_id | A 临时版 Macro-F1 | B 官方版 Macro-F1 | 混淆矩阵 | accuracy / 每类 P·R·support |
|---|---|---|---|---|
| E1_EEG_label100_seed0_v01 | 0.7645 | 0.7645 | 完全相同 | 完全相同 |
| E1_EOG_label100_seed0_v01 | 0.6827 | 0.6827 | 完全相同 | 完全相同 |

每类 F1 最大差异 1.1e-16(浮点舍入)。结论:A 的 E1 分数**无需更改**,此后以官方版为准。
