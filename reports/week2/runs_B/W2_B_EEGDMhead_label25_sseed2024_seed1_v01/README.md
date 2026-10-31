# W2_B_EEGDMhead_label25_sseed2024_seed1_v01

状态:**completed**(最后成功步骤:写出 predictions_val.csv 并由官方 evaluate.py v1 复核)。

- 验证 Macro-F1 0.7433;N1 精确率/召回率/F1 = 0.4748 / 0.3740 / 0.4184
- 最佳轮次 4 / 实际 14 轮;更新数 3542
- 有标签受试者:14 人(见 labeled_subjects.csv);标准化只拟合这些人
- 样本集合 SHA256(验证参考清单):2768885c5cddc3ea540f5da8b6c1a487b9bda9e096e6fca48b4acd34cc1ccf43

原始冻结 EEGDM 特征;本 run 只用预算内有标签者的信号与标签,其余训练者未读取。
