# W2_B_CTX1_EEGDM_label25_sseed2024_seed0_v01

状态:**completed**(最后成功步骤:写出 predictions_val.csv 并由官方 evaluate.py v1 复核)。

- 验证 Macro-F1 0.7533;N1 精确率/召回率/F1 = 0.4446 / 0.4422 / 0.4434
- 最佳轮次 9 / 实际 19 轮;更新数 4807
- 有标签受试者:14 人(见 labeled_subjects.csv);标准化只拟合这些人
- 样本集合 SHA256(验证参考清单):2768885c5cddc3ea540f5da8b6c1a487b9bda9e096e6fca48b4acd34cc1ccf43

上下文对照:长度 1 与长度 3 共用同一头、同一中心样本集合;邻居缺失处中心复制(window_index.csv 记录 mask)。
