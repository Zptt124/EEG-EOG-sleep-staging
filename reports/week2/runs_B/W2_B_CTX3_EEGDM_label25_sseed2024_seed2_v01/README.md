# W2_B_CTX3_EEGDM_label25_sseed2024_seed2_v01

状态:**completed**(最后成功步骤:写出 predictions_val.csv 并由官方 evaluate.py v1 复核)。

- 验证 Macro-F1 0.7716;N1 精确率/召回率/F1 = 0.4591 / 0.5093 / 0.4829
- 最佳轮次 4 / 实际 14 轮;更新数 3542
- 有标签受试者:14 人(见 labeled_subjects.csv);标准化只拟合这些人
- 样本集合 SHA256(验证参考清单):2768885c5cddc3ea540f5da8b6c1a487b9bda9e096e6fca48b4acd34cc1ccf43

上下文对照:长度 1 与长度 3 共用同一头、同一中心样本集合;邻居缺失处中心复制(window_index.csv 记录 mask)。
