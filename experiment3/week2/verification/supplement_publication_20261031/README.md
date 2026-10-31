# 三个补充实验上传复核

160份完整预测、50轮历史、训练名单、类别权重、取层配置及B固定v1点指标复算通过；160份本地外部checkpoint的SHA、配置、形状和标准化参数核对通过。结果见prediction_audit.json。3+6+8共17项原实验测试在发布目录中运行通过，见checks.json及日志。

测试使用本机已固定的vendor依赖，未向Git添加vendor、模型权重或大数组。复核不执行新真实训练，不选择新模型，不读取测试数据。原始源码/证据副本逐字节保留，详见supplement_source_snapshot.json。原报告中的“尚未同步GitHub”为计算完成时历史状态，当前入口为SUPPLEMENTS.md和publication_status.json。
