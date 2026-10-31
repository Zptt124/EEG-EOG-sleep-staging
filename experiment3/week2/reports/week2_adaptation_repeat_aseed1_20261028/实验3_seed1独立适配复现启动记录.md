# seed1 独立适配复现

2026-10-28 20:08:51（Asia/Shanghai）在 WSL 后台启动。实时状态见 [progress.json](../../runs/W2_C_EEG_S1_aseed1_steps10000_v01/progress.json)，后台日志见 [formal_console.log](formal_console.log)。实际完成状态以运行目录的 summary.json 为准。

## 选择这一项的依据

seed0 适配、全量 EEG 特征、20 格 EEG 诊断对照及结果校验均已完成；双模态主比较仍缺 D 的完整 EOG 缓存。原锁定配置已预登记独立适配种子1、2，触发依据是先导材料齐备和资源可用，不是正向分数。本次推进 seed1：启动前 GPU 利用率0%，D盘可用约166.96 GiB。

seed0 的 EEG 结果保持原样，无论正负均报告。本轮没有根据10%下小幅正差或25%基本持平而修改方法。seed2尚未启动。

## 方法和独立性

使用与seed0完全相同、SHA不变的 `train_sleep_adaptation.py`。新配置只改变三个随机种子（适配/数据/噪声均为1）、当前复现种子及独立配置名、运行名、特征目的路径；原始训练配置和锁不覆盖。新配置为 [adapt_W2_C_S1_aseed1_v01.yaml](../../configs/adapt_W2_C_S1_aseed1_v01.yaml)，来源和精确允许差异见 [derivation_audit.json](derivation_audit.json)。

从原始预训练EMA重新加载。54名训练者的无标签EEG，末两层和输出层527233个参数；batch2、FP32、AdamW lr1e-5、固定10000次成功优化、新EMA beta0.999逐步更新、固定选取step10000新EMA。分期标签、验证和测试信号不进入适配训练。既有seed0结果和权重均不作为初始化或恢复来源。

独立目录：`runs/W2_C_EEG_S1_aseed1_steps10000_v01`。未来seed1特征预留目录为 `features/eeg_cond14_S1_aseed1_steps10000_v01`，当前尚未提取。

## 启动前核验

- 6项新的派生检查通过：拒绝未登记种子、超参数和步数变化、seed0输出覆盖、继承适配权重/优化器计数和验证/标签访问。
- 沿用原训练器的8项CPU测试证据；代码、测试和原测试日志SHA完全一致，未重复运行相同测试。新派生检查日志为 [derivation_tests.log](derivation_tests.log)。
- [零步初始化验证](initialization_validation.json)通过：优化器和新EMA计数均0，模型/新EMA的初始参数摘要与原始EMA完全一致，冻结参数摘要一致，无NPZ信号或标签读取。
- 新配置SHA `6d04861ba9fab8237fd12e492c9f797cdff5685ded8f7185ff98e512f686e9d2`，32个锁定输入；训练器SHA `a3f8e87b1632a88507c8968c9c19cf0c46f16d066b51e2f7ad0aa402d07445f2`。
- [运行绑定](runtime_binding.json)在首次更新前保存，后台派发信息见 [launch.json](launch.json)。首批实际更新核验单独记录在 `launch_audit.json`。

按seed0的实际计算时间，适配预计约1.5–2小时；系统暂停会延长墙钟时间。每100步刷新进度，每1000步保存完整恢复状态，保留最近两个恢复点。完成并验收固定新EMA后，再准备seed1特征重提及对应对照；当前没有seed1分期结果。
