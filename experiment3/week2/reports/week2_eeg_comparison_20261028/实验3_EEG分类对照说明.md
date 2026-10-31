# 第二周 S0/S1 EEG 分类对照

本阶段比较原始主干 S0 与固定 10000 步新 EMA 主干 S1 的 EEG 特征。EEG-only 是诊断实验；双模态主比较等待 D 提供原始冻结 EOG 的完整缓存和校验信息。当前状态以 [execution/summary.json](execution/summary.json) 为准。

## 执行边界

- 仅使用训练和开发验证的 122 晚、154124 段；测试集保持封存。
- 完整索引逐行核对 sample_id、subject_id、record_id、原 epoch_index、划分、标签、文件和行号；拒绝缺项、重复、重排、测试数据与跨目录路径。
- 两份原始缓存逐文件检查 shape `[N,20,30,128]`、float16、所有值有限并记录 SHA。S1 每晚的 SHA 和选定新 EMA 的 SHA 必须与提取收据一致。
- 固定在 float32 中对 30 个时间窗求均值，保留全部 20 层，展平为 2560 维；池化文件是派生缓存，原始 S0/S1 缓存保留。
- 原适配配置及其锁定输入保持不变。新入口、测试、评估器、两个索引、特征配置、名单和矩阵在首次头部训练前绑定精确 SHA，见 [runtime_binding.json](runtime_binding.json)。

## 预定 20 格

S0/S1 × label10/label25 × downstream seed0–4。名单固定 sampling_seed=2024，v2 的前 5/14 人，训练段数为 11936/32310。每一个种子依次训练 S0 和 S1，确保同维度初始线性头与初始样本置乱随机状态一致；所有预定位置均执行，不按中途成绩增删。

每个运行独立用当前预算的有标签训练者拟合均值与标准差，禁止将其余训练者或开发验证数据用于标准化。std 下限 1e-6。Linear(2560,5)、无权重交叉熵、AdamW lr=0.001/weight_decay=0.0001、batch128、最多50轮、patience10。严格提升开发验证 Macro-F1 才替换最佳头；N1 和 REM 均从这一份最佳头计算。CPU 4线程，确定性算法开启；不更新主干。

每个 run 保存配置、逐轮历史、最佳头与标准化参数、11列逐段验证预测、B v1 正式指标、混淆矩阵、资源记录和小结。B 指标含按 8 名验证者聚类的 bootstrap 1000次，并生成同预算/同种子的 S1−S0 配对差。五个下游种子共享一个适配主干，不能把它们报告成五次独立适配复现；验证成绩也不是封存测试集成绩。

## 验证与状态

[runtime_tests.log](runtime_tests.log) 的 10 项测试通过：覆盖完整性拒绝、标签/顺序错位、测试与越界路径、预算和标准化隔离、float32池化、严格选模、配对初始化、SHA损坏及运行绑定篡改。

后台日志见 [comparison_console.log](comparison_console.log)。特征审计见 `execution/S0_feature_audit.json` 和 `execution/S1_feature_audit.json`；每格结果见 `execution/run_registry.csv`，总体均值/标准差见 `execution/aggregate_metrics.csv`，配对差见 `execution/paired_deltas.csv`。这些文件随实际进度生成，尚未生成不能视为已完成。

本任务不会自动发布 GitHub，也不会发送材料给其他协作者。后续双模态仍须先完成 D 的 EOG 来源与 sample_id 对齐验收。
