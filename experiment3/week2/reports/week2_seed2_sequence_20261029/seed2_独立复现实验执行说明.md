# seed2 独立复现及自动接续

2026-10-29 00:12:52（Asia/Shanghai）派发seed2训练，00:13开始实际运行；随后启动独立后台接续进程。整体状态见 [summary.json](summary.json)，实际训练步数见 [training progress](../../runs/W2_C_EEG_S1_aseed2_steps10000_v01/progress.json)。

seed1已于2026-10-28 23:35完成全量特征和10个新分类头，并在29日通过独立归档/预测/指标核验。当前按原配置预登记的seed2继续复现，不根据seed0/1分数调整超参数。seed0、seed1结果保留；seed2是原计划最后一个适配种子。

## 顺序

1. 原始预训练EMA重新初始化，适配/数据/噪声种子均2；54人无标签训练EEG，固定10000次优化、10000次新EMA更新。超参数与seed0/1相同，主干训练源码SHA保持不变。
2. 训练完成后独立核验连续10000步、loss/梯度有限、EMA计数、最终权重文件SHA、冻结参数摘要及训练数据访问规则。
3. 完整核验通过后，自动为122晚/154124段提取seed2 EEG特征，核验并池化，训练10%/25%各5种子的10个新分类头，与10个已匹配的S0基线配对分析。

任何阶段失败即保留日志并停止接续，不自动改变配置或忽略缺失结果。后续代码、配置、测试证据与原始训练派发信息已在 [continuation_binding.json](continuation_binding.json)绑定，实际接续由 `scripts/continue_seed2_after_training.py`执行。

## 来源和验证

训练配置：[adapt_W2_C_S1_aseed2_v01.yaml](../../configs/adapt_W2_C_S1_aseed2_v01.yaml)，SHA为 `db16fe27723690db1af0fc5b330da6cda5428709d2028e8ab5ec3781c6748546`。派生仅改变三个种子、当前种子和独立配置/运行/特征路径。零步初始化通过，模型和新EMA参数摘要与原始EMA完全一致，优化器和新EMA计数均0。

训练材料目录沿用原登记命名：`reports/week2_adaptation_repeat_aseed2_20261028`，实际时间以launch/summary为准。训练器与原8项测试、派生检查6项及后续特征/分类10项检查的源码均未变化，原通过日志按SHA复用；本次新特征配置额外验证适配种子、来源配置SHA和独立输出路径。

后续特征配置：[features_W2_C_S1_aseed2_v01.yaml](../../configs/features_W2_C_S1_aseed2_v01.yaml)。特征路径为 `features/eeg_cond14_S1_aseed2_steps10000_v01`，结果和阶段日志会写入 `reports/week2_seed2_features_heads_20261029`。准备/探测/绑定/全量执行分别完成后才进入下一阶段，固定最终新EMA的数值探测不用于选模。

## 时间与范围

适配预计约1.5小时；完整训练、特征提取、池化及分类约3–4小时，以实际日志为准。全程测试集不使用；适配不读取分期标签、验证或测试信号。双模态主比较仍等待D的完整原始EOG缓存，本流程是EEG诊断复现，不包含EOG融合。
