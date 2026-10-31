# seed1 全量 EEG 特征重提与分类对照

2026-10-28 21:45:33（Asia/Shanghai），完整后续流程已在 WSL 后台派发。实际阶段与完成状态见 [execution/summary.json](execution/summary.json)；特征重提进度见 [特征 summary](../../features/eeg_cond14_S1_aseed1_steps10000_v01/summary.json)，详细后台日志见 [pipeline_console.log](pipeline_console.log)。

## 自动接续顺序

1. 固定使用seed1第10000步新EMA，为训练/开发验证全部122晚、154124段提取EEG特征；新建独立缓存 `features/eeg_cond14_S1_aseed1_steps10000_v01`。
2. 对完整索引逐行核对原sample_id、受试者、夜晚、epoch_index、划分、标签、文件与行号，校验每晚特征SHA、shape、float16及所有值有限，再用float32对30个时间窗均值池化为2560维。
3. 训练10个新S1分类头：label10/label25 × downstream seed0–4；10份S0分类基线在完整匹配核验后复用。
4. 使用B官方v1评估、按8名验证者聚类的1000次bootstrap，以及每个同预算/同下游种子的S1−S0配对比较。输出汇总均值/标准差和各格差值。

整个接续由同一后台进程执行，特征未通过全量检查时不会进入分类。当前只启动seed1的EEG诊断对照，双模态主比较仍等待D的完整原始EOG缓存。seed2未启动。

## 选择与方法固定

所用权重为 `runs/W2_C_EEG_S1_aseed1_steps10000_v01/selected_ema_step10000.pt`，SHA256：

`c344b5f31f68f89d6f19b31b244770f1c5b02ebdec6507f3d65a94b886b9a94b`

准备阶段验证最终优化器/EMA计数均10000、选定角色与训练源码/配置一致，严格加载全模型，冻结参数和选定参数摘要匹配。固定最终步选模，不根据分期成绩切换主干。

提取规则、预处理、推理缓存、池化与分类训练数值代码均沿用seed0版本；新运行代码只调整适配种子标识及可传入的独立路径。原始训练/提取/分类入口及其绑定保持不变。条件14、t=1、无随机输入噪声、20层gate、1秒std窗口、float32计算/float16存储、batch4；新模型载入并完全冻结后新建自身S4缓存。

标准化分别只用当前预算的5/14名有标签训练者，样本数11936/32310。分类头为Linear(2560,5)、无权重交叉熵、AdamW lr0.001/weight_decay0.0001、batch128、最多50轮、patience10。严格提升开发验证Macro-F1才更新最佳头；N1/REM来自同一个最佳分类头。

## 启动前检查

- [runtime_tests.log](runtime_tests.log)的9项检查通过，包含数值函数与原版本逐行等价、错误适配种子/路径/方法拒绝、S0配置匹配和运行绑定校验。小型合成分类训练实际执行配置保存、优化、选模、权重读取与预测输出，保存在临时目录，不构成真实EEG训练结果。
- [两段数值检查](probe/probe_audit.json)通过：取固定首段训练及首段开发验证信号。新缓存与未缓存最大差0.0；冻结前18层相对原S0 float16缓存最大差0.0001220703125，在预定容差内。此检查不用于评价分期性能。
- [S0复用核验](S0_reuse_audit.json)覆盖10个基线的预算名单、样本数、标签来源、标准化规则、优化/选模参数、种子、源码/评估器/源特征配置；归档SHA及完整验证预测身份通过检查，官方指标重算一致。
- [extraction_binding.json](extraction_binding.json)绑定权重、配置、原始S0索引和各版代码；[pipeline_binding.json](pipeline_binding.json)在全量提取前进一步绑定新入口、测试、数值检查和10份S0基线的配置/预测/指标。
- [comparison_plan.csv](comparison_plan.csv)记录10个新运行和10个复用基线。首个准备目录未派发全量任务，v02是补齐分类入口依赖并增加合成执行检查后的正式启动记录。

## 输出和恢复边界

每个新运行目录为 `runs/W2_C_S1_EEG_aseed1_label10_sseed2024_seed0_v01` 等，保存配置、最佳头及标准化参数、逐轮历史、11列验证预测、B正式指标、混淆矩阵、资源与摘要。

本目录`execution/`随阶段生成`S1_feature_audit.json`、`S0_reuse_audit.json`、`run_registry.csv`、`aggregate_metrics.csv`、`paired_deltas.csv`和10份配对bootstrap。缺失的文件表示阶段尚未完成，不应视为已产出成绩。

同权重/代码/配置的提取中断可在确认原进程已退出后使用相同启动命令追加`--resume`，已经完成的每晚必须先重新校验SHA；并发任务被排他锁拒绝。分类阶段拒绝覆盖既有run目录，不会静默重新训练已存在的分类头。

测试信号不处理；NPZ仅读取signal/epoch_index，特征模型不接收分期标签。原始S0和seed0缓存与成绩保留。当前预计完整流程约100–120分钟，实际以日志为准；五个下游种子共享同一seed1主干，不等于五次独立适配复现。
