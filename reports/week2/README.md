# 第二周小文件与登记

本目录的CSV仅定义交接字段；除`task_status.csv`外初始均为表头，不包含实验结果。完整run按[RUN_TEMPLATE](../../RUN_TEMPLATE.md)在团队存储保存。共同要求见[protocol_v2](../../docs/week2/protocol_v2.md)。

| 文件 | 维护方 | 用途 |
| --- | --- | --- |
| task_status.csv | 全组 | 本次核查证据与未核验项，不代表成员本地进度 |
| run_registry.csv | B汇总、各组填报 | 每次实际run一行；旧run复用记录原run与逐项核验依据 |
| metrics_summary.csv | B | 同一最佳权重的Macro-F1、各类F1、N1三指标、按人产物路径 |
| paired_comparison.csv | B | 同样本/同受试者配对差，种子波动与按人区间分开 |
| resource_profile.csv | 各组 | 独立设备/精度/参数/耗时/显存/更新数，教师和校准单独计费 |
| encoder_index.csv | A | 预算/seed匹配编码器、标准化、维度、权重和存储 |
| artifact_index.csv | 各组 | 原始/适配主干、教师、特征、运行大产物的团队位置与SHA |
| window_index.template.csv | B | 中心与前后sample_id、mask、原epoch_index；模板无真实窗口 |
| pseudo_label_trace.template.csv | D | 所有候选、拒绝项及权重；没有真实未标注标签字段 |

`run_registry`的`status`使用`planned/running/incomplete/failed/completed/reused`。`completed/reused`须有完整六件套、预测核验、正确选模与数据权限证据；已启动或有表头不算完成。`task_status`的`not_verified`表示核查分支未找到第二周完成证据，不能解释为本人未开工。

`nominal_budget`填label10/25/50/100，人数填实际5/14/27/54。`unlabeled_signal_subjects`填写该方法实际额外使用无标签信号的人数：监督或D0填0，C/S1填54（与有标签人员重叠，不能相加作独立人数），D1/D2填49或40。D即使伪标签全部被拒，教师推理已使用该池信号，仍应记录池规模，接收用量另记。适配、教师、下游seed分列，不适用填NA，不留作默认0。

所有路径均为仓库相对路径或团队存储URI，不填私人绝对路径、访问令牌或伪造地址。SHA填完整SHA256；代码commit另用Git SHA。`sample_set_sha256`基于按sample_id排序后的共同中心清单，写明清单文件，比较前必须相同。

配对表`delta_*`统一comparison减reference，`uncertainty_unit`写subject；`training_seed_summary`写五seed波动的独立汇总位置，不把种子×人当独立样本。区间含0不代表等效，当前8人仅开发集证据。

窗口字段`*_is_real`用0/1：复制中心填充时为0，中心本身始终真实；不得跨夜、跨人或跨缺口。伪标签`accepted`用0/1，`weight`为实际学生权重，拒绝项也保留。缺少额外交付（如适配loss、阈值内折、N1错向等）时按[WEEK2.md](../../WEEK2.md)增添对应真实产物，不由这些索引表代替。
