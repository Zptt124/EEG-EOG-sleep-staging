# 阶段交接登记

**2026-10-31 C三个补充实验已完成：** [结论与原始证据](experiment3/week2/SUPPLEMENTS.md)包含固定50轮40头、类别加权40头、取层消融80新头，合计160份完整验证预测；全部审计通过。加权40/40配对的N1 F1改善，末5层有一般性分类收益，整体仍未证明稳定的适配收益。均为同一开发验证集上的探索性补充，测试集封存。

**C→B 特征下载补充（2026-10-31）：** [EEG/EOG 两份完整池化 NPZ](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-pooled-features-v01-20261031)已发布到 GitHub Release；各154124×2560、float32、未标准化，键为sample_id和x，附样本索引、SHA和B接口核验。EEG为原始冻结S0，EOG为C按D同协议本地重建。

**2026-10-31 C第二周交付：** [实验③结果与复评入口](experiment3/week2/README.md)已备妥3轮适配、80份验证预测、配置/历史/指标及N1分析；10%双模态平均退步、25%基本持平，未证明稳定适配收益。B已发布v2名单核验和80份预测复评；D共同EOG来源认可及实际接收验收仍待补；测试集封存。第一周C材料保留在[原目录](experiment3/README.md)。


第二周以[工作文档](docs/week2/work_plan_W2_v1.1.pdf)、[WEEK2清单](WEEK2.md)及[协议](docs/week2/protocol_v2.md)为准。完成即滚动交接，不等待所有run或全组同阶段。交付记录写commit、版本/SHA、run_id、存储位置、接收人及接收核验；PR中的链接不代表对方已经收到或验收。

## 第二周交接条件

| 条件 | 交付方→接收方 | 产物与用途 | 当前可核查状态 |
| --- | --- | --- | --- |
| 名单/协议核验后 | B→全组 | v2名单、夜晚/片段/类别数、协议 | B已发布21/21核验记录，见reports/week2/b_v2_verification.md；旧v1保留 |
| 首批编码器可加载后 | A→B/D | 10/25%单模态及25%双通道seed0，模型、标准化、维度、加载例子 | W2接口尚未在核查分支中确认 |
| 随后滚动 | A→B/D | 各预算其余seed编码器及11列预测 | 待登记；不能只用seed0代替全部重复 |
| 无需等待新训练 | 现有C/D→B/D | 第一周原始EEG/EOG池化特征、索引、SHA和团队存储 | 固定commit入口见下表；大文件仍需实际接收核验 |
| 首个适配完成后 | C→B | S1权重SHA、新特征、配置、逐段预测 | C已发布3轮适配SHA、80份预测与索引，见experiment3/week2；两份S0池化NPZ已发布Release，B已复评80份预测；特征实际接收及D来源认可待补 |
| 每批run完成 | A/C/D→B | 11列预测、配置、run索引、资源/失败记录 | 滚动填写reports/week2/run_registry.csv |
| 汇总阶段 | B→全组 | 预算曲线、上下文、配对比较、缺项 | 空表已定义，尚无W2结果行 |

## 第一周已发布入口（2026-10-27快照）

| 交付 | 固定版本与位置 | 合并/接收说明 |
| --- | --- | --- |
| A划分、153个NPZ、CNN、预测 | main `02f120ee80c1c7a7822c33746ce92017632f1faa`；`manifests/`、`scripts/`、`交付包/实验①交付包_20261018/`、`reports/run_index.md` | 已在main；54/8/16，旧单次CNN不自动满足W2矩阵 |
| B官方evaluate.py、名单、少标签结果 | [6e7df99](https://github.com/Zptt124/EEG-EOG-sleep-staging/tree/6e7df9978c48c84bbef4d2f7e9d66e9ccd93b923)，[PR #1](https://github.com/Zptt124/EEG-EOG-sleep-staging/pull/1) | 已发布未合并；main仍为A临时评价脚本 |
| C冻结EEG特征、100/25%结果 | [b6ee028/experiment3](https://github.com/Zptt124/EEG-EOG-sleep-staging/tree/b6ee02852d379cc2ee748d817cb421de619cac89/experiment3)，[PR #2](https://github.com/Zptt124/EEG-EOG-sleep-staging/pull/2) | draft未合并；不等于主干适配已完成 |
| D EOG特征、多种子融合与N1分析 | [f00a16b/experiment4](https://github.com/Zptt124/EEG-EOG-sleep-staging/tree/f00a16ba73bdabed65c819a9bfd2d9e7fe6198ed/experiment4)，[PR #3](https://github.com/Zptt124/EEG-EOG-sleep-staging/pull/3) | 已发布未合并；不等于D0/D1/D2已完成 |

## 编码器/特征接收核验

- 输入模态、预算及人员名单、抽样/训练seed、代码与权重SHA全部匹配。
- 100Hz输入为`[B,C,3000]`，每通道单位与标准化可复现；EEGDM另记录200Hz及其变换。
- 输出维度、dtype、eval模式、sample_id索引和批推理例子可用；单模态拼接与双通道CNN编码器分开。
- 内折留出时重训监督编码器与统计；大文件位置及SHA可访问并实际核验后再登记received/verified。
- 预测样本覆盖、唯一ID、标签及概率检查通过；失败项如实保留，不伪造指标。

人员record_ids用分号连接并按CSV规则转义。模板仅表头，不能当真实数据；`reports/week2/README.md`解释所有新表字段与状态。
