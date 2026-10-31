# EEG–EOG 睡眠分期研究

**2026-10-31 C三个补充实验已完成：** [结论与原始证据](experiment3/week2/SUPPLEMENTS.md)包含固定50轮40头、类别加权40头、取层消融80新头，合计160份完整验证预测；全部审计通过。加权40/40配对的N1 F1改善，末5层有一般性分类收益，整体仍未证明稳定的适配收益。均为同一开发验证集上的探索性补充，测试集封存。

**C→B 特征下载补充（2026-10-31）：** [EEG/EOG 两份完整池化 NPZ](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-pooled-features-v01-20261031)已发布到 GitHub Release；各154124×2560、float32、未标准化，键为sample_id和x，附样本索引、SHA和B接口核验。EEG为原始冻结S0，EOG为C按D同协议本地重建。

**2026-10-31 C第二周交付：** [实验③结果与复评入口](experiment3/week2/README.md)已备妥3轮适配、80份验证预测、配置/历史/指标及N1分析；10%双模态平均退步、25%基本持平，未证明稳定适配收益。B已发布v2名单核验和80份预测复评；D共同EOG来源认可及实际接收验收仍待补；测试集封存。第一周C材料保留在[原目录](experiment3/README.md)。


利用预训练EEG表征与未标注EEG/EOG，研究少标签睡眠分期和N1识别。

## 当前阶段：第二周 W2 v1.1

本轮依据2026-10-26修订的[第二周工作文档](docs/week2/work_plan_W2_v1.1.pdf)，按阶段与交接条件推进，不对应固定日期。第一周已经得到冻结特征及融合的初步结果；尚未证明EEGDM稳定优于公平CNN，也未解决N1的稳定识别。

- [WEEK2.md](WEEK2.md)：A/B/C/D任务矩阵、推进顺序与完成标准。
- [公共协议v2](docs/week2/protocol_v2.md)：数据权限、预算、选模、上下文和评价规则。
- [交接登记](HANDOFF.md)、[运行模板](RUN_TEMPLATE.md)与[协作规则](CONTRIBUTING.md)。
- [预算名单](manifests/label_subsets_v2.csv)、[人数/片段/类别统计](manifests/label_counts_v2.csv)：沿用B的seed2024排序，仅新增前5人的label10；[来源与复核说明](docs/week2/budget_provenance.json)。名单已由B完成21项核验，见[核验记录](reports/week2/b_v2_verification.md)；C已完成的本地实验见上方交付入口。
- [GitHub核查与更新记录](docs/week2/github_audit.md)、[任务状态](reports/week2/task_status.csv)、[运行登记](reports/week2/run_registry.csv)。公共主表待B统一登记；C的80份运行索引见交付入口。

| 成员 | 第二周职责 | 优先产物 |
| --- | --- | --- |
| A | 公平监督CNN：双通道四预算、单模态三预算，各seed0–4 | 10%/25%单模态及25%双通道seed0编码器与标准化接口 |
| B | 四档监督曲线、25%长度1/3上下文、全组统一评估 | v2名单、统计、协议，随后滚动核验预测 |
| C | 训练池无标签睡眠域适配：S0/S1、10%/25%、两模态设置 | 数据隔离、主干梯度与测速，再交适配权重和新特征 |
| D | 两表征、两预算的D0监督/D1普通自训练/D2跨模态自训练 | 原始冻结EEGDM的25% D0/D1先启动，不等待C |

共享划分为**54训练/8开发验证/16封存测试**；预算按人计数：5/14/27/54。正式下游种子0–4，按验证Macro-F1选同一权重并报告N1 precision/recall/F1，早停patience=10。监督仅使用预算内人员信号；C/D额外使用无标签信号时单列人数。详情以公共协议为准。

## 第一周交付在哪里

截至2026-10-27核查，main含A的交付；B/C/D已在独立分支和PR交付，仍未合并。分支有交付不等于第二周矩阵已完成，旧结果须逐项匹配后登记复用。

| 成员 | 固定交付commit | 入口 | PR状态（核查快照） |
| --- | --- | --- | --- |
| A | `02f120ee80c1c7a7822c33746ce92017632f1faa` | [数据与运行记录](reports/run_index.md) | 已在main；A分支也保留第一周交付，不自动替换共同划分 |
| B | `6e7df9978c48c84bbef4d2f7e9d66e9ccd93b923` | [官方评估、名单与复盘](https://github.com/Zptt124/EEG-EOG-sleep-staging/tree/6e7df9978c48c84bbef4d2f7e9d66e9ccd93b923) | [PR #1](https://github.com/Zptt124/EEG-EOG-sleep-staging/pull/1)，open |
| C | `b6ee02852d379cc2ee748d817cb421de619cac89` | [冻结EEG表征与结果](https://github.com/Zptt124/EEG-EOG-sleep-staging/tree/b6ee02852d379cc2ee748d817cb421de619cac89/experiment3) | [PR #2](https://github.com/Zptt124/EEG-EOG-sleep-staging/pull/2)，draft |
| D | `f00a16ba73bdabed65c819a9bfd2d9e7fe6198ed` | [EOG、融合、多种子与错误分析](https://github.com/Zptt124/EEG-EOG-sleep-staging/tree/f00a16ba73bdabed65c819a9bfd2d9e7fe6198ed/experiment4) | [PR #3](https://github.com/Zptt124/EEG-EOG-sleep-staging/pull/3)，open |

[WEEK1.md](WEEK1.md)保留第一周清单；旧勾选与“待交付”是当时记录，当前入口以上表为准。历史数据及结果见[第一周报告](reports/结果报告.md)和[文件清单](文件清单.md)。

## 获取与复核预算

```bash
git clone https://github.com/Zptt124/EEG-EOG-sleep-staging.git
cd EEG-EOG-sleep-staging
python scripts/build_week2_budgets.py --output-dir /tmp/w2-budget-check
python -m unittest discover -s tests -p 'test_week2_budgets.py'
```

脚本默认仅使用仓库内受试者清单及训练夜晚类别计数，不访问信号或测试标签。需从A已核验NPZ重建计数时，安装NumPy并使用`--processed <NPZ目录>`，只读取54人训练记录的`labels`。输出目录不可已有同名文件；比较CSV后交B核验。完整说明见[协议](docs/week2/protocol_v2.md)。Windows将输出路径换成空目录。训练仍由各组在独立设备实施，不因同步协作文档而启动训练。

## 文件与大产物

`manifests/`保存划分和名单；`scripts/`保存入口；`reports/week2/`保存小表及登记。完整run在团队存储保存六件套；权重、原始信号及特征不新增进Git，仅登记位置、SHA256、索引及版本。第一周已有分发包保留，新适配缓存不能覆盖S0。

## 资料

- [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
- [EEGDM代码](https://github.com/jhpuah/EEGDM)与[权重](https://huggingface.co/jhpuah/eegdm)：使用已核验版本，不自动追随上游。

本轮不启动HMC/ISRUC，不切换62人CV。尚未指定开源许可证；第三方代码须核查许可并保留来源。
