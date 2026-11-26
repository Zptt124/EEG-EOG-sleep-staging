# EEG–EOG 睡眠分期研究

## 当前阶段 · 新一轮第 2 周（2026-11-27 任务书）

**多标一个人，还是多标一夜？** 第一周 10 份名单上，同样新增 600 段标签，「新人新夜」比「原人原夜」的 Macro-F1 在 36 人主评中平均高 **1.58 pp**，8/10 份同向，原 dev 8 人上高 **1.06 pp**。本周加入「原人另一夜」，分别估计在原受试者上多标一夜的收益，以及同样增加 5 夜时换成新人的额外收益。

[第一周 C 结果与预测](experiment3/week1_budget_20261126/results/README.md) · [第一周 D 结果与旧头](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-budget-week1-results-v01-20261126) · [共享特征与指纹](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w6-shared-cache-v01-20261126)

- **表一｜复验第一周结论。** 对原 01–10 的 10 份名单，在 W4 预留 8 人上比较「新人新夜 − 原人原夜」。复用 20 个旧分类头及各头原有的标准化参数，统一推理。
- **表二｜拆分人和夜的收益。** 在 01–04、06–10 的 9 份名单上，各加拟合一个「原人另一夜」头。原 5 人各从另一夜取 2 块，逐人逐块匹配「原人原夜」新增块所在的夜间三分位。三组都保留原 1,200 段起点并新增 600 段，比较「原人另一夜 − 原人原夜」和「新人新夜 − 原人另一夜」。05 号起点成员 SC452 只有一夜，因此 05 仍进表一。

### 分工

| 成员 | 新拟合名单 | 本周职责 |
| --- | --- | --- |
| A | 01、07、09 | 串起训练和开发预测，复算各阶段 F1，重点核查 N3 的贡献及 N1 AP。 |
| B | 02、08 | 验证旧头重放，锁定评分器，组织预留 8 人统一推理、两张配对表和逐人剔除。 |
| C | 03、10 | 生成并审计 9 份匹配块，核对夜号、阶段标签和夜间位置，交 B 第一周 01–05 的旧头及各自标准化参数。 |
| D | 04、06 | 整理 06–10 的旧头和标准化参数，核查缓存与运行指纹，独立复算两张表并汇总选题决定。 |

### 五个工作日

| 工作日 | 验收节点 |
| --- | --- |
| 第 6 天 | 锁定名单、两项拆分差、实际意义界值和评分器版本。C 先交 01、04，再交 02、03，随后交其余五份，并交接 01–05 旧参数。A/B 用 dev8 对照旧预测验证回放，D 独立抽查 04 的匹配块。 |
| 第 7 天 | 完成 9 次新增拟合，按名单交付模型、标准化参数、训练日志，以及 pool36 和 dev8 的逐段五类概率。 |
| 第 8 天 | 在开发数据上交叉核验样本、阶段和指标。锁定 20 个旧头、9 个新头及对应参数、块表和评分规则。 |
| 第 9 天 | B 组织一次预留 8 人统一评分。A 复算分期结果，C 核对受试者、夜晚和 epoch 覆盖，D 独立复算两表。 |
| 第 10 天 | 交付两张配对表和一页选题决定，报告点估计、95% 区间、同向名单数和逐人剔除范围。 |

继续使用 W4 30k 冻结 EMA 的 384 维 EEG/EOG 特征和第一周线性头。新组每份名单仅用选中的 1,800 段拟合标准化，head seed 0、固定 30 轮、batch 128 保留尾批，共 450 次更新。原 pool36 和 dev8 用于开发核查，W4 预留 8 人在方案锁定后统一评分，16 人测试集继续封存。

主指标为原始五类 Macro-F1。名单内配对差等权汇总，同一受试者跨名单共用 bootstrap 权重。本周建议在第 6 天开工前冻结 1 pp 的实际意义参考和 ±1 pp 的等效带。等效判断要求整个 95% 区间落入约定等效带。

## 新一轮第 1 周（2026-11-26 定稿）

[五个工作日安排](docs/budget-temporal-20261126/README.md) · [交接登记](HANDOFF.md#新一轮第-1-周交接) · [C→D 交接包](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/c-budget-week1-v01-20261126/c_budget_week1_handoff_v01.zip)

本轮检验两件事。相同的 600 段新增标签用于加人或加深，哪种办法更有效。极少标签下，真实相邻信息能否改善 N1 排序和短 N1 识别。第 5 个工作日按配对结果决定优先复验方向。

| 成员 | 本轮分工 |
| --- | --- |
| A | 负责真实邻接 A1、名单 1 的 A0 监督基线，以及时序配对汇总。 |
| B | 负责同夜随机邻接 A2、名单 2 的 A0 监督基线，以及预算结果复评。 |
| C | 负责共用固定轮数训练器、01–05 的预算配对及标签覆盖核查。已上传供 D 使用的 [06 号名单](experiment3/week1_budget_20261126/rosters/06/blocks.csv) 和[运行说明](experiment3/week1_budget_20261126/README.md)。 |
| D | 先交连续记录小包和新版全量共享特征，再跑 06–10 的预算配对并独立复评时序结果。 |

两条线共用 W4 30k 冻结 EMA 的 384 维 EEG/EOG 特征。预算线从 1,200 段增至 1,800 段，分别评估 36 人和固定 dev 8 人。时序线固定 dev 8 人主评，另报告 9 人和合并 17 人。16 人测试集继续封存。

**C 正式结果已发布。** 01–05 共 15 次拟合。36 人主评中，加人 − 加深平均 +1.88 pp，5/5 份同向。固定 dev 8 人平均 +2.05 pp，4/5 份同向。[C 结果报告](experiment3/week1_budget_20261126/results/README.md) · [完整预测与复评文件](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-budget-week1-results-v01-20261126)

## 第五周 W5（2026-11-23 至 2026-11-27）

[W5 实验安排](WEEK5.md) · [时间安排 PDF](docs/week5/work_plan_W5.pdf) · [交接登记](HANDOFF.md#第五周交接) · [周五结果登记](WEEK5.md#周五结果登记)

本周四条实验线并行推进，**11 月 27 日（周五）**统一登记各自的结果。下游沿用 5 人和 14 人标签预算，按人五折选轮，在 8 人开发集评分。16 人测试集继续封存。

| 实验 | 本周任务 | 周五交付 |
| --- | --- | --- |
| A | 在现有 30k 特征上比较单 epoch、直接拼接和 BiGRU | 两档预算的时序对照与结论 |
| B | 在 Mac CPU 上为跨模态三组各补 head seed 3、4 | EOG-only 配对结果与逐 seed 判读 |
| C | 比较普通与 self-cond 模型的全层和后三层读出 | 四格性能及前向、存储、读出分项开销 |
| D | 测试连续 30 秒预训练输入，并与 5 秒配置测速 | 显存、步时、可行性判断及已完成的下游 probe |

## 第四周 W4（2026-11-14）

本周验证小 EEGDM 的预训练停止点、LFT 的读出效率，以及同步 EEG 条件能否改善 EOG-only 睡眠分期。

| 实验 | 本周任务 | 优先交接 |
| --- | --- | --- |
| A | 比较 1 秒与 2 秒 std pooling 的 LFT 耗时和分数，实现同窗 EEG 条件并训练 30k 步 | 第 2 天给 C 池化对照和推荐配置，第 3 天给 B 跨模态实现 |
| B | 固定预训练验证名单和分段 MSE，训练错时 EEG 对照，评估生成质量并统一复评 | 第 1 天给 A、D held-out 名单与 MSE 脚本，第 6 天上午给 C、D 跨模态三组汇总 |
| C | 定稿跨模态三组比较规则，筛查 linear / LFT × pretrained / random-init | 第 1 天给 A、B、D [跨模态三组对照规则](experiment3/week4/cross_modal_protocol.md)，第 6 天下午给 D 判读结论 |
| D | 维护公共训练脚本，确定预训练停止点，筛查 self-conditioning | 第 2 天给 A、B 公共脚本，第 3 天给 A、B、C 共享的无 EEG 30k checkpoint |

新一轮预训练从 54 人训练池中留出 8 名不在 14 人标签名单中的受试者做验证。下游沿用 5 人和 14 人预算，按人五折在训练侧选轮，全预算重训后在 8 人开发集最终评分。16 人测试集继续封存。跨模态三组使用相同的 30k 训练预算，下游统一只输入 EOG，主指标为 N1 AP 和 Macro-F1。

## 第三周 W3

[第三周工作安排](WEEK3.md) · [四天方案 PDF](docs/week3/work_plan_W3.pdf) · [交接登记](HANDOFF.md#第三周交接)

**C 第三周交付。** [固定 20 轮主对照与扩展](experiment3/week3/README.md)于 11 月 9 日入库。[训练侧五折选轮主对照](experiment3/week3/inner_cv_v01/README.md)于 11 月 12 日完成并补充三组 C1 与 B 线性头的配对结果。

A 比较睡眠分期基线、标签效率与 EEG/EOG 输入。B 检验 pooling、diffusion step 与 N1 判别表现。C 比较相同特征下的 linear head 与 LFT。D 在 Sleep-EDF 训练池上开展从零扩散预训练 pilot。四项实验沿用按人划分，核心比较使用 14 位有标签受试者，16 位测试受试者继续封存。

**2026-11-08 A第三周交付(PR #15):** 实验A基线比较完成——选定 ContraWR(上游重实现,许可核查在案)为主比较基线,SleepTransformer 完成协议与实现核查(TF1不可运行,L=21上下文冲突);18 格主矩阵(5/14/54人 × CNN/ContraWR × seed0-2)完成,CNN 侧 9 格复用 W2 且 12/12 逐项核查通过;14 人 EOG 对照 6 格复用;结果与配对 CI 见 [reports/week3/a_w3_results.md](reports/week3/a_w3_results.md)。9 个 ContraWR run 六件套已发布 Release [a-contrawr-w3-v01-20261108](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/a-contrawr-w3-v01-20261108)。

**2026-11-02 C追加两项验证已完成：** [EEG-only多名单与按受试者留出：报告、完整预测和模型下载](experiment3/week2/ADDITIONAL_VALIDATION.md)。前者固定8人上平均+0.136pp，后者按折重训9个主干后平均−0.042pp，尚未确认稳定跨人适配收益；新增96份最终预测、216次内层拟合，全部本地审计通过。新结果独立于原W2全组登记，等待B独立复评；测试集封存。

**C→B 特征下载补充（2026-10-31）：** [EEG/EOG 两份完整池化 NPZ](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-pooled-features-v01-20261031)已发布到 GitHub Release；各154124×2560、float32、未标准化，键为sample_id和x，附样本索引、SHA和B接口核验。EEG为原始冻结S0，EOG为C按D同协议本地重建。

**2026-10-31 C第二周交付：** [实验③结果与复评入口](experiment3/week2/README.md)已备妥3轮适配、80份验证预测、配置/历史/指标及N1分析；10%双模态平均退步、25%基本持平，未证明稳定适配收益。B已发布v2名单核验和80份预测复评；D共同EOG来源认可及实际接收验收仍待补；测试集封存。第一周C材料保留在[原目录](experiment3/README.md)。


利用预训练EEG表征与未标注EEG/EOG，研究少标签睡眠分期和N1识别。

## 第二周 W2 v1.1

第二周依据2026-10-26修订的[第二周工作文档](docs/week2/work_plan_W2_v1.1.pdf)，按阶段与交接条件推进，不对应固定日期。第一周已经得到冻结特征及融合的初步结果；尚未证明EEGDM稳定优于公平CNN，也未解决N1的稳定识别。

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
