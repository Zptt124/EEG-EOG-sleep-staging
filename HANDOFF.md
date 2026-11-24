# 阶段交接登记

## 第五周交接

以 [W5 实验安排](WEEK5.md)和[时间安排 PDF](docs/week5/work_plan_W5.pdf)为准。A 和 B 自行取用已有发布资产。若 C 缺 self-cond 30k 权重或分层特征，D 在开工时将现成资产放入共用目录一次。

11 月 27 日（周五），四人各在[共用结果表](WEEK5.md#周五结果登记)登记一行，附结果链接和一句结论。A 从同一张表汇总汇报草稿。

**2026-11-24 B 跨模态补 head seed（12/12 完成）：** 用 W4 同一份三组 EOG 特征（SHA 与 W4 登记一致）和 34 号探针，在 5 人和 14 人档补 seed 3、4。试跑重算 W4 的 seed 0，预测逐字节一致。30 个 run 经 39 号复评自检 30/30 通过。合并 seed 0–4 后，N1 AP 和 α 调整后 N1 F1 的六个配对区间仍全部跨 0，W4 判读不变。详见 [reports/week5/b_xmodal_seeds.md](reports/week5/b_xmodal_seeds.md)。完整 run 包待发布到 Release。

## 第四周交接

**2026-11-16 B → A、D（第 1 天）：** 预训练 held-out 名单 `w4_v01` 已锁定：SC414、SC416、SC431、SC450、SC457、SC463、SC467、SC477（不在任何标签预算名单中，5/14/27 人预算者全部留在预训练池；按编号分 8 层各抽 1 人）。预训练池剩 46 人、114,041 段。分段 MSE 脚本 `scripts/32_b_w4_pretrain_mse.py` 使用固定的 4,000 个窗×模态和固定噪声，每 5 个扩散步一段，EEG/EOG 分开报告，默认按空条件、用 EMA 权重评估。用法与核对见 [reports/week4/b_pretrain_val_for_AD.md](reports/week4/b_pretrain_val_for_AD.md)。

**2026-11-16 B 跨模态工具：** 错时 EEG 配对（`scripts/33_b_w4_shift_pairs.py`，60–300 秒，逐窗确定，与窗清单范围无关）在 46 人训练池 684,246 个目标窗上覆盖率 100%，条件窗在后的占 50.0%。第四周探针 `34_b_w4_probe.py` 可只用 EOG，并在同一次运行里做训练侧 α 调整。三组复评 `35_b_w4_crossmodal_eval.py` 先逐行核对对齐，再报三个配对差。说明见 [reports/week4/b_crossmodal_tools.md](reports/week4/b_crossmodal_tools.md)。

**2026-11-17 B 核对 D 的公共预训练脚本（PR #23）：** B 用自己的代码按 D 的配方重建 val 集，窗、t、窗清单与 D 一致（噪声 SHA 因 x86/ARM 的 CPU 随机数实现不同而末位不同）。复算 D 主实验 0–30k 共 7 个 checkpoint 的 val MSE（EMA 与 online），stop_metric 最大差 5.8e-6，最佳步同为 30k。错时配对表重算后与 D 的 npy 逐字节一致，规则逐条核对通过（覆盖 100%、两向各半、候选内均匀）。三组对照以 D 的配对表为准。详见 [reports/week4/b_d_pretrain_recheck.md](reports/week4/b_d_pretrain_recheck.md)。

**2026-11-19 B 复评 C 第四周四格（PR #25）：** 12 个 run 的预测、OOF、原始指标、α 与调整后指标、三个配对差点估计全部复现。预训练的 Macro-F1 增益在 linear/LFT 都稳定（+3.2/+3.3 个百分点）；N1 上只有 linear 头有收益（N1 AP +0.050，α 调整后 N1 F1 +0.038，区间不含 0），LFT 的 N1 AP 与 α 调整后 N1 F1 都没有变化。预训练下 LFT 比 linear 的 N1 AP 低 −0.049 [−0.092, −0.014]，原始 N1 F1 的优势来自判定门槛。详见 [reports/week4/b_c_reeval_w4.md](reports/week4/b_c_reeval_w4.md)。

**2026-11-21 B 跨模态三组正式汇总与复评：** C 的 6 个跨模态 run 复评 6/6 一致，D 的 18 个三 head seed run 从 Release 下载逐段预测后复评 18/18 一致（此前 C 只引用了 D 的配对表）。B 用 D 发布的三组特征自己重跑 18 个探针，结果与 D 逐项相同。B 新增 α 调整后的配对差：原始 N1 F1 上三个区间不含 0 的差（5 人档同窗/错时−无 EEG、14 人档错时−无 EEG），经训练侧门槛校正后全部跨 0，N1 AP 六个区间也都跨 0，结论仍是未确认稳定跨模态收益。Mac 初步结果与正式结果只差 1 段预测。错时、同窗两组 30k 的 val MSE（空条件与给 EEG 条件）复算与交付值相差 ≤ 7.4e-6；给 EEG 条件后 EOG 去噪误差同窗降 0.7%、错时降 0.1%。同窗特征包的 index.csv 是 LF 换行，与 features_info 登记的 CRLF 版本 SHA 不同，内容一致。详见 [reports/week4/b_xmodal_final.md](reports/week4/b_xmodal_final.md)。

## 第三周交接

以[第三周工作安排](WEEK3.md)和[四天方案 PDF](docs/week3/work_plan_W3.pdf)为准。交付时记录配置、模型来源、run ID、样本索引与接收核验。

| 交付方 → 接收方 | 第三周产物 |
| --- | --- |
| B → C | t=1 的 std/mean 特征索引、`sample_id`、模态与层轴定义及兼容的 linear head 基线 |
| C → B | LFT 入口、官方 extractor 的适配说明和输入加噪设置 |
| B → D | 线性探针、训练侧标准化、特征与数据权限核查、统一评分入口 |
| A、C、D → B | 固定配置、模型来源与逐段预测，由 B 核对 `sample_id` 并统一复评 |

**2026-11-09 B第三周进展与选模口径提醒：** 详见 [reports/week3/b_status_w3.md](reports/week3/b_status_w3.md)。
B 的线性读出改为训练侧选模（预算内按受试者内层交叉验证选轮次，开发集只评估一次，`scripts/27_b_w3_linear_probe.py`）。
按此重跑的 std 6 格与第二周开发集早停相比：5 人预算 Macro-F1 −0.032、N1 F1 −0.120，14 人 −0.004 / −0.027，开发集早停结果偏乐观。
A 已按同一协议重训 21 格（PR #17，Release `a-w3-baselines-v02-20261109`），v01 开发集早停结果废止。
B 已复评 A v02 的 21 份与 C 的 32 份去重预测，sample_id 覆盖与各自报告的均值一致，结论见 b_status_w3.md。
D 的配对探针请用 B 的探针脚本，用法见 [reports/week3/b_probe_for_D.md](reports/week3/b_probe_for_D.md)。
**2026-11-10 B 主实验完成：** C 的 6 个 NPZ 按交付清单 SHA256 校验通过，B 已补齐 mean 6 格、t=10/25 与 mean 的 N1 调整。
mean − std 的 Macro-F1 在 5/14 人分别为 −0.020 / −0.025（区间不含 0，8 名验证者全部下降），14 人 N1 F1 −0.070；
mean 的 N1 损失主要在判定门槛（N1 AP 仅 −0.018）。t=10/25 的 Macro-F1 低于 t=1。结论见 [reports/week3/b_status_w3.md](reports/week3/b_status_w3.md)。
**2026-11-09 C 第三周交付。** [已完成的固定 20 轮实验](experiment3/week3/README.md)包含 18 组配对、36 份逐段预测和复评入口。C 已完成 B 要求的 std 核验、mean pooling 与 t=10/25 六组特征。[交付索引与 SHA256](experiment3/week3/deliveries/B_week3_features/README.md)已入库，[六个池化 NPZ](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-b-w3-pooled-features-20261109)可从 GitHub Release 下载。

**2026-11-12 C 主对照补跑完成。** [训练侧五折结果](experiment3/week3/inner_cv_v01/README.md)提供三组 C1 的选轮曲线、折外与开发预测，以及与 B Release 中三个同 seed C0 的逐样本配对。

**2026-11-08 A第三周交付(PR #15):** 实验A 基线比较完成。交付物:基线选择与核查
[reports/week3/a_baseline_selection.md](reports/week3/a_baseline_selection.md);
W2 复用核查 [reports/week3/a_w3_reuse_check.md](reports/week3/a_w3_reuse_check.md)(12/12 通过);
9 个 ContraWR run 六件套已发布 Release
[a-contrawr-w3-v01-20261108](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/a-contrawr-w3-v01-20261108);
18 格主矩阵/配对差/EOG 对照/错误去向/曲线/资源表见
[reports/week3/a_w3_results.md](reports/week3/a_w3_results.md)。
A → B:登记表 reports/week3/{run_registry,metrics_summary,resource_profile,paired_comparison}.csv
与逐段预测(新 run 六件套内),请 B 按 sample_id 核对覆盖并汇总;第三周四档曲线 CNN
部分含 5/14/54 三档(27 人 label50 沿用 W2)。

**2026-11-02 C追加两项验证已完成：** [EEG-only多名单与按受试者留出：报告、完整预测和模型下载](experiment3/week2/ADDITIONAL_VALIDATION.md)。前者固定8人上平均+0.136pp，后者按折重训9个主干后平均−0.042pp，尚未确认稳定跨人适配收益；新增96份最终预测、216次内层拟合，全部本地审计通过。新结果独立于原W2全组登记，等待B独立复评；测试集封存。

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
