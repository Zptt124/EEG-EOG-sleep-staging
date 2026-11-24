# 睡眠分期课题 · W5 实验安排

2026 年 11 月 23 日至 27 日，**11 月 27 日（周五）统一交付**。A、B、C、D 四条线并行，各自完成预测、评分和结论。 [时间安排 PDF](docs/week5/work_plan_W5.pdf) · [交接登记](HANDOFF.md#第五周交接)

## 全组共同规则

- 沿用 54 人训练池、8 人开发集、封存的 16 人测试集，以及 5 人和 14 人标签名单。D 沿用 W4 预训练划分，在训练池中用 46 人预训练，另留 8 人作 held-out 去噪验证。
- 下游在标签预算内按人五折选轮，标准化只在对应训练折拟合，之后用全名单重训。配置锁定后在同一开发集上评分。报告 Macro-F1、N1 AP、N1 F1 和按受试者配对差，注明 head seed 与 backbone seed。
- 匹配的现有预测和特征可复用。各人保留配置、逐段预测、结果表和适用的资源记录。

## A ｜Temporal context 对照

- **周一。** 用已有的 30k EEG+EOG epoch 特征，核对 `sample_id`、记录和时间索引。
- **周二至周三。** 在 5 人和 14 人预算下比较单 epoch linear、直接拼接、小 BiGRU。拼接和 BiGRU 使用当前 epoch 和前后各两个 epoch，只预测中心 epoch，统一处理缺窗及记录边界。上下文限于同一受试者的同一记录。
- **周四。** 核对 OOF 与开发集结果，判断 BiGRU 能否缓解 5 人档的下降并保留 14 人档的收益。30k 对照完成后，再评估是否补 240k 或训练折内降维拼接。
- **周五交付。** 三种读出的同口径结果表、预测链接和 temporal context 取舍。

## B ｜跨模态补 head seed

- **周一。** 用已发布的统一 EOG 特征在 Mac CPU 上试跑，核对样本顺序、评分入口和耗时。
- **周二至周三。** 固定同窗 EEG、错时 EEG、无 EEG 三套 backbone，统一做 EOG-only linear probe。在 5 人和 14 人两档补配对 head seed 3、4，共 12 个新增最终结果。
- **周四。** 与已有 head seed 0–2 一起汇总逐 seed 方向及按人配对差，判断结论在固定主干下是否稳定。
- **周五交付。** 三组更新结果、逐 seed 方向与一句结论。若只完成部分预定 seed，记录数量并保留三组配对。生成质量沿用 W4 报告作为参考。

## C ｜后层读出的性能与成本

- **周一。** 核对普通 30k 与 self-cond 30k 的权重、层顺序和特征口径。有普通模型特征即可先开始。
- **周二至周三。** 做普通／self-cond × 全 12 层／第 10–12 层的四格对照。统一 pooling、线性头、标签预算和 head seed，复用协议匹配的全层结果。
- **周四。** 分开测 backbone 前向、特征保存和分类头读出。只取第 10–12 层时，backbone 仍须完成 12 层前向，开销差异看特征存储和下游读出。可选做全层 LFT 与后三层 LFT 的计时对照。
- **周五交付。** 四格性能表、分项耗时和默认读出建议。若缺 self-cond 资产，交普通模型对照并注明待补项。

## D ｜连续 30 秒预训练与可行性

- **周一。** 用现有模型尺寸、标准化和 noise schedule，核对模型入口收到的是连续 30 秒信号。
- **周二。** 在同一设备上短跑比较 5 秒和 30 秒，记录峰值显存、稳定步时、有效 batch 和每步信号量。据此确定筛查规模，说明按相同信号量还是相近 GPU 时间对齐。
- **周三至周四。** 时间和显存允许时，做固定预算的 30 秒训练和同口径下游 probe。25k、35k 的加密 probe 使用现有 5 秒 checkpoint，有余力再补。
- **周五交付。** 5 秒／30 秒成本表和可行性判断，附已完成的 probe。如果本轮资源不足，记录实测结果和下一轮所需资源。

## 资产和交接

A 和 B 自行取用已发布特征。若 C 本地缺 self-cond 30k 权重或可按层切分的特征，由 D 开工时将现成资产放入共用目录一次。

- [W4 普通 30k 双模态特征、无 EEG 与错时组特征](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w4-xmodal-shifted30k-v01-20261119)
- [W4 同窗组统一特征与探针](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w4-probe-extras-v01-20261119)
- [W4 预训练及步数报告](https://github.com/Zptt124/EEG-EOG-sleep-staging/blob/57c78f88b6cd69636fdd8af3def158298dc252f2/experiment4/week4/report_pretrain_w4.md)
- [B 的 W4 生成质量评估，PR #22 分支](https://github.com/Zptt124/EEG-EOG-sleep-staging/blob/fb4370d4b610723653332157be5dab7047fcb710/reports/week4/b_gen_quality.md)

## 周五结果登记

11 月 27 日各人填写一行。A 从同一张表汇总汇报草稿。

| 负责人 | 实验对照 | 关键数字 | 一句话结论 | 结果链接 |
| --- | --- | --- | --- | --- |
| A | 单 epoch／拼接／BiGRU（5 人 3 seed + 14 人 1 seed） | 5 人档 BiGRU−single 调整后 ΔMacro **+2.97 [2.25, 4.20]**、原始 ΔN1 F1 +4.69 [1.37, 7.30]；14 人档 BiGRU−cat 调整后 ΔN1 F1 **+13.25 [5.53, 19.31]**；cat−single 14 人调整后 ΔN1 F1 **−15.23** | 保留 temporal context，用 BiGRU 不用直接拼接；5 人档收益显著、14 人档不亏 | [A 报告](reports/week5/a_temporal_ctx.md) |
| B | 同窗／错时／无 EEG，固定主干补 head seed（12/12 完成，合计 seed 0–4） | ΔN1 AP 六格全跨 0（5 人档同窗−无 EEG +0.56 [−0.27, +1.72]）；调整后 ΔN1 F1 六格全跨 0；5 人档同窗−无 EEG ΔMacro-F1 −0.78 [−1.23, −0.30]，5/5 seed 为负；head seed SD：Macro-F1 约 0.5、N1 AP 约 0.7 | 补到 5 个 head seed 后判读不变，未确认稳定跨模态收益；原始 N1 F1 的差来自门槛 | [B 报告](reports/week5/b_xmodal_seeds.md) |
| C | 普通／self-cond × 全层／后 3 层 | 待登记 | 待登记 | 待登记 |
| D | 5 秒／30 秒预训练输入 | 待登记 | 待登记 | 待登记 |
