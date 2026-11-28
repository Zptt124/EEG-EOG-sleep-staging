# B 第 3 周｜从原始记录独立复跑 Ying 外部复验（2026-11-28）

**负责人：B · 分支 `feat/b-w3-ying` · 对照对象：C 的归档 `experiment3/week4_budget_external_ying/`（main `8293adf`）**

## 一、结论

从 OSF 原始包重新下载并校验，B 用自己的代码完成标签解析、EDF 读取、抽块设计、拟合汇总和 bootstrap，独立复跑了 480 次拟合。外部主要比较「新人新夜 − 原人原夜」复现了 C 的结果：

| 比较 | B 均值 | B 95% 区间 | C 均值 | C 95% 区间 | 均值偏差 |
|---|---:|---|---:|---|---:|
| **新人 − 原夜（主要）** | **+1.785** | **[+0.59, +3.08]** | +1.786 | [+0.58, +3.07] | −0.001 |
| 另一夜 − 原夜 | +0.466 | [−0.43, +1.51] | +0.466 | [−0.43, +1.47] | 0.000 |
| 新人 − 另一夜 | +1.319 | [+0.15, +2.57] | +1.320 | [+0.15, +2.57] | −0.001 |

4 折差值（主要比较，pp）：

| 折 | B | C |
|---|---:|---:|
| 1 | +1.58 | +1.58 |
| 2 | +1.54 | +1.54 |
| 3 | +1.50 | +1.50 |
| 4 | +2.52 | +2.52 |

4/4 折为正，24/32 份名单同向，逐折剔除后均值在 +1.54 到 +1.88 之间。按预设规则判读，结果同为 **external_support**。

**偏差来源。** 480 次拟合里，331 次的混淆计数有个别格子不同（共 60,000 格，1,131 格不同）。单次 Macro-F1 最大差 0.06 pp，平均差 0.009 pp。原因是 B 在 Apple MPS 上提特征，C 在 CUDA（TF32）上提。B 在同一夜（SLP014 Day1）分别用 CPU 和 MPS 提取，特征最大差 7.5e-6，特征量级约 0.34。区间上的差别来自 bootstrap 抽样误差：B 的 bootstrap 跑在 C 的混淆计数上，主要比较区间为 [+0.59, +3.09]，和跑在 B 自己的计数上几乎相同。

**没有发现需要定位的复现差异。** 不过敏感性分析（第五节）显示，是否达到推进门槛对抽块设计的随机种子比较敏感。

## 二、原始包与输入核查

| 项目 | 结果 |
|---|---|
| OSF 节点 gx36m | 27 个文件，共 3.67 GiB，全部按 OSF 元数据的 SHA-256 和字节数校验通过，见 [osf_manifest.json](osf_manifest.json)，取回时间为 2026-11-28 08:33 UTC |
| 夜数 | 20 人 × 前两夜，SLP004 和 SLP020 的 Day2 在压缩包里标了 “no data”，Readme 说明原因是夜间起床中断，因此共 38 夜 |
| 导联 | 38 夜都有 F4-A1、EOG-R、EOG-L，采样率 500 Hz，单位 µV，见 [label_audit.csv](label_audit.csv) |
| 专家标注 | 每夜 3 位专家的 TXT 区间和 CSV 逐段标签逐段一致。多数投票，三人各不相同时取 Expert 1，任一专家标 NotScored 时该段记为不可用 |
| 段数 | 共 40,321 段，其中 816 段被剔除，**39,505 段可评分，和 C 一致** |
| 对齐 | 37 夜的 EDF 整段数等于标注段数。SLP010 Day1 的 EDF 比标注多 5 段，Expert 1 的标注在末尾短了 5 段，取三位专家的公共前缀（1,082 段）。和 C 的处理一致，但规则是 B 从数据里独立发现的 |
| 冻结表示 | `ckpt_step30000.pt` 的 SHA 为 `21c8514e…`，W4 源码从 6590ec9 导出，EEGDM 为 003b5a2 |

B 的实现没有使用 C 的 `download.py`、`extract.py`、`confirm.py`。EDF 按标准格式直接读取，因为本机 mne 和 scipy 版本不兼容，也没有用 mne。这样读取这一步也是独立实现。信号路径按冻结表示的要求和 C 保持一致：F4-A1 和 EOG-R − EOG-L，整夜从 500 Hz 降到 100 Hz，换成 µV，再经 W4 30k EMA 后三层 gate 和 1 秒 std 池化。线性头训练器沿用组内共用的 `week1_budget_20261126/train.py::fit`（seed 0，30 轮）。

**数据观察**（不影响本次复跑）：

- Readme 说是"连续三夜"，但 EDF 开始日期显示部分受试者的 Day1 和 Day2 并不相邻，比如 SLP005 是 04-08 和 04-12，SLP007 是 04-19 和 04-26，SLP013 是 04-14 和 04-19。
- C 协议里写的 "same-amplifier Days 1 and 2"，在 Readme 和 EDF 头里都找不到依据（设备字段是 `X X X`），需要 C 补上出处。

## 三、人员隔离与抽块设计

- 4 个评估折由种子 20261128 重建，和 `protocol.json` 一致。20 人不重叠，训练块里出现本折评估人的次数为 0。
- **B 独立生成的块表和 C 的 `blocks.csv` 内容逐行相同**（5,440 行）。两边 SHA 不同只是因为 C 的文件用 CRLF 换行。
- 用 B 自己解析的标签逐条审计 C 的块表：块内 60 段全部可评分、块中点所在三分位正确、原夜 5 人各 4 块、三臂时段多重集相同、原夜加深只在原夜里取块、另一夜取同一批人的另一夜、新人和原人不重叠且夜号和时段与另一夜对应、预算之间不重叠。**0 个问题**。见 [design_report.json](design_report.json)。
- 各折训练池 12–13 人。除只有一夜的 004 和 020 外，SLP010 也被资格门槛排除：它的 Day1 上 Expert 2 零散标了 269 段 NotScored，只剩 5 个完整块。

## 四、协议修订时间线（可核验证据）

| 时间 | 事件 | 证据 | 可核验程度 |
|---|---|---|---|
| 2022-03 至 05 | 原始记录采集 | EDF 头的开始日期，压缩包内 EDF 时间戳为 2022 年 | 可核验 |
| 2024-10-26 至 11-06 | 专家标注文件写成 | 压缩包内 TXT 和 CSV 的时间戳 | 可核验，但时间戳由作者机器写入 |
| 2024-11-25 | OSF 上传 20 个 SLP 压缩包 | OSF `date_created` | 可核验 |
| 2024-11-27 | OSF 上传 Readme | OSF `date_created` | 可核验 |
| 未知 | 协议 v0（对齐核查前）：target 为 "complete 30 s annotation-aligned epochs"，没有资格门槛 | `protocol_before_alignment_review.json` | **内容可核验，时间无证据** |
| 未知 | 协议 v1（资格核查前）：加入 "omit author NotScored epochs" 和 SLP010 Day1 公共前缀 | `protocol_before_eligibility_review.json` | **内容可核验，时间无证据** |
| 未知 | 协议 v2（最终版）：加入 "each night ≥ 6 usable blocks across ≥ 2 thirds" 的训练资格门槛 | `protocol.json` | **内容可核验，时间无证据** |
| 2026-11-28 02:16 +0800 | 三版快照、代码、块表、混淆计数和结果在同一提交 `c48160e` 中一次性入库 | git | 可核验 |
| 2026-11-28 14:14 +0800 | README 补充窗口说明（`8293adf`），协议和结果文件没有改动 | git | 可核验 |

**已能确认的：**

1. 跑出结果的 `confirm.py` 就是发布版：`data_lock.json` 里的 `code_sha256` 和仓库中的文件相同（`7974d817…`）。
2. 实际运行版 `extract.py` 的 SHA（`018da6f7…`）和发布版（`4aa7ef96…`）不同。C 说明只是把源码路径改成了命令行参数，但原文件不在仓库里，B 无法核对。
3. 去掉 v2 的资格门槛后，B 的设计代码在抽 SLP010 时报错：Day1 的 5 个完整块凑不出三臂共用的匹配时段（见 [sensitivity/sens_no_eligibility_error.log](sensitivity/sens_no_eligibility_error.log)）。也就是说，如果不加门槛或其他处理，设计跑不起来。这支持"门槛是为了让设计能运行才加的"这种解释，但不能证明加门槛发生在看到结果之前。

**还需要 C 提供：**

- 三版快照的原始文件，带文件系统修改时间，或当时的提交、聊天或备份记录；
- `confirm.py --freeze` 和正式运行的终端日志（带时间戳）；
- 特征 npz 的生成时间；
- 实际运行版 `extract.py`。

有了这些才能判断 v1 和 v2 的修订是在看到任何拟合结果之前做的，还是之后。

## 五、敏感性分析（探索性，不替代预设分析）

| 变体 | 评估段数 | 新人 − 原夜 | 95% 区间 | 正向折 | 同向名单 | 判读 |
|---|---:|---:|---|---:|---:|---|
| 主分析（预设，设计种子 +1） | 39,505 | +1.79 | [+0.59, +3.08] | 4/4 | 24 | external_support |
| 只换评估标签：打分专家内多数投票，设计和拟合不变 | 40,321 | +1.71 | [+0.53, +3.00] | 4/4 | 25 | external_support |
| 标签规则同上，训练池随之变化，设计全部重抽 | 40,321 | +0.71 | [−0.49, +2.13] | 4/4 | 15 | inconclusive |
| 设计种子 +2 | 39,505 | +1.42 | [+0.30, +2.66] | 3/4 | 20 | external_support |
| 设计种子 +3 | 39,505 | +1.14 | [−0.10, +2.50] | 3/4 | 23 | inconclusive |
| 设计种子 +4 | 39,505 | +0.82 | [−0.49, +2.34] | 3/4 | 17 | inconclusive |
| 设计种子 +5 | 39,505 | +1.67 | [+0.33, +3.31] | 3/4 | 23 | external_support |
| 设计种子 +6 | 39,505 | +1.22 | [−0.03, +2.59] | 3/4 | 19 | inconclusive |
| 去掉资格门槛 | — | — | — | — | — | 设计无法运行 |

解读：

- NotScored 规则本身影响很小：只换评估标签时，主要比较从 +1.79 变为 +1.71。
- 结果对"抽到哪些名单和块"更敏感。协议和数据都不变、只换设计种子时，5 个种子都为正，均值在 +0.82 到 +1.67 之间，但只有 2/5 达到推进门槛。预设种子的 +1.79 比这 5 个都高。包括预设种子在内的 6 个种子平均为 +1.34。
- 这和预设区间（已经包含小于 +1 的效应）并不矛盾。但它说明"达到推进门槛"这一判读不够稳：方向稳定为正，效应大小更可能在 +1 pp 左右，而不是 +1.8 pp。建议组会上把这一点和 C 的 Sleep-EDF 结果（+1.53）一起看。

## 六、复跑环境与产物

| 项目 | 值 |
|---|---|
| 机器 | Apple M5，macOS 26.6，提特征用 MPS，拟合用 CPU（4 线程） |
| Python 和包 | 3.14.6，PyTorch 2.14.1，NumPy 2.4.6，SciPy 1.18.0 |
| 耗时 | 下载 136 s，38 夜提特征 4,346 s（MPS），480 次拟合加汇总约 40 s |
| 脚本 | [54 下载与校验](../../scripts/54_b_w3_ying_download.py)，[55 标注审计与提特征](../../scripts/55_b_w3_ying_extract.py)，[56 设计复核、拟合与汇总](../../scripts/56_b_w3_ying_confirm.py) |
| 结果 | [summary_b.json](summary_b.json)，[per_fit_scores.csv](per_fit_scores.csv)（480 行，B 和 C 逐拟合对照），[b_confusions_flat.csv](b_confusions_flat.csv)（按评估者的混淆计数），[sensitivity/](sensitivity/) |
| 本地大文件（未入库） | 原始包在 `cache/ying_osf/raw/`，38 夜特征在 `runs/b_w3_ying/features/` |

复跑命令：

```bash
python -I scripts/54_b_w3_ying_download.py cache/ying_osf
python -I scripts/55_b_w3_ying_extract.py cache/ying_osf --out runs/b_w3_ying \
  --w4-source <W4 6590ec9 src> --eegdm-vendor <EEGDM 003b5a2> --ckpt <ckpt_step30000.pt> --device mps
python -I scripts/56_b_w3_ying_confirm.py runs/b_w3_ying
python -I scripts/56_b_w3_ying_confirm.py runs/b_w3_ying --variant design_seed --design-seed-offset 2
```
