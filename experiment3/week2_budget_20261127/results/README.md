# 实验 C｜第二周另一夜配对

冻结了 01–04、06–10 共 9 份名单。生成顺序为 01、04，随后 02、03，再到 06–10。使用第一周的连续候选块和夜间三分位，按 20261127 + 名单号抽样。选块只用受试者、记录和原始 epoch 键。05 的起点成员 SC452 只有 SC4522 一条记录，因此 05 仅纳入表一。

独立复核了全部 90 个新增块。每份名单保留原 1,200 个共同起点键。同一位受试者的两块取自同一条另一夜记录，每块 60 个连续有效 epoch，与原人原夜的追加块逐人逐块匹配时段。九份名单保留原 pool36 和 dev8 评分对象。01（A）和 04（D）的共同起点键已独立对照原名单。

[九份匹配名单、新头与开发预测](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/c-budget-week2-results-v01-20261127/C_week2_matching_and_results_20261127.zip) · [给 B 的十套旧头及对应标准化](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/c-budget-week2-results-v01-20261127/C_week1_heads_for_B_20261127.zip) · [SHA-256](SHA256SUMS.txt)

## C 的正式拟合与开发核查

03 和 10 各用 1,800 段拟合训练侧标准化及五类线性头，seed 0、30 轮、450 次更新。复用了第一周的训练函数和特征行排序。下表仅为开发评分，Macro-F1 以百分比表示。

| 名单 | 评分集 | 原人原夜 | 原人另一夜 | 新人新夜 | 换夜收益（pp） | 新人较另一夜（pp） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 03 | pool36 | 55.57 | 57.56 | 57.56 | +2.00 | ≈0.00 |
| 03 | dev8 | 58.92 | 61.53 | 61.68 | +2.61 | +0.15 |
| 10 | pool36 | 56.59 | 55.64 | 56.32 | -0.95 | +0.68 |
| 10 | dev8 | 66.30 | 64.52 | 66.71 | -1.78 | +2.19 |

03 在两套开发评分上均支持换夜，10 在两套均反向。两份名单还不足以确认普遍收益，正式判读使用预留 8 人的九名单配对表及区间。

10 的两个旧条件读取 D 的完整精度结果。03 的两个旧条件读取 C 的归档结果。[D 10 号原始分数](https://github.com/Zptt124/EEG-EOG-sleep-staging/blob/5daead1b720b63404c3ef4b7db0ca0b4f37a22c0/experiment4/week4/budget_w1/runs/10/paired/summary.json)

匹配块冻结后，统计了三组新增标签的五类段数、N1/N3 涉及人数和块数，并按夜间前、中、后分列。

C 的第一周 01–05 两个追加条件共 10 个旧头已与其各自标准化参数配对打包。dev8 重放的样本键和 argmax 全部一致，五类概率最大绝对差为 1.1e-16。

## 后续验收

A/D 实际拟合产物交来后，可对照已冻结的 01、04 共同起点键检查其运行。预留 8 人的覆盖核查、反向名单分析及适用人群结论，等待 B 统一评分并交付逐段预测。C 此轮只跑了开发评分，预留 8 人和封存测试 16 人均未评分。

- [匹配和数据边界核对](audit/nine_roster_block_audit.csv)
- [五类标签和夜间覆盖](audit/stage_and_night_coverage.csv)
- [旧头 dev8 回放](audit/old_head_dev8_replay.csv)
- [C 两个新头的开发分数](audit/c_new_head_development.csv)
- [D 共享特征版本与指纹](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w6-shared-cache-v01-20261126)

- [开发配对差 CSV](audit/c_development_pairwise.csv)
- [预留集预期的样本覆盖](audit/heldout_expected_coverage_by_night.csv)供统一评分后核对。
