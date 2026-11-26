# 第二周（11-27 起，五个工作日）：D 的任务与进度

## 本周问题

第一周的起点是 5 人、1,200 段，再加 600 段：

- 新人新夜比原人原夜高约 1.6 pp（pool36，8/10 名单同向）。

但这两组同时改变了"人"和"夜"。本周新增第三组"**原人另一夜**"，把收益拆成两部分：

- 换夜收益 = 原人另一夜 − 原人原夜；
- 新增个体的额外收益 = 新人新夜 − 原人另一夜。

两张表：

- **表一**：01–10 共 10 份名单，在**预留 8 人**（SC414、SC416、SC431、SC450、SC457、SC463、SC467、SC477）上复验"新人新夜 − 原人原夜"。
- **表二**：01–04、06–10 共 9 份名单的三组拆分。05 号的 SC452 只有一夜，所以不进表二。

预留 8 人只在第 8 天全部锁定后由 B 统一评一次。现在所有数字都只是开发评分（pool36 / dev8）。

## D 的分工与进度

| 天 | 任务 | 状态 |
| --- | --- | --- |
| 6 | 整理 06–10 的 10 个旧头 | ✅ dev8 重放与存档预测逐行一致，argmax 全部相同，概率最大差 3e-7。清单见 `out/old_heads_D_06_10.csv` |
| 6 | 与 C 对齐 20 套旧头清单 | ✅ `out/old_heads_all20.csv`（C 的 01–05 + D 的 06–10） |
| 6 | 用 04 名单独立检查另一夜抽块 | ✅ 9 份名单全部通过（`01_check_matched_blocks.py`，不调用 C 的代码；日志 44） |
| 6–7 | 跑 04、06 的原人另一夜拟合 | ✅ 用 C 的入口 `train_other_night.py`，各 1,800 段、450 次更新（日志 45） |
| 6 | 为三组结果建统一目录 | 待和 B 定格式 |
| 7 | 登记缓存、清单、模型、标准化的指纹；把可用模型分批交给 B | 指纹已写在各 run 的 `config.json`；**推送前需要你确认** |
| 8 | 在开发集上用独立实现重算 Macro-F1，核对"新人新夜 − 原人原夜 = 两个拆分差之和"；与第一周 9 名单参考值对照（pool36 ≈ +1.69，dev8 ≈ +1.47） | 等 A（01、07、09）和 B（02、08）跑完 |
| 9 | 用 B 冻结的预测独立重算两张主表，逐名单对齐 B | 等 B |
| 10 | 签出独立复算和复现包；汇总一页决定 | — |

## 04、06 开发分数（仅开发评分，Macro-F1 %，head seed 0，单次拟合）

| 名单 | 评分集 | 原人原夜 | 原人另一夜 | 新人新夜 | 换夜收益 | 新人较另一夜 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 04 | pool36 | 51.89 | 53.78 | 55.01 | +1.89 | +1.23 |
| 04 | dev8 | 56.12 | 61.24 | 63.60 | +5.11 | +2.36 |
| 06 | pool36 | 57.03 | 57.21 | 59.06 | +0.19 | +1.85 |
| 06 | dev8 | 65.68 | 64.68 | 64.94 | −0.99 | +0.25 |

数据来源：

- 04 的两组旧条件来自 C 第一周的 `paired_macro_f1.csv`；06 的来自 D 第一周的 `summary.json`。
- 新组是本次拟合，分数已由预测表重算核对。

加上 C 的 03、10，目前 4 份名单的情况：

- **pool36 上"新人较另一夜"4 份全部为正**：+0.00、+0.68、+1.23、+1.85。
- **换夜收益有正有负**：+2.00、−0.95、+1.89、+0.19。

只有 4/9 份名单，而且是开发集，**不能下结论**；正式判读要等预留 8 人和 9 份名单配对区间。

## 文件

- `01_check_matched_blocks.py`：D 独立核对 9 份另一夜选块，结果在 `out/matched_blocks_check_D.csv`。
- `02_replay_and_dev_table.py`：重放 06–10 的旧头，并生成 04、06 的开发三组表。
- `out/old_heads_D_06_10.csv`、`out/old_heads_all20.csv`：旧头清单，含 SHA，以及在 Release 中的位置。
- `out/dev_three_arms_04_06_D.csv`：04、06 的开发分数。
- `runs/{04,06}/other_night/`：`config.json`（代码、名单、特征、权重、标准化指纹）、`loss.csv`、`metrics_dev.json`。

Release [`d-budget-week2-other-night-v01-20261127`](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-budget-week2-other-night-v01-20261127)
里有 `head.pt`、`normalization.npz`，以及 pool36 / dev8 的逐段预测（11 列）。SHA256 记录在 `registry/artifact_index_D.csv`。
06–10 的旧头仍在第一周 Release `d-budget-week1-results-v01-20261126` 里。

运行命令（从 C 的第二周交付包解压根目录运行）：

```text
python experiment3/week2_budget_20261127/train_other_night.py --plan experiment3/week2_budget_20261127/rosters --old-plan experiment3/week1_budget_20261126/rosters --data data --bundle <D full shared cache> --output runs --feature-info-sha256 392122c602030d8dcfa903a8d5a5f55fd582f0e09dc6d086248c86bac4267bdf --roster 04 06
```

预留 8 人和封存测试 16 人均未评分。
