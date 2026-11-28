# 新一轮第 2 周 · B 进度（2026-11-27）

## 1. 02、08 原人另一夜拟合（第 7 天任务）

用 C 的 `train_other_night.py`（第一周 trainer 不变），D 共享缓存（回执 `392122c6…`），seed 0、30 轮、1,800 段、450 次更新。
config、loss、开发分数见 [b_other_night/](b_other_night/)；head.pt、normalization.npz 与逐段预测在 Release `b-budget-week2-other-night-v01-20261127`（zip SHA-256 `251920fb…`）。

入口复现核对：B 在本机重拟 C 的 03，标准化参数逐位相同，头权重最大差 4.5e-8，dev8 / pool36 逐段 argmax 全同，
Macro-F1 与 C 完全一致。

开发分数（Macro-F1 %，仅开发核查，不作判读）：

| 名单 | 评分集 | 原人原夜 | 原人另一夜 | 新人新夜 | 换夜收益 | 新人较另一夜 |
|---|---|---:|---:|---:|---:|---:|
| 02 | pool36 | 62.14 | 59.94 | 64.54 | −2.20 | +4.60 |
| 02 | dev8 | 66.21 | 61.90 | 66.55 | −4.32 | +4.65 |
| 08 | pool36 | 56.67 | 54.82 | 56.31 | −1.84 | +1.48 |
| 08 | dev8 | 63.50 | 60.75 | 60.93 | −2.75 | +0.18 |

## 2. 旧头回放与统一评分器（第 6、8 天任务）

`scripts/51_b_w2_unified_scorer.py` 直接调用第一周 trainer 的 `load_bundle` 和 `score`，推理算术与拟合时相同。

- dev8 回放 26 个头（20 个旧头 + 02、03、04、06、08、10 六个新头）：样本键全同、逐段 argmax 全同，
  概率最大差 4e-6（跨机器；C 同机回放为 1e-16）。见 [b_replay_dev8/replay_check.csv](b_replay_dev8/replay_check.csv)。
- 26 个头的 head.pt / normalization.npz SHA-256 与 D 的 `old_heads_all20.csv`、C 的 `SHA256_manifest.json` 逐个相同。
- 锁定清单草稿 [heads_lock_draft.json](heads_lock_draft.json)（`locked: false`），来源见 [heads_sources.csv](heads_sources.csv)。
  **还缺 A 的 01、07、09 三个新头。**
- 预留 8 人模式的保护已测试：没有 `--lock`、清单未标 `locked: true`、头集合不完整（缺 01/07/09）三种情况都拒绝运行，
  输出目录不会生成；输出目录已存在也拒绝（只评一次）。评分器只写逐段预测，不算指标。
- 预留 8 人覆盖（只看样本键）：D 缓存里 heldout_val 的 8 人、16 夜、18,897 段，与 C 的
  `heldout_expected_coverage_by_night.csv` 逐夜的段数、首末 epoch 完全一致。

## 3. 两张配对表（第 10 天任务）

`scripts/52_b_w2_paired_tables.py`：表一 10 份名单 新人新夜 − 原人原夜；表二 9 份名单 换夜收益、新人较另一夜。
同一受试者跨名单共用 Poisson(1) 权重（1000 次，seed 0），逐人剔除，同号名单数，±1 pp 等效带判读。
dev8 开发核查：表一点估计 +1.06 pp、逐人剔除 +0.82 至 +1.22，与第一周 48 号 dev8 主表一致（区间因抽样顺序不同略有差别）。

## 4. 29 个头锁定与预留 8 人复现（同日更新）

A 交来 01、07、09，并在 `feat/a-w2-budget` 上用 A 自己的脚本完成了预留 8 人评分和第 10 天决定。
B 在 dev8 上回放全部 29 个头，逐段一致后锁定清单 [heads_lock.json](heads_lock.json)；
再用 51 号在预留 8 人上推理，结果与 A 的 29 份逐段预测完全一致，52 号的两张表与 A 的点估计相同。
复核意见见 [b_review_a_heldout.md](b_review_a_heldout.md)。

## 待办

1. ~~A 按复核意见修改 `b_decision.md`（署名、"明显好于"、"相当"、开发集与预留集的方向不一致）。~~
   已完成（2026-11-28）：B 按复核意见直接修改，推在 `feat/a-w2-budget`（445fae9），数字未动。
2. ~~全组确认：实际意义 1 pp、等效带 ±1 pp。~~ 已完成（2026-11-28）：全组在群里确认。
3. C、D 用提交的逐段预测复算两张表。D 已完成（`bd07724`，29/29 预测一致）；C 未见单独复算。

## 5. 复算 C 的 11-28 扩大复验（2026-11-28）

用 53 号从归档混淆计数独立复算 Sleep-EDF 开发复验、Ying、BiGRU、两档预算和关灯窗口：点估计全部复现，区间只差蒙特卡洛误差。
52 号新增"名单和受试者一起重抽"的区间。按这个口径，预留 8 人上三项比较都跨 0，和 C 的结果不矛盾。
见 [b_review_c_followups.md](b_review_c_followups.md)。
