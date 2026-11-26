# W5 · B 复评 A（PR #28）与 D（PR #27）（2026-11-25）

## 1. A：temporal context 三臂（PR #28）

**复评方式**：`scripts/43_b_w5_a_temporal_reeval.py`（复用 39 号逐 run 核对与配对流程，只换三臂和三个对比），
输入为 A 分支 `reports/week5/a_temporal_ctx/` 下 12 个 run 的逐段预测、OOF、inner_cv、config、metrics。
输出见 [b_a_tctx_reeval/](b_a_tctx_reeval/)。

**结果：12/12 一致。**

- 每个 run：官方 evaluate.py v1 对照 val_reference（21,186 段、8 人），y_pred = argmax，预测 SHA 与 config 一致；
  在 inner_cv 上重选 E* 与报告一致；五折按受试者划分且与 27 号 fold_seed 0 相同；OOF 受试者 = 预算名单；
  B 用 28 号网格独立重选 α，α 与调整后指标、`predictions_val_n1adj.csv` 全部一致。
- 三臂在每个 budget×seed 上 sample_id 完全对齐。
- 原始配对差（3 对比 × 2 档，点估计与区间）与 A 的 `eval/paired.csv` 最大差 2.8e-7（A 表只保留 6 位有效数字）。
  α 调整后的 Macro-F1、N1 F1 配对差也一致。
- 输入特征 eeg_std / eog_std / index 的 SHA 与 B 本地 W4 主实验 30k 特征（Release d-w4-xmodal-shifted30k-v01）一致。
  B 用 34 号 `--modalities eeg+eog` 在 Mac 上重跑 label10 seed0：E* 36、Macro-F1 0.7059、α 6.22，
  与 A 的 single 臂 21,186 段 y_pred 完全相同，概率最大差 6.7e-6（平台差异）。

**需要 A 修改或补充的地方**

1. **合并冲突**：`WEEK5.md` 与 main 冲突（add/add）。A 分支的版本是 CRLF 换行，并把 C 已登记的那一行改回“待登记”。
   请基于最新 main 只改 A 那一行。
2. **`eval/paired_adjusted.csv` 的 N1 AP 列**：label10 三格与原始表不同（例如 bgru−single +0.58 vs 原始 +1.03）。
   B 的 28 号 α 调整只改判定、不改概率，N1 AP 在调整前后应相同；A 的 42 号在三个 seed 合并后 AP 发生了变化
   （各 seed α 不同，缩放 p_N1 改变了合并后的排序）。报告正文已写“N1 AP 不受 α 影响，同上”并用了原始值，
   结论不受影响；建议把该列改为原始值或删掉，避免被引用。
3. **训练侧 OOF 方向与 dev 不一致，报告里应写明**（seed 合并均值，原始 Macro-F1）：

   | 预算 | 臂 | OOF Macro-F1 | dev Macro-F1 |
   |---|---|---|---|
   | 5 人 | single | 0.7101 | 0.7155 |
   | 5 人 | cat | 0.7135 | 0.6953 |
   | 5 人 | bgru | 0.7026 | 0.7248 |
   | 14 人 | single | 0.7286 | 0.7370 |
   | 14 人 | cat | 0.7311 | 0.7135 |
   | 14 人 | bgru | 0.7274 | 0.7487 |

   BiGRU 在 OOF 上两档都略低于 single，cat 两档都略高于 single，与 dev 方向相反。dev 只有 8 人，
   “5 人档 BiGRU 显著改善”只有 dev 一方的证据，建议在结论里注明训练侧未见同方向差异。
4. **结论措辞**：
   - 5 人档显著的是 α 调整后 Macro-F1（+2.97 [2.25, 4.20]）和原始 N1 F1（+4.69）；原始 Macro-F1
     +0.89 [−0.49, +2.24] 跨 0，调整后 N1 F1 +2.18 [−0.79, +5.49] 跨 0，N1 AP 所有对比都跨 0。
     按组内口径，原始 N1 F1 主要受判定门槛影响，不宜作为主要依据。
   - 14 人档 cat−single 调整后 N1 F1 −15.2pp 来自 cat 臂在 OOF 上选出的 α≈1.07（single 为 4.16），
     原始 N1 F1 −4.64 与 N1 AP −1.23 都跨 0。更准确的说法是“直接拼接在 14 人档的 N1 门槛迁移差”，
     而不是表征变差；且 14 人档只有 1 个 head seed。
   - “14 人档 BiGRU 不亏”：调整后 Macro −1.87 [−4.42, +0.33]，bootstrap 为正的比例只有 5%，建议改为“未检出差别（单 head seed）”。
5. **本机路径**：12 个 `metrics.json` 的 `source` / `sample_set_reference` 写的是 `D:\EEG-EOG-sleep-staging\…`，按协作规则请改成相对路径。

**判定**：数字可以复现，协议与 B 一致。改完第 1 条后可以合并，第 2–5 条建议在合并前一并处理。

## 2. D：连续 30 秒预训练（PR #27）

**能核对的部分**

- `analysis/w5_window30/README.md` 的三张表与 `d_summary.csv`、`d_paired.csv`、`speed_v01.csv` 逐项一致；
  “同信号量多约 60% GPU 时间”（6.13 h / 3.78 h）和“效率约 62%”（733.8 / 1189.6）计算正确。
- 5 秒 30k 基线 `d_pre030k` label10 三个 seed 的 Macro-F1（0.7059 / 0.7327 / 0.7079）与 A 的 single 臂、
  B 的 34 号重跑逐个相同；label25 均值 0.734 与 W4 报告一致。
- 代码：`WINDOW_SECS[5] = (WIN, N_WIN)`，5 秒 run 的 config.json 不新增字段，30 秒与 `--cross-modal` 同用会报错退出；
  停止规则仍用 5 秒标准 val 集（`pool.with_window(5)`，共用内存）。从代码看“5 秒路径不变”成立。

**无法复评的部分 / 需要 D 补交**

1. README“文件”一节列出的 `runs/W5_D_w30_m64x12_seed0_45k_v01/`（config、history、val_mse、val_mse_30s、checkpoints）、
   `analysis/w5_w30/d_n1_adjust`、日志都不在 PR 里；30 秒 probe 的逐段预测和特征没有上传，也没有 Release。
   B 目前无法逐段复评 12 个 30 秒 probe run，OOF 列（README 第 3 节）也无法核对。
2. 配对表只有原始 N1 F1，没有 α 调整后 N1 F1；按组内口径请补上（`d_n1_adjust` 已在本地生成）。
3. 合并顺序：#27 叠在未合并的 #23 上（PR 对 main 显示 104 个文件，实际新增 11 个），D 的 W6 分支又叠在 #27 上。
   需要先合 #23（B 已于 11-17 核对过，见 HANDOFF），再合 #27。

**判定**：汇总表与代码没有发现问题，结论（同信号量未见收益、不建议做正式 5 秒 / 30 秒对照）与表内数字一致。
逐段复评要等 D 补上 run 小文件、probe 预测与 α 调整表之后再做。
