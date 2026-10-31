# 实验④（D）第二周交付：跨模态半监督 N1（D0 / D1 / D2）

交付日期：2026-11-01　负责人：D　对应文档：第二周工作文档 v1.1 六、实验④；协议 v2

## 一句话

在 25%（14+40）和约 10%（5+49）标签下，用其余训练者的无标签 EEG/EOG 打伪标签，比较
D0（只用真标签）、D1（普通自训练）、D2（跨模态自训练：EEG-only、EOG-only 教师都同意才收）。
两种冻结表征：R1 = 原始 EEGDM 双模态（5120 维）；R2 = A 的预算匹配 EEG-only + EOG-only CNN 编码器拼接（256 维，不是双通道编码器）。

**矩阵完整：2 表征 × 2 预算 × 3 方法 × 5 下游 seed = 60 个学生 run**，另有 60 个教师头、4 次内部校准、38 个 R2 内折 CNN。

## 给 B：怎么读

| 你要的 | 位置 |
|---|---|
| 逐 run 登记 | `runs/run_index.csv`（60 行；run_id 形如 `W2_D_{D0,D1,D2}_{R1,R2}_{label10,label25}_sseed2024_seed{k}_v01`） |
| 11 列验证预测 | `runs/<run_id>/predictions_val.csv`（已用官方 evaluate.py v1 校验并生成 `metrics.json`） |
| 每 run 其余文件 | `README.md`、`config.yaml`、`history.csv`、`metrics.json`；D1/D2 另有 `pseudo_label_stats.csv` |
| 权重与大文件 | Release `d-ssl-w2-v01-20261101` 的 `w2_d_artifacts_v01.zip`；逐文件 sha256 见 `artifact_index_D.csv` |
| 教师 | `runs/teacher_index.csv`（双模态教师 = 同 seed 的 D0；EEG-only、EOG-only 教师只用于 D2 规则） |
| 资源 | `analysis/resource_profile_D.csv`（列名与 `reports/week2/resource_profile.csv` 相同，可直接追加） |
| 汇总与分析 | `analysis/R*_label*_v01/`：`ssl_comparison.csv`、`n1_direction.csv`、`per_subject.csv`、`bootstrap_paired.csv`、`threshold_control.csv`；`analysis/pseudo_label_by_stage.csv` |
| 名单与校准 | `manifests/v2/`、`calibration/`、`semisupervised_config.yaml` |

**预算曲线标注**：D 的点请按文档另列为 **5+49** 和 **14+40**（有标签人数 + 无标签信号人数），不要和只用有标签者的监督点混称。D0 不使用无标签信号，与监督点同口径。

## 主表（验证集，5 个 seed 均值；按 seed 与按人的配对差见 `analysis/`）

| 表征 | 预算 | 方法 | run 数 | Macro-F1 | N1-F1 | REM-F1 | 平均使用伪标签 |
|---|---|---|---|---|---|---|---|
| R1 | label10 | D0 | 5 | 0.7291 | 0.3674 | 0.6993 | 0 |
| R1 | label10 | D1 | 5 | 0.7353 | 0.3952 | 0.7121 | 24223 |
| R1 | label10 | D2 | 5 | 0.7324 | 0.3808 | 0.7059 | 19998 |
| R1 | label25 | D0 | 5 | 0.7456 | 0.4066 | 0.7236 | 0 |
| R1 | label25 | D1 | 5 | 0.7511 | 0.4399 | 0.7306 | 40261 |
| R1 | label25 | D2 | 5 | 0.7511 | 0.4226 | 0.7436 | 33888 |
| R2 | label10 | D0 | 5 | 0.7041 | 0.2471 | 0.7233 | 0 |
| R2 | label10 | D1 | 5 | 0.7066 | 0.2563 | 0.7224 | 24209 |
| R2 | label10 | D2 | 5 | 0.7036 | 0.2448 | 0.7211 | 20791 |
| R2 | label25 | D0 | 5 | 0.7420 | 0.4044 | 0.7383 | 0 |
| R2 | label25 | D1 | 5 | 0.7455 | 0.4255 | 0.7319 | 40253 |
| R2 | label25 | D2 | 5 | 0.7450 | 0.4137 | 0.7350 | 34928 |

内部校准选出的按类接收比例 p：R1 25% = 0.4，R1 10% = 0.2，R2 25% = 0.4，
R2 10% = 0.2（R2 10% 三个候选都未达到 0.85，按事先规则退回最小 p，`rule_satisfied: false`）。

## 需要 B 知道的几点

1. **R1 的名单**：R1 全部 run 用 `manifests/v1_provisional/`（10% 由 sampling_rank 前 5 推出）。与 B 的 v2 逐人比较 0 处差异、
   片段统计相同（`manifests/roles_v1_vs_v2_note.md`），按严格匹配复用为 v2 结果。R2 全部直接用 v2。
2. **D0 = 第一周 25% 拼接的复现**：R1 25% 的 D0 五个 seed 与第一周 `multiseed_budget.py` 2024 名单结果逐位一致（0.7449/0.7433/0.7431/0.7508/0.7458）。
3. **R2 的 D0 不是 A 的端到端 CNN**：是「A 的编码器（去分类层，eval）+ D 的线性头」，与 R1 同一学生头和训练规则。
   提取核对：用 A 的分类层接我们的特征，复现了 A 的 `predictions_val.csv`（最大概率差 2.8e-3，argmax 变化 48/423720，全部可由浮点误差解释）。
4. **阈值没有用验证集或无标签真标签选**：只在有标签者内部留一人交叉；R2 内折用 A 的 `train_cnn_w2.py --subjects` 重训编码器（见留出者标签的编码器不复用）。
   内折编码器与教师的早停沿用全组规则（验证 Macro-F1），这是 A 接口的固定行为。
5. **标签隐藏**：所有入口通过 `src/label_guard.py` 把无标签者 y 置 −1 并断言；`pseudo_label_trace.csv` 不含无标签者真标签。
6. **路径**：仓库内文件的本机绝对路径已替换为占位符（如 `<D_local_cache>`）；因此 `.py` 的 sha256 与 run 配置里记录的不同，
   对照表见 `code_sha_map.csv`，原版代码在 Release zip 的 `original_code/`。
7. **A 编码器导出的一个小问题**：单模态 `normalization.json` 里 EOG 编码器的通道名也写成 "EEG Fpz-Cz"；D 从 `config.yaml` 读模态，并用上面的复现核对确认通道正确。
8. **EOG 来源**：R1 的 EOG 是 D 第一周提取的 `eog_cond14_v01`（EEGDM 条件 14，与 C 的 EEG 同一主干和预处理）；与 C 第二周本地重建的 EOG 是否逐值一致，D 尚未核对。

## 结论边界（详见方法说明）

- 训练随机性下 D1 相对 D0 有小而稳定的 Macro 提升（多数格 5/5 seed 为正，+0.3~0.6），按人 bootstrap 不显著。
- D1 的 N1 提升主要来自判 N1 倾向的移动：把 D0 的 N1 概率等比放大到相同召回，可复现大部分提升（`threshold_control.csv`）。
- D2 相对 D1 在四格中都没有 Macro 收益，稳定地以 N1 召回换 N1 精确率。
- 只有 1 份抽样名单、8 名验证者同时用于早停，以上均为开发集探索性证据；测试集未触碰。
