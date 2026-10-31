# B 第二周进度与待办(实验②)

## 最新状态(2026-10-31,CNN 补齐后)

**B 的 40 个 run 全部完成:EEGDM 四档线性头 20 个、EEGDM 上下文 10 个、CNN 上下文 10 个。** A 的 50 格矩阵已并入统一口径,缺项表 A 50/50、B 40/40、C 80/80、D 0/60。
本节优先于下文所有较早记录。还差一件:D 的 D0/D1/D2 预测进来后,做半监督曲线与全组最终汇总。

- **按人分析**:见 `b_by_subject_analysis.md`。EEGDM 与 CNN 的优势落在不同的人身上,总体差值小是方向相反的差异相互抵消,不代表两者在每个人身上都接近。

- **A 的输入**:Release `a-cnn-w2-v01-20261031`。63 个预测表全部通过官方 evaluate v1 + 验证参考清单,样本集合相同,重算值与 `a_supervised_comparison.csv` 不一致 0 项(`external_A/`)。复用的 2 个 E1 run 预测不在 Release 内,本次未重算,核验见 `a_reuse_check_w1.md`。
- **CNN 特征**:`scripts/19_extract_cnn_features.py` 用同 seed 的 25% 双通道编码器在 CPU 上提取,53496 段×128 维;权重与标准化 SHA 与 `encoder_index.csv` 一致。用 A 的分类层接特征还原验证预测,类别不同的段每 seed 0–4 段(21186 段中),全部是前两类概率差小于 1e-3 的打平段,概率最大偏差 1.23e-3,来自 CPU 与 A 的 GPU 的浮点差(`b_cnn_extraction_check.csv`)。
- **设备**:上下文头均在 CPU 上运行。

| 25% 上下文(BiGRU 头,5 seed 均值) | Macro-F1 | Macro-F1 种子标准差 | N1 F1 | REM F1 |
| --- | --- | --- | --- | --- |
| EEGDM 长度 1 | 0.7544 | 0.0018 | 0.4298 | 0.7137 |
| EEGDM 长度 3 | 0.7719 | 0.0026 | 0.4686 | 0.7500 |
| CNN 长度 1 | 0.7382 | 0.0102 | 0.3947 | 0.7222 |
| CNN 长度 3 | 0.7597 | 0.0141 | 0.4275 | 0.7743 |

- **长度 3 减长度 1**:EEGDM 的 Macro-F1 +1.75pp、N1 F1 +3.88pp、REM F1 +3.63pp;CNN 的 Macro-F1 +2.14pp、N1 F1 +3.28pp、REM F1 +5.21pp。CNN 的 Macro-F1 按人区间 5 个 seed 都不含 0,N1 F1 有 2 个含 0,seed 4 为负。
- **EEGDM 减 CNN(25%,上下文头)**:长度 1 为 Macro-F1 +1.62pp、N1 F1 +3.51pp;长度 3 为 +1.22pp、+4.11pp。Macro-F1 按人区间在长度 1、3 各有 4 个 seed 含 0,不能据此说 EEGDM 更好。REM F1 反而是 CNN 高(长度 3 差 2.44pp)。

| 监督预算曲线(5 seed 均值) | 5 人 | 14 人 | 27 人 | 54 人 |
| --- | --- | --- | --- | --- |
| EEGDM 线性头 Macro-F1 | 0.7321 | 0.7456 | 0.7684 | 0.7740 |
| CNN 双通道 Macro-F1 | 0.7183 | 0.7420 | 0.7710 | 0.7788 |
| EEGDM 线性头 N1 F1 | 0.3845 | 0.4066 | 0.4655 | 0.4901 |
| CNN 双通道 N1 F1 | 0.3724 | 0.3862 | 0.4747 | 0.4904 |

- **EEGDM 减 CNN(预算曲线)**:20 组配对的 Macro-F1 按人区间全部含 0。名义上 EEGDM 在标签少时高,CNN 在标签多时高,这个方向只是开发集线索,不是结论(`external_A/paired_comparison.csv`)。
- **比较口径**:EEGDM 主干由外部数据预训练,CNN 只见过本预算有标签者,这是表征比较,不是纯架构比较。上下文头的 CNN 特征来自已在同 14 人上训练过的编码器,而 EEGDM 主干没见过这些人的标签,两者的训练特征并不等价。BiGRU 长度 1 与线性头、A 的端到端 CNN 都不是同一模型,不能互相替代。8 人仅为开发集探索性证据。
- **run 文件**:小文件在 `reports/week2/runs_B/`;`best.ckpt` 与 `normalization.npz` 在 Release `b-week2-runs-v01-20261031`,SHA 见 `artifact_index.csv`。汇总表中的 `runs/...` 路径对应这两处的同名目录。资源表未记录内存峰值,写 NA。提取特征只在本地 `features/a_w2_v01/`,可由步骤 W2-B11 重新生成。
- **EEGDM 输入**:C 的 Release `c-pooled-features-v01-20261031`,EOG 是 C 本地重建,不是 D 的原始缓存;10%/25% 与 C 的 S0 一致,100% 与 D 的多 seed 结果一致。

## 较早记录

截至本记录,B 的**工具链与核验已完成**,**真实实验尚未开始**:缺冻结特征与 A 的编码器。
下表的"已验证"只指合成数据或现有第一周产物上的检查,不代表任何第二周成绩。

## 已完成并有证据

| 步骤 | 产物 | 证据 |
| --- | --- | --- |
| 1 评估脚本与负面测试 | `src/evaluation/evaluate.py`(新增 `--reference`,指标定义不变)、`tests/test_evaluate_w2.py` | 缺样本/多样本/错标签/重复 ID/概率异常均报错;v1 的 8 项原测试仍过 |
| 1 应有样本清单 | `manifests/val_reference_v1.csv`(21186 段、8 人;只读验证者 NPZ 的 labels) | 与 E1、E2 预测表逐段一致 |
| 2-3 v2 名单核验 | `reports/week2/b_v2_verification.md`、`scripts/10_verify_week2_budgets.py` | 21/21 项:嵌套、与验证/测试互斥、rank 截断、5 人=11936 段/N1 1719、25%=14 人/27 晚/32310 段与第一周一致 |
| 7-8 窗口索引 | `reports/week2/window_index.csv`、`scripts/11_make_window_index.py`、`tests/test_window_index.py` | 训练窗 32310(14 人)、验证窗 21186(8 人);缺口不跨越,边缘复制中心并记 mask;验证中心集合与参考清单相同 |
| 9 上下文配置锁定 | `reports/week2/context_config.yaml` | 结构/超参在看到任何真实结果前写定 |
| 4-6,10 训练与汇总工具 | `scripts/12_eegdm_budget_head.py`、`13_context_head.py`、`14_w2_aggregate.py`、`15_plot_budget_curves.py`、`w2_common.py` | `tests/test_w2_heads.py`(合成数据):六件套、其余训练者特征行未被读取、标准化只来自有标签者、长度 1/3 中心集合相同、配对前样本集合校验 |
| 全组评估 | `reports/week2/b_prediction_audit.csv`、`scripts/16_audit_predictions.py` | 五个分支共 22 个不同内容的预测表全部通过严格检查并覆盖同一 21186 段 |

## 被什么挡住(都不是 B 能单方面解决的)

| 要做的事 | 缺什么 | 找谁 |
| --- | --- | --- |
| 10%/50% 的 EEGDM 线性头 ×5 seed(8 个 run) | EEG 特征 `eeg_cond14_v01` 与 EOG 特征 `eog_cond14_v01`(各约 23.67 GB,不在仓库) | C(EEG)、D(EOG) |
| 25% 长度 1/3 上下文,EEGDM 表征 ×5 seed(10 个 run) | 同上 | C、D |
| 25% 长度 1/3 上下文,CNN 表征 ×5 seed(10 个 run) | A 的 25% 双通道 CNN 各 seed 编码器逐段特征 | A |
| 四档曲线的 CNN 部分 | A 的双通道 CNN 预测(10/25/50/100%,各 seed) | A |

CNN 特征文件格式(写给 A):`np.savez(path, sample_id=<str 数组>, x=<float32 [N,F]>)`,须覆盖该 seed 对应的
14 名有标签者与 8 名验证者的全部片段,用 eval 模式导出;`13_context_head.py` 缺任何 sample_id 即报错。

## 拿到输入后的命令

```bash
# 10%/50% 监督头(label25/100 仅在严格匹配时才复用 D 的旧结果,否则同样重跑)
for b in label10 label50; do for s in 0 1 2 3 4; do
  python scripts/12_eegdm_budget_head.py --eeg-dir <eeg_cond14_v01> --eog-dir <eog_cond14_v01> --budget $b --seed $s
  # 或池化输入:--eeg-pooled e.npz --eog-pooled o.npz --eeg-index <feature_index.csv>
done; done
# 上下文:两表征 × 长度 1/3 × seed 0-4
for L in 1 3; do for s in 0 1 2 3 4; do
  python scripts/13_context_head.py --repr eegdm --eeg-dir <eeg> --eog-dir <eog> --context $L --seed $s
  python scripts/13_context_head.py --repr cnn --cnn-features <A_seed${s}_enc.npz> --context $L --seed $s
done; done
# 汇总、配对、登记、作图
python scripts/14_w2_aggregate.py --runs-root runs --out-dir reports/week2 --pair-context --update-registry
python scripts/15_plot_budget_curves.py --summary reports/week2/seed_summary.csv
python scripts/16_audit_predictions.py <各组分支引用> --out reports/week2/b_prediction_audit.csv
```

## 已知问题与口径提醒

- 旧的 25%/100% EEGDM 头要复用,须核对名单、标准化、特征 SHA、头、种子、更新预算、选模和评价样本。
  D 的 `train_linear_concat.py` 固定了 `torch.manual_seed(0)` 与确定性算法(多 seed 脚本我没读过);`12_eegdm_budget_head.py` 沿用同一配方,
  但**尚未用真实特征做过逐位复现**,所以现阶段不把任何旧结果登记为 reused。
- 10% 是 5 人(9.26% 受试者),不是严格 10% 的片段;图表横轴为人数。
- 5 seed × 8 人不是 40 个独立样本;seed_summary 与 paired_comparison 分开报告。
- `run_registry.csv` 目前没有任何 B 的真实 run 行。

## 更新:C 的第二周结果到货后(2026-10-31)

| 事项 | 产物 | 结果 |
| --- | --- | --- |
| 并入 C 的 80 个 run | `scripts/17_ingest_external_runs.py`、`reports/week2/external_C/`(metrics_summary、seed_summary、paired_comparison 60 组、by_subject、mismatch_vs_reported) | 80 份预测全部通过严格检查并覆盖同一 21186 段;与 C 登记的 Macro-F1/N1/REM 指标 **0 项不一致**;S1 减 S0 的均值差都在约 ±0.7 个百分点内(10% EEG+EOG 的 N1 F1 为 −0.69pp,其余接近 0);按人区间多数含 0 |
| 缺项表 | `scripts/18_gap_table.py`、`reports/week2/b_gap_table.csv` | B 40 个格子 0 个完成;C 80/80;A 0/50、D 0/60 在各分支里未找到(不代表对方本地没有) |
| 池化特征输入 | `--eeg-pooled/--eog-pooled/--eeg-index`(`w2_common.py`) | 只需每模态约 1.6 GB 的池化特征,不必传 13.5 GB 原始文件;与原始目录输入的预测逐位相同(有测试) |
| 按人 CV 草案 | `reports/week2/b_cv_draft.md` | 仅讨论,不改变划分 |
| **修复**:种子顺序 | `w2_common.seed_everything` | 之前种子在建模型之后才设,同一 `--seed` 两次结果不同(测试发现);现在先设种子再建模型,与 D 的脚本顺序一致 |

池化特征文件格式:`np.savez(path, sample_id=<str 数组>, x=<float32 [N,2560]>)`,为 `[20,30,128]` 在 30 个窗上取均值后拉平的**未标准化**特征;
`--eeg-index` 指向对应的 `feature_index.csv`(提供 split、y_true)。

C 的 EEGDM EEG+EOG S0 头在 10%/25% 各有 seed 0–4(EOG 是 C 的本地重建,非 D 缓存)。要登记为 B 的 10% 头复用,还需逐项核对
名单、标准化、特征 SHA、头、种子、更新预算、选模,**目前没有登记**。

## 更正

此前版本曾写"D 的 `official_metrics.csv` 疑似列错位"。**该说法是错的,已删除。** 我把 `models_summary.csv` 里相邻的
`accuracy`、`F1_W`、`F1_N2` 列误读成了区间。`official_metrics.csv` 14 行的点估计都落在各自 95% 区间内
(例:EEGDM_EEGEOG_cond14 Macro-F1 0.7766 [0.7297, 0.8050],N1 F1 0.4913 [0.4483, 0.5341]),D 的文件无需复核。
