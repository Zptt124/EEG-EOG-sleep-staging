# B 第二周进度与待办(实验②)

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

- **D 的 `official_metrics.csv` 疑似列错位**:`EEGDM_EEGEOG_cond14` 点估计 Macro-F1 0.7766,区间却是
  [0.8237, 0.9161],N1 F1 点估计 0.4913 而区间下限 0.8604,点估计落在区间外。预测表本身通过核验、
  重算的点估计与 D 一致,问题在区间列。需 D 复核后再引用该文件的区间。
- 旧的 25%/100% EEGDM 头要复用,须核对名单、标准化、特征 SHA、头、种子、更新预算、选模和评价样本。
  D 的 `train_linear_concat.py` 固定了 `torch.manual_seed(0)` 与确定性算法(多 seed 脚本我没读过);`12_eegdm_budget_head.py` 沿用同一配方,
  但**尚未用真实特征做过逐位复现**,所以现阶段不把任何旧结果登记为 reused。
- 10% 是 5 人(9.26% 受试者),不是严格 10% 的片段;图表横轴为人数。
- 5 seed × 8 人不是 40 个独立样本;seed_summary 与 paired_comparison 分开报告。
- `run_registry.csv` 目前没有任何 B 的真实 run 行。
