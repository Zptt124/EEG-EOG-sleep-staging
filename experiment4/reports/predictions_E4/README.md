# 实验④（D）的验证集逐段预测

这里只放 **D 自己产生的** 9 份预测表。A、B、C 的预测表已经在他们各自的提交里，不重复入库（对应关系见 `../run_index.csv`）。

- 全部是验证集：8 人，21,186 段；测试集未使用。
- 格式统一为 11 列：`sample_id, subject_id, record_id, epoch_index, y_true, y_pred, p_W, p_N1, p_N2, p_N3, p_REM`。`y_pred` 取五个概率中最大的那一类。
- 每份表的 `sample_id` 集合、`y_true` 都和 A 的 `E1_EEG_label100_seed0_v01` 完全一致（`scripts/prepare_inputs.py` 会再核对一遍）。

| 文件 | 怎么来的 | Macro-F1 |
|---|---|---|
| `predictions_val_E4_CNN_fusedavg_label100_seed0_v01.csv` | `fuse_avg.py`：A 的 E1_EEG + E1_EOG 概率等权平均 | 0.7557 |
| `predictions_val_E4_CNN_fusedavg_label25_sseed2024_seed0_v01.csv` | `fuse_avg.py`：B 的 E2_EEG + E2_EOG（名单 2024，种子 0） | 0.7344 |
| `predictions_val_E4_EEGDM_EOG_cond14_label100_seed0_v01.csv` | 运行 `D_EOG_label100_seed0_cond14_v01`：EOG 特征（条件 14）+ C 的 `train_linear100.py` | 0.6835 |
| `predictions_val_E4_EEGDM_EOG_cond10_label100_seed0_v01.csv` | 运行 `D_EOG_label100_seed0_cond10_v01`：同上，条件 10（对照） | — |
| `predictions_val_E4_EEGDM_fusedavg_cond14_label100_seed0_v01.csv` | `fuse_avg.py`：C 的 E3_EEG_label100 + 上面的 EOG cond14 | 0.7641 |
| `predictions_val_E4_EEGDM_EEGEOG_cond14_label100_seed0_v01.csv` | 运行 `D_EEGEOG_label100_seed0_cond14_v01`：`train_linear_concat.py`，EEG+EOG 拼接 | 0.7766 |
| `predictions_val_E4_EEGDM_EOG_cond14_label25_sseed2024_seed0_v01.csv` | 运行 `D_multiseed_label25_cond14_v01` 中名单 2024、种子 0 的 EOG 头 | 0.6638 |
| `predictions_val_E4_EEGDM_fusedavg_cond14_label25_sseed2024_seed0_v01.csv` | 同一运行中 EEG 头与 EOG 头的概率平均 | 0.7304 |
| `predictions_val_E4_EEGDM_EEGEOG_cond14_label25_sseed2024_seed0_v01.csv` | 同一运行中的拼接头 | 0.7449 |

Macro-F1 由 B 的官方 `evaluate.py` v1 计算，95% 区间见 `../official/official_metrics.csv`。

**注意**：这些都是**单次**结果（种子 0）。比较两个设置时应以多种子结果为准（`../runs_E4/` 里的 `summary_mean_std.csv`）。多种子的其余逐段预测（100%：20 份；25%：60 份）体积较大，保存在 D 本地，未入库。
