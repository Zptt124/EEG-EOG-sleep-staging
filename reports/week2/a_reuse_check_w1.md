# A 第一周 run 复用核验(W2 v1.1 label100 seed0)

负责人:A · 日期:2026-10-31 · 依据:协议v2"旧run复用须核对名单、标准化、编码器与
特征SHA、头、种子、更新预算、选模及评价样本"。

两个第一周 run 被登记复用为 W2 `label100` seed0 的 EEG/EOG 单模态监督基线:

| run_id | 位置 | 复用为 |
| --- | --- | --- |
| `E1_EEG_label100_seed0_v01` | 交付包/实验①交付包_20261018 + 本机 runs/ | W2 `label100` EEG 单模态 seed0 |
| `E1_EOG_label100_seed0_v01` | 同上 | W2 `label100` EOG 单模态 seed0 |

## 逐项匹配证据

| 核验项 | W2 要求 | 第一周 run 实际值 | 结论 |
| --- | --- | --- | --- |
| 名单 | label100 = 训练 54 人全部 | `label_subsets_v2.csv` label100 记录集合 == 训练划分 106 记录(集合相等,已程序断言) | ✅ |
| 划分 | 读 `split_subjects.csv`,54/8/16 | 第一周加载逻辑按 train/val 划分,划分文件发布后未改(seed42,2026-10-18 00:52) | ✅ |
| 模型 | 团队文档 SmallCNN,单模态首层输入通道 1,不加层不加上下文 | `05_train_cnn.py` 结构相同;新 `cnn_model.SmallCNN(in_channels=1)` **strict load_state_dict 通过**(a_w2_checks.py [5]) | ✅ |
| 标准化 | 每通道标量 z-score,只拟合有标签者 | 第一周:训练集(54 人)拟合;label100 有标签者即全部 54 人。数值复算 **逐位一致**:EEG mean=-0.10209347 std=21.15585709;EOG mean=-2.59598804 std=49.51515579(与 config.yaml 完全相同) | ✅ |
| 种子 | 下游 seed=0 | config `seed: 0` | ✅ |
| 训练规则 | Adam 1e-3、batch 128、max 60 epoch、验证 Macro-F1 早停 patience=10 | 第一周 config 与 `train_cnn_w2.py` 默认值一致;EEG 17 epoch / EOG 21 epoch 早停 | ✅ |
| 选模 | 按验证 Macro-F1 选最佳权重,同一权重报 N1 | best.ckpt 记录 val_macro_f1 为选模依据 | ✅ |
| 评价样本 | `manifests/val_reference_v1.csv` 21,186 段,8 人 | 两表均 **通过 B 官方 evaluate.py v1 --reference 核对**(样本集合一致、y_true 一致、概率/ID 校验通过) | ✅ |
| 官方指标 | B 官方 evaluate.py | EEG Macro-F1 0.7645 / EOG 0.6827,与第一周 metrics.json 数值一致 | ✅ |
| 数据权限 | 监督仅用有标签者,无未标注信号 | 第一周只读训练集信号与标签,未使用无标签信号 | ✅ |

## 差异与说明

- 第一周 run 无 `normalization.json` 独立文件:标准化参数在 `config.yaml#normalization`
  (mean/std/computed_on),登记表中 `normalization_path` 指向该位置。后续若 B/D 需要,
  可用 `train_cnn_w2.py` 的同款统计格式补写(数值已逐位验证一致)。
- 第一周 `metrics.json` 由 A 临时 `evaluate.py` 产出;本核验已用 B 官方 evaluate.py v1
  重算并落盘为 `metrics_official.json`(数值一致,见 B 的 reports/b_recompute_check.md)。
- 第一周 run 未记录 GPU 显存/耗时等资源字段:资源行不补,在 run README 中注明
  "第一周 run,资源字段未记录"。

## 结论

两个第一周 run 逐项满足 W2 `label100` seed0 复用条件,登记 status=`reused`,
不计入 W2 新训练(50 次主矩阵中省 2 次,实际新训练上限 48 次)。
