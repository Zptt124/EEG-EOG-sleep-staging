# A 的运行索引(run index)

负责人:A · 更新日期:2026-10-18 · 分支:`feat/a-data-baseline`

## 运行记录

| run_id | 内容 | 状态 | best epoch | val Macro-F1 | val Acc |
|--------|------|------|-----------|--------------|---------|
| `E1_EEG_label100_seed0_v01` | 小 CNN,EEG Fpz-Cz,100% 标签 | 成功 | 7 | 0.7645 | 0.8182 |
| `E1_EOG_label100_seed0_v01` | 小 CNN,EOG horizontal,100% 标签 | 成功 | 11 | 0.6827 | 0.7759 |

- 指标明细:本目录 `metrics_E1_EEG_label100_seed0_v01.json` / `metrics_E1_EOG_label100_seed0_v01.json`
  (含每类 precision/recall/F1/support、N1 指标、混淆矩阵)
- 配置:本目录 `config_E1_*.yaml`(两模型仅通道与归一化参数不同,结构/超参/种子一致)
- 汇总报告:[结果报告.md](结果报告.md);数据检查:[data_checks.md](data_checks.md);
  预处理规则:[preprocess.yaml](preprocess.yaml)
- 完整运行目录(README/history/best.ckpt/predictions_val.csv)不在 git 内,
  随交付包 `实验①交付包_20261018.zip` 分发(其 `5_运行记录_完整run文件夹\` 子目录)

## 逐段预测表(交接点⑤,给 D)

21,186 行/表,两表 sample_id 集合与 y_true 完全一致;格式见仓库根 `predictions_val.template.csv`。

| 文件 | sha256 | 位置 |
|------|--------|------|
| predictions_val.csv (EEG run) | `9d9c360ae16b0e0ce9309515a233136bd7d5c768e5b280e2fe81eaa099dec1af` | 交付包 `3_验证集预测_给D\predictions_val_EEG.csv` |
| predictions_val.csv (EOG run) | `993d70832654ba230d889fdb5a0ef708f4d560395ec91ccd068f49acb35a95c3` | 交付包 `3_验证集预测_给D\predictions_val_EOG.csv` |

## 预处理数据(交接点③,给 B/C)

153 个 `<record_id>.npz`(signal(n,2,3000)µV + labels + epoch_index),共 195,469 段,
位于交付包 `2_预处理数据与脚本_给B和C\data\processed\`;读取与训练代码在仓库 `scripts/`。

## 评估说明

- 分数用 A 的临时版 `scripts/evaluate.py` 计算;**B 的官方 evaluate.py 交付后需重算**
- 测试集(16 人 / 41,345 段)本周未使用
