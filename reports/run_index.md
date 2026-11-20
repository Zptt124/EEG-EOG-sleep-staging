## W4 运行记录（2026-11-19，分支 feat/a-week4-pooling-cv）

同窗 EEG 条件 30k 小 EEGDM（m64x12）的 EOG-only 线性探针（B 34 号协议：训练侧五折 50 轮
OOF 选轮 + OOF 选 α；开发集只最终评分一次）。权重 Release `a-w4-xmodal-same-30k-v01-20261119`。

| run_id | 内容 | 状态 | E* | dev Macro-F1 | N1 AP | α | N1 F1(调整后) |
|--------|------|------|----|--------------|-------|---|---------------|
| `W4_A_xm_sync_label10_sseed2024_seed0_v01` | 同窗30k EOG-only 探针,label10 | 成功 | 44 | 0.5967 | 0.2475 | 7.61 | 0.3062 |
| `W4_A_xm_sync_label10_sseed2024_seed1_v01` | 同上,下游 seed1 | 成功 | 24 | 0.6013 | 0.234 | 6.73 | 0.305 |
| `W4_A_xm_sync_label10_sseed2024_seed2_v01` | 同上,下游 seed2 | 成功 | 40 | 0.5915 | 0.240 | 9.04 | 0.304 |
| `W4_A_xm_sync_label25_sseed2024_seed0_v01` | 同窗30k EOG-only 探针,label25 | 成功 | 44 | 0.6404 | 0.310 | 5.54 | 0.376 |

指标与 bootstrap 区间见各 run 目录 metrics.json（小文件归档于
`experiment3/week4/pool_compare/probe_results/<run_id>/`）；完整 run 目录（含 README）在本地
`runs/<run_id>/`。LFT 提速对照（1s/2s/3s/去classifier/downsampling）见
`experiment3/week4/pool_compare/lft_results/` 与 `reports/week4/a_w4_pooling_compare.md`。

### W4 LFT 提速对照 run（2026-11-21 完成，seed0）

| run_id | 内容 | E* | OOF Macro | dev Macro | dev N1 F1 | 单独每轮 |
|--------|------|----|-----------|-----------|-----------|----------|
| `lft_w1s_seed0_v01` | 1s std LFT(基线) | 16 | 0.7222 | 0.7283 | 0.3452 | 109 s |
| `lft_w2s_seed0_v01` | 2s std LFT(推荐) | 15 | 0.7209 | 0.7497 | 0.4012 | 106 s |
| `lft_w3s_seed0_v01` | 3s std LFT(提速②) | 16 | 0.7254 | 0.7384 | 0.3758 | 100 s |
| `lft_w2s_nocls_seed0_v01` | 2s 去 classifier(提速②) | 9 | 0.7206 | 0.7524 | 0.4167 | 91 s |
| `lft_w2s_ds100_seed0_v01` | 2s 信号 100Hz(提速②) | 19 | 0.7261 | 0.7401 | 0.3705 | 103 s |

小文件归档于 `experiment3/week4/pool_compare/lft_results/<run_id>/`；单独校准数据
`experiment3/week4/pool_compare/timing_calibration.json`（脚本 40 号）。

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
