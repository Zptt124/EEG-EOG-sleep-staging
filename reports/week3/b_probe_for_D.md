# B → D：第三周线性探针与统一评分入口

D 第 4 天的 random-init 与 pilot-pretrained 配对探针，请直接使用 B 的 [27_b_w3_linear_probe.py](../../scripts/27_b_w3_linear_probe.py)，与 B、A 用同一套训练侧选模，结果可直接放进全组比较。

## 选模协议

- 预算内训练者按受试者分 K=min(5,人数) 折（fold_seed=0），每折标准化只拟合折内训练者，固定训练 50 轮。
- 取拼接留出预测 Macro-F1 最高的轮次，在全部预算内训练者上重新标准化、同 seed 重训，开发集只预测一次。
- 线性头 Linear(EEG 维 + EOG 维, 5)，AdamW lr=1e-3 wd=1e-4，batch 128，CPU 即可（14 人约 1 分钟/格）。

## D 需要准备的文件

每个 checkpoint（random-init、pilot-pretrained）、每个模态各一个 NPZ：

| 键 | 形状 | 类型 | 说明 |
| --- | --- | --- | --- |
| `sample_id` | `(N,)` | NumPy Unicode 字符串 | 与组内 `sample_id` 一致且唯一 |
| `x` | `(N, d)` | `float32` | 未标准化的 30 秒池化特征，d 不限，EEG 与 EOG 可以不同 |

另需一份样本索引 CSV，列为 `sample_id,subject_id,record_id,epoch_index,split,y_true`，至少覆盖 14 名预算内训练者与 8 名验证者，不含测试者。可直接复用 [C 交付的索引](../../experiment3/week3/deliveries/B_week3_features/eeg_mean_t1_cond14_label25/sample_index.csv)（53,496 行）。

请使用相同的 pooling（与 B/C 一致：每秒特征对 30 个时间位置取均值），并在 `feature_config` 中记录 backbone seed、更新步数和层选择。

## 运行

```
python scripts/27_b_w3_linear_probe.py --group D --arm randinit --budget label25 --seed 0 \
    --eeg-pooled <eeg_randinit.npz> --eog-pooled <eog_randinit.npz> --eeg-index <sample_index.csv>
python scripts/27_b_w3_linear_probe.py --group D --arm pilot --budget label25 --seed 0 \
    --eeg-pooled <eeg_pilot.npz> --eog-pooled <eog_pilot.npz> --eeg-index <sample_index.csv>
```

seed 0、1、2 各跑一次，共 6 格。run 目录为 `runs/W3_D_<arm>_label25_sseed2024_seed<N>_v01/`，含 `predictions_val.csv`、`metrics.json`、`config.yaml`、`inner_cv.csv`、`oof_train.csv` 等。

## 交付给 B

- 6 个 run 目录（或至少 `predictions_val.csv`、`config.yaml`、`metrics.json`）。
- 一份登记 CSV，列为 `run_id,group,arm,budget,seed,predictions`，`predictions` 为相对仓库根目录的路径。

B 用 [29_b_w3_evaluate.py](../../scripts/29_b_w3_evaluate.py) 核对 sample_id 后统一复评，配对 `pilot:randinit` 按受试者 bootstrap 报告 Macro-F1、N1 F1 与 N1 AP 的差值。
