# B 第四周：跨模态三组的配对、探针与统一复评

这三个工具不依赖 D 的公共训练脚本，先写好以便后面直接用。规则依据 C 的[跨模态三组对照规则](../../experiment3/week4/cross_modal_protocol.md)，名单和验证窗沿用 [b_pretrain_val_for_AD.md](b_pretrain_val_for_AD.md) 的 `w4_v01`。

## 1. 错时 EEG 配对：[scripts/33_b_w4_shift_pairs.py](../../scripts/33_b_w4_shift_pairs.py)

- **目标窗和候选窗**：
  - 目标窗是 5 秒 EOG 窗 `(record, epoch_index, k)`，起点为 30·epoch + 5·k 秒。
  - 候选是同一记录里的 Fpz-Cz 5 秒窗，起点间隔满足 60 ≤ |Δ| ≤ 300 秒，两端都包含。
  - 「完整有效」指候选窗所在的 30 秒段存在于 A 的预处理 NPZ 里，配对全程不读睡眠标签。
- **抽样**：在候选里均匀抽 1 个，抽样逐窗确定，用 splitmix64(窗键 ⊕ 种子 20261118) 算出。
  - 一个窗的配对只取决于它自己和所在记录，跟窗清单的范围、顺序都无关。
  - D 锁定的训练窗清单只要是全表的子集，结果就和全表一致。
  - A、B 也可以在数据加载器里直接调用 `pair_windows(record_id, epoch_index, k)` 现场计算。
- **覆盖率**（46 人预算训练池的全部窗）：

| 项目 | 结果 |
| --- | --- |
| 目标窗 | 684,246 个，全部有候选，**覆盖率 100%**，没有需要置空的窗 |
| 候选数 | 中位 98（两侧都完整时就是 98 个），1% 分位 74，最少 1 |
| 候选不满 98 的窗 | 19,338 个（2.8%），在记录两端或裁剪造成的段缺口附近 |
| 只有单侧候选 | 只在前面 1,128 个，只在后面 1,128 个 |
| 方向 | 条件窗在目标窗之后的占 50.01% |
| 间隔 | \|Δ\| 平均 179.8 秒、中位 180 秒，恰好等于 ±60 秒的 14,009 个，恰好等于 ±300 秒的 13,748 个 |

  - 在候选完整的 664,908 个窗上，98 个间隔值的计数用卡方检验均匀性，p = 0.11。
  - B 初版的 2,000 个固定验证窗位置也全部配上了，结果见 [shift_pairs_val_windows_w4_v01.csv](b_shift_pairs/shift_pairs_val_windows_w4_v01.csv)。正式 val MSE 已改用 D 的窗清单，等清单入库后用 `--windows` 给它重新配对。
- **输出**：
  - 统计文件在 [b_shift_pairs/](b_shift_pairs/)：`coverage_summary.json`（含各表 SHA256）、`coverage_by_subject.csv`、`offset_hist.csv`（每个间隔值的计数）。
  - 全训练池的配对表约 56 MB，没有入库，写在 `cache/w4_shift_pairs/`。几秒就能重建，SHA256 登记在 summary 里。
- **D 的窗清单出来后**，用这条命令核对覆盖情况：

```bash
python scripts/33_b_w4_shift_pairs.py --windows <D 的窗清单.csv> --out <输出.csv>   # 需含 record_id, epoch_index, window_k
```

## 2. 第四周探针：[scripts/34_b_w4_probe.py](../../scripts/34_b_w4_probe.py)

- **配方和选模**：和第三周 27 号完全一样，都是预算内按人 5 折、训练侧选轮，全名单重训后开发集只预测一次。改动只有三处：
  - 模态可选（`--modalities eog|eeg|eeg+eog`），跨模态三组用 `eog`。
  - run_id 前缀改成 `W4_`。
  - 把 N1 分数调整并进同一次运行：在 E\* 轮的 OOF 上选 α，开发集只应用一次。
- **metrics.json**：多出 `n1_ap`、`n1_auroc` 和 `n1_adjust` 三项，`n1_adjust` 里有 α、原始和调整后的 N1 F1、调整后的 Macro-F1。另外写出 `predictions_val_n1adj.csv`。
- **arm 命名建议**：`xm_sync`（A）、`xm_shift`（B）、`xm_none`（D）。
- **核对 1**：用 C 的 S0 特征按 `eeg+eog` 跑 label10 seed 0，21,186 行预测与已发布的 `W3_B_std_t1_label10_sseed2024_seed0_v01` 逐行一致，概率最大差为 0。
- **核对 2**：同一份特征只用 EOG 跑，Macro-F1 0.583，N1 AP 0.223，N1 F1 0.193，α=9.24 调整后为 0.308。这是官方 EEGDM 在 EOG-only 下的参考值，只用于冒烟测试，不属于三组比较。

```bash
python scripts/34_b_w4_probe.py --group B --arm xm_shift --budget label10 --seed 0 \
    --eog-pooled <eog.npz> --index <sample_index.csv> --feature-config '{"backbone":"...","step":30000}'
```

## 3. 三组统一复评：[scripts/35_b_w4_crossmodal_eval.py](../../scripts/35_b_w4_crossmodal_eval.py)

- **核对**：每份预测先对照 `val_reference_v1.csv` 检查覆盖和真值。然后同一 budget、同一 head seed 的三组逐行对齐，sample_id、受试者、记录、epoch_index（即时间）、真值有任何不一致就报错停止。
- **输出**：
  - 绝对分数：Macro-F1、N1 AP、N1 F1、N1 AUROC，以及 34 号写入的 α 和调整后指标。
  - 三个配对差：同窗−无 EEG、错时−无 EEG、同窗−错时。各 seed 合并后按开发集受试者成对 bootstrap 1000 次，给出 95% 区间（和 29 号脚本同一实现），另报逐 seed 差值。
  - 文件：`metrics.csv`、`summary.csv`、`paired.csv`、`paired_by_subject.csv`、`alignment_check.csv`、`eval_run.json`。
- **核对**：用上面三个冒烟测试 run 当作三组，跑通了全部输出。把其中一份的 epoch_index 改掉一行后，脚本报错停止。

```bash
python scripts/35_b_w4_crossmodal_eval.py --registry <A.csv> <B.csv> <D.csv> \
    --sync-arm xm_sync --shift-arm xm_shift --none-arm xm_none
```

登记表的列和第三周一样：`run_id,group,arm,budget,seed,predictions`。

## 还要等的

- 错时组的训练要等 D 的公共脚本（第 2 天）和无 EEG 的 30k checkpoint（第 3 天）。条件编码器和 dropout p=0.2 要和 A 共用同一套实现。
- 拿到 D 锁定的训练窗清单后，用 `--windows` 再核对一次覆盖率。
