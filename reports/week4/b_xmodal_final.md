# 跨模态三组：B 的统一复评与正式汇总（2026-11-21）

按 C 的协议（`experiment3/week4/cross_modal_protocol.md`「评估与交付」），B 原定第 6 天上午交三组汇总，这份晚了两天。期间 C（PR #25 之后的 `0b1ad3d`）和 D（`dfc7220`）已各自用 B 的 34 号探针跑过三组。B 这次做三件事：

1. **复评 C 的 6 个跨模态 run**：用 B 自己的代码重算全部指标、E*、α 和配对差。
2. **复评 D 的 18 个三 head seed run**：C 的报告引用了 D 的配对表，但 D 的逐段预测没有入库，C 也写明「未独立核验」。B 从 D 的 Release 下载预测后补做了核验。
3. **B 自己从特征重跑 18 个探针**：特征用 D 发布的三组特征，B 重新转换并核对 SHA。这一份作为 B 的正式汇总，并按 B 第四周状态报告里的承诺，和 Mac 上的初步结果对比。

结论：**C、D 的数字全部复现**，三组的判读与 C 一致，没有确认到稳定的跨模态收益。另外，B 在 α 调整后的预测上补算了配对差。原始 N1 F1 有三个区间不含 0，调整后全部跨 0，说明这三个差来自判定门槛。

## 1. 复评 C 的 6 个 run（`39_b_w4_xmodal_reeval.py`）

输入是 C 分支 `0b1ad3d` 的 `experiment3/week4/crossmodal_final_v01`，结果在 [xmodal_reeval/c/](xmodal_reeval/c/)。

| 核对项 | 结果 |
| --- | --- |
| 官方 evaluate.py v1 对照 val_reference：sample_id、y_true | 6/6 通过 |
| 预测 SHA256 与 C 的 `artifact_sha256.json`、config 里登记的值一致 | 6/6 |
| y_pred = 概率 argmax；`predictions_val_n1adj.csv` 等于 B 用 α 重算的结果 | 6/6 |
| 在逐轮曲线上按「OOF Macro-F1 最高，并列取最早」重选 E*，与 C 登记的一致 | 6/6（全部 E*=44） |
| 五折与 27 号的 fold_seed=0 划分一致，OOF 受试者正好是预算名单、不含开发集 | 6/6 |
| B 在 OOF 上独立重选 α，α 和调整后 N1 F1、Macro-F1 与 C 一致 | 6/6，差 < 1e-9 |
| 原始 Macro-F1、N1 F1、N1 AP、AUROC 与 C 一致 | 6/6，差 < 1e-9 |
| 三组逐行对齐（sample_id、受试者、记录、段号、真值） | label10、label25 各 21,186 段，通过 |
| 三个配对差（点估计和 bootstrap 区间）与 C 的 `comparison/paired.csv` 比较 | 最大差 4.7e-8（CSV 舍入） |

C 的 config 里登记的 `val_reference_sha256` 是 `0f71748d…`，B 这边是 `9ad48c0d…`。两者是同一份文件：C 的机器以 CRLF 换行检出，把 B 的文件换成 CRLF 后，SHA 正好是 `0f71748d…`。

## 2. 复评 D 的 18 个 run（同一脚本）

输入是 D Release `d-w4-probe-extras-v01-20261119` 中 `w4_d_probe_runs_v01.zip` 的 `runs_bprotocol/`，α 读 C 收录的 `d_n1_adjust.csv`。结果在 [xmodal_reeval/d/](xmodal_reeval/d/)。

- **18/18 全部通过**：核对项同上。预测 SHA 与 D 的 config.json 一致；E* 在曲线上重选一致；B 在 OOF 上重选的 α 与 D 的表一致。
- D 的 α 表只保留 6 位有效数字，所以 α 及调整后指标按 1e-6 比较（实际最大差 4.9e-7）。α 网格相邻两点相差约 0.7%，真不一致的话差会远大于这个容差。
- 三个配对差与 D 的 `d_paired.csv` 最大差 4.8e-8。C 报告里「三 head seed 补充汇总」那张表因此有了逐段预测支撑，不再只是引用。
- D 的 seed 0 与 C 的 6 个 run 相比，y_pred 逐段 100% 相同。无 EEG、错时两组的概率差 ≤ 4e-6，因为两边用的是同一份特征（B 核对过 npz 的 SHA）。同窗组的概率差 ≤ 2.8e-3，因为 C 的同窗特征是 C 自己重提的，D 的是 D 重提的，字节不同。

## 3. B 的正式汇总：从特征重跑 18 个探针

**特征**：使用 D 的两个 Release 中三组的 04 号 CUDA 特征（30k EMA、t=1、gate、12 层、1 秒 std、空 EEG 条件）。

| 组 | 来源 | B 转换后 eog_std.npz SHA256 |
| --- | --- | --- |
| 无 EEG | `d-w4-xmodal-shifted30k-v01-20261119` / `w4_d_main30k_feat_pool1s_v01.zip` | `3f9f92c5…`（与 C 所用相同） |
| 错时 EEG | 同上 / `w4_b_shifted30k_eog_feat_pool1s_v01.zip` | `6ddf5c6b…`（与 C 所用相同） |
| 同窗 EEG | `d-w4-probe-extras-v01-20261119` / `w4_a_same30k_eog_feat_pool1s_by_d_v01.zip` | `93d3f994…`（D 重提；C 用的是自己重提的 `52e836f1…`） |

转换用 37 号脚本，核对以下几项：各 npy 的 SHA 与 `features_info.json` 一致；EOG-only 验收通过；人员是 14+8；y_true 与 A 的 NPZ 逐段一致。

**同窗特征包的 index.csv 换行问题**：这个包的 `index.csv` 是 LF 换行，而 `features_info.json` 里登记的 SHA 对应 CRLF 版本。B 把它转成 CRLF 后，SHA 正好等于登记值，并且与无 EEG 包的 index.csv 逐字节相同。内容没有问题，建议 D 以后打包时统一换行。

**探针**：34 号，`--modalities eog`，3 组 × 2 个预算 × head seed 0/1/2。18 个 run 的 E*、α 和全部指标与 D 的 18 个逐项相同（39 号自检 18/18，配对表与 D 最大差 3.5e-7）。

完整 run 目录（含 OOF、预测、final.ckpt）打包为 `w4_b_xmodal_formal_runs_v01.zip`。各文件的 SHA 见 [b_crossmodal/artifact_sha256.txt](b_crossmodal/artifact_sha256.txt)，小文件已入库到 [b_crossmodal/runs/](b_crossmodal/runs/)。

### 3.1 绝对分数（三个 head seed 的均值，%）

| 预算 | 组 | Macro-F1 | N1 AP | 原始 N1 F1 | α（范围） | 调整后 N1 F1 | 调整后 Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 人 | 无 EEG | 60.26 | 23.53 | 18.46 | 7.10（5.23–10.37） | 30.16 | 57.38 |
| 5 人 | 错时 | 59.97 | 24.02 | 16.45 | 8.85（8.06–9.24） | 30.87 | 56.74 |
| 5 人 | 同窗 | 59.63 | 24.03 | 16.05 | 7.86（6.93–8.98） | 30.47 | 57.01 |
| 14 人 | 无 EEG | 63.24 | 29.92 | 22.09 | 5.59（4.16–6.54） | 37.66 | 63.80 |
| 14 人 | 错时 | 63.33 | 29.93 | 23.57 | 5.93（3.95–7.34） | 37.21 | 62.65 |
| 14 人 | 同窗 | 63.53 | 30.20 | 22.99 | 6.37（5.79–7.23） | 37.33 | 63.16 |

### 3.2 三个配对差（百分点，三个 seed 合并后按 8 名开发者成对 bootstrap，95% 区间）

原始预测（与 C 报告中引用 D 的那张表相同）：

| 预算 | 比较 | ΔMacro-F1 | Δ原始 N1 F1 | ΔN1 AP |
| --- | --- | --- | --- | --- |
| 5 人 | 同窗−无 EEG | **−0.63 [−1.11, −0.07]** | **−2.38 [−3.49, −0.87]** | +0.48 [−0.32, +1.50] |
| 5 人 | 错时−无 EEG | −0.29 [−1.00, +0.78] | **−1.99 [−3.08, −0.40]** | +0.48 [−0.45, +1.13] |
| 5 人 | 同窗−错时 | −0.34 [−1.25, +0.16] | −0.39 [−1.09, +0.36] | +0.00 [−0.45, +0.80] |
| 14 人 | 同窗−无 EEG | +0.29 [−0.25, +0.85] | +0.93 [−0.03, +2.45] | +0.26 [−0.53, +1.94] |
| 14 人 | 错时−无 EEG | +0.12 [−0.43, +0.67] | **+1.59 [+0.47, +2.48]** | −0.13 [−2.68, +1.24] |
| 14 人 | 同窗−错时 | +0.18 [−0.55, +0.91] | −0.66 [−1.43, +0.76] | +0.39 [−0.69, +2.95] |

**α 调整后的预测**（B 新增，C、D 都没有报；N1 AP 不受 α 影响，与上表相同）：

| 预算 | 比较 | Δ调整后 Macro-F1 | Δ调整后 N1 F1 |
| --- | --- | --- | --- |
| 5 人 | 同窗−无 EEG | −0.43 [−0.80, −0.10] | +0.27 [−0.10, +0.81] |
| 5 人 | 错时−无 EEG | −0.69 [−1.56, +0.43] | +0.67 [−0.48, +2.95] |
| 5 人 | 同窗−错时 | +0.27 [−0.60, +0.86] | −0.40 [−2.19, +0.52] |
| 14 人 | 同窗−无 EEG | −0.65 [−1.21, −0.14] | −0.32 [−1.34, +0.80] |
| 14 人 | 错时−无 EEG | −1.14 [−1.96, −0.20] | −0.45 [−2.20, +1.57] |
| 14 人 | 同窗−错时 | +0.50 [−0.30, +1.18] | +0.13 [−0.92, +1.16] |

### 3.3 判读

- **N1 判别（主指标 N1 AP）**：六个配对区间全部跨 0。
- **原始 N1 F1 的三个「显著」差都来自门槛**：分别是 5 人档同窗−无 EEG、5 人档错时−无 EEG、14 人档错时−无 EEG。用训练侧选的 α 校正门槛后，调整后 N1 F1 的六个区间全部跨 0，与 N1 AP 一致。这与第四周 C 四格复评中「LFT 的原始 N1 F1 优势来自判定门槛」是同一现象。三组的预测分数排序能力几乎相同，只是 argmax 处的 N1 门槛不同：无 EEG 组原始预测的 N1 比较多，所以原始 N1 F1 在 5 人档较高。
- **Macro-F1**：原始预测下只有 5 人档同窗−无 EEG 的区间不含 0（−0.63），三个 seed 方向一致（−0.57/−0.53/−0.81）。调整后有三个区间不含 0，方向都是有 EEG 条件的组更低。但调整后 Macro-F1 的逐 seed 差摆动很大，比如 5 人档同窗−无 EEG 为 +1.78/−0.59/−2.28，主要反映 α 选点在 OOF 上的不稳定，不宜单独解读。
- **同窗−错时**（是否依赖时间对齐）：所有指标、原始与调整后的区间全部跨 0。
- **局限**：只有一个 backbone seed，三个 head seed 共用同一初始化。开发集 8 人反复使用，区间只反映受试者抽样。

结论与 C 的「当前未确认稳定跨模态收益」一致。B 的补充是：原始 N1 F1 上看到的差别经过门槛校正后消失，不能作为跨模态影响 N1 的证据。若要追加预算，建议按 C 的建议先跑一个独立 backbone seed 的 30k 三组，并以 N1 AP 和调整后 N1 F1 为准。

## 4. 无 EEG 组：Mac 初步结果与正式结果（`40_b_w4_prelim_vs_formal.py`）

11-18 的初步结果用 B 在 Mac（MPS）上提的特征。正式结果用 D 在 N 卡上提的特征。两边是同一 checkpoint、同一配方、同一探针、head seed 0。见 [xmodal_reeval/prelim_vs_formal.csv](xmodal_reeval/prelim_vs_formal.csv)。

| 预算 | E* | 开发集 y_pred 不同的段 | ΔMacro-F1 | ΔN1 AP | p_N1 最大差 | α 初步 → 正式 | 调整后 N1 F1 初步 → 正式 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 5 人 | 44 → 44 | 0 / 21,186 | 0 | −1e-6 | 2.3e-3 | 10.37 → 10.37 | 0.3071 → 0.3071 |
| 14 人 | 44 → 44 | 1 / 21,186 | −3e-5 | −2e-6 | 1.2e-3 | 4.70 → 4.16 | 0.3759 → 0.3777 |

MPS 与 CUDA 提取的特征末位不同，开发集原始预测只差 1 段。14 人档的 α 换了一个网格点，原因是 OOF 概率的微小差异让 OOF N1 F1 的最高点从 4.70 移到 4.16。因此调整后 N1 F1 差 0.002。初步报告里的量级判断不受影响，正式数字以本报告为准。

## 5. 预训练 val MSE：带 EEG 条件时（`36_b_w4_check_d_pretrain.py mse`）

协议要求同窗、错时两组另报给 EEG 条件时的 val MSE，C 的报告只给了空条件。B 用 36 号脚本独立重建 val 集，用 30k EMA 权重分别算空条件和给 EEG 条件（`eeg_cond=1`，只有 EOG 目标窗带条件）两种情况，结果在 [xmodal_val_recheck/](xmodal_val_recheck/)。

- **val 集**：窗、t、窗清单、held-out 名单的 SHA 与 D 登记的一致。噪声 SHA 不同，是已知的 ARM/x86 CPU 随机数平台差异（见 [b_d_pretrain_recheck.md](b_d_pretrain_recheck.md)）。
- **checkpoint**：SHA 与各自的 checkpoints.csv 一致，错时组 `cf17e3bd…`，同窗组 `ba81448a…`。
- **错时配对表**：B 重建的配对表 SHA `2f446ba4…` 等于错时 run_config 中登记的值。

| 组 | 条件 | EEG t∈[5,25) | EOG t∈[5,25) | stop_metric（B） | 交付方 val_mse.csv | 最大差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 错时 | 空 | 0.052464 | 0.055613 | 0.054038 | 0.054038 | 5.0e-6 |
| 错时 | 给 EEG | 0.052464 | 0.055558 | 0.054011 | 0.054011 | 5.0e-6 |
| 同窗 | 空 | 0.052496 | 0.055632 | 0.054064 | 0.054065 | 7.0e-6 |
| 同窗 | 给 EEG | 0.052496 | 0.055262 | 0.053879 | 0.053881 | 7.4e-6 |
| 无 EEG（参照，C 的统一噪声表） | 空 | 0.052310 | 0.055394 | 0.053852 | — | — |

- 所有格子与 A、D 交付的 val_mse.csv 相差 ≤ 7.4e-6，属于平台差异的量级（D 主实验复算时最大差为 3.9e-5）。空条件的数值也与 C 的统一噪声表一致。A 的 val_mse.csv 已经是团队统一噪声口径。
- 给 EEG 条件后，EOG 去噪误差的降幅：同窗 −0.00037（−0.7%），错时 −0.00006（−0.1%）。EEG 分支不接收条件，所以 EEG 的数值不变。同窗条件确实给 EOG 去噪提供了一点信息，错时条件几乎没有，方向符合预期，但幅度很小。
- 空条件下，两个有条件训练的模型的 EOG 误差都略高于无 EEG 对照，分别为 +0.00024 和 +0.00022。这是去噪目标上的差别，下游以第 3 节为准。单个 backbone seed，不做显著性判断。

## 复现

```bash
# 1. C 的 6 个 run（C 分支导出到 <C>）
python scripts/39_b_w4_xmodal_reeval.py --root <C> \
  --registry experiment3/week4/crossmodal_final_v01/c_crossmodal_registry.csv \
  --artifact-sha experiment3/week4/crossmodal_final_v01/artifact_sha256.json \
  --ref-paired experiment3/week4/crossmodal_final_v01/comparison/paired.csv --out reports/week4/xmodal_reeval/c
# 2. D 的 18 个 run（登记表路径改指 Release 解压目录）
python scripts/39_b_w4_xmodal_reeval.py --root / --registry <d_registry_local.csv> \
  --ref-adjust <C>/.../external/d_three_head_seed/d_n1_adjust.csv --ref-paired <C>/.../d_paired.csv \
  --arm-map a_same030k_eog=sync b_shift030k_eog=shift d_pre030k_eog=none --out reports/week4/xmodal_reeval/d
# 3. B 的正式汇总
python scripts/37_b_w4_d_features_to_npz.py --src <D 特征目录> --out <B-local>/w4_formal_feat/<arm> --stats std
python scripts/34_b_w4_probe.py --group B --arm <arm> --budget <label10|label25> --seed <0|1|2> \
  --eog-pooled <B-local>/w4_formal_feat/<arm>/eog_std.npz --index <B-local>/w4_formal_feat/<arm>/sample_index.csv
python scripts/35_b_w4_crossmodal_eval.py --registry reports/week4/xmodal_reeval/b_formal_registry.csv --out reports/week4/b_crossmodal
python scripts/39_b_w4_xmodal_reeval.py --registry reports/week4/xmodal_reeval/b_formal_registry.csv \
  --ref-paired <D 的 d_paired.csv> --arm-map ... --out reports/week4/xmodal_reeval/b_formal
python scripts/40_b_w4_prelim_vs_formal.py
```

环境：macOS arm64、CPU、torch 2.14.1。测试集未读取。
