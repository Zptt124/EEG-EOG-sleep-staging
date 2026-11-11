# 实验 C 第三周主对照 · 训练侧五折选轮

三组 C1 于 2026 年 11 月 12 日 00:27 完成。C1 使用冻结的 EEGDM、EEG/EOG、all20、t=1 和每秒 std。C0 使用 [B 已发布的三组线性头](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/b-week3-runs-v01-20261110)。两边在固定的 14 位有标签训练者上使用相同的五折划分，逐折训练 50 轮。每折的标准化仅由该折训练者拟合。选合并折外预测的 Macro-F1 最高的一轮，并列取最早一轮。随后用全部 14 位训练者重训，在同一批 8 位开发受试者的 21,186 段上评分。两种读出沿用各自的优化器配方、batch size 和输入标准化粒度。

| Seed | C0 / C1 选中轮数 | C0 / C1 Macro-F1 | Macro-F1 差值 | C0 / C1 N1 F1 | N1 F1 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0](seed0_v01/paired_vs_b_c0.json) | 11 / 16 | 74.49% / 74.03% | −0.46 | 36.34% / 38.40% | +2.06 |
| [1](seed1_v01/paired_vs_b_c0.json) | 24 / 18 | 74.85% / 74.63% | −0.21 | 42.63% / 45.55% | +2.92 |
| [2](seed2_v01/paired_vs_b_c0.json) | 16 / 37 | 72.89% / 71.33% | −1.56 | 35.87% / 40.48% | +4.61 |
| 三组平均 | — | 74.07% / 73.33% | −0.74 | 38.28% / 41.48% | +3.20 |

差值为 C1 减 C0，单位为百分点。三组的 N1 F1 点估计均为正，95% 受试者配对 bootstrap 区间均跨零。Macro-F1 差值在三组均为负，seed 2 的区间为 [−3.26, −0.06] 个百分点。三组使用同一批开发受试者。

- [逐 seed 比较与区间](comparison_vs_b_c0.csv) · [B 基线来源及 SHA256](b_c0_reference.json)
- [运行登记](run_registry.csv) · [分折、配置与输入审计](plan.json) · [文件 SHA256](artifact_hashes.json)

从仓库根目录复核三组 C1 的选轮、折外预测和开发集指标。

```bash
python experiment3/week3/scripts/verify_inner_cv.py
```

下载 B Release 的 `w3_b_runs_v01.zip` 后，使用 `--b-archive <ZIP 路径>` 复算三组配对区间。C1 训练入口见 [run_week3_c_inner_cv.py](../scripts/run_week3_c_inner_cv.py)，分折使用的 B 代码固定在 `8d21834`。分类头权重、训练检查点和原始特征按团队约定外置。
