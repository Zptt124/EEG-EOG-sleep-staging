# B 第 7 天复盘(第一周)

负责人:B · 2026-10-22 · 分支 `feat/b-evaluation`

## 完成情况
| 项 | 状态 | 位置 |
|---|---|---|
| evaluate.py + 假数据检查 | ✅ 真实实现;假数据 8 项检查通过 | `src/evaluation/evaluate.py`、`tests/test_evaluate.py` |
| 对 A 的 E1 重算 | ✅ 与 A 临时版完全一致 | `reports/b_recompute_check.md` |
| 25% 名单、抽样种子、类别数 | ✅ 种子 2024;14 人/27 晚/32,310 段 | `manifests/label_subsets_v1.csv`、`reports/label_budget_v1.md` |
| 未标注训练对象 | ✅ 40 人,CNN 基线不使用 | `reports/label_budget_v1.md` |
| 25% EEG/EOG CNN | ✅ 真实实验(验证集) | 见下表 |
| 汇总与不确定性 | ✅ | `reports/b_summary.md`、`b_summary.csv` |

补充:另跑抽样种子 2025/2026 的 25% 重复(名单 `label_subsets_v1-s2025/s2026.csv`),仅用于估计名单抽样波动,**主结果是种子 2024**。

## run_id 与预测表(逐段预测 21,186 行,格式同模板;完整 run 在本地 `runs/`,不入库)
| run_id | val Macro-F1 | best epoch / 跑了 | predictions_val.csv sha256 |
|---|---|---|---|
| E2_EEG_label25_sseed2024_seed0_v01 | 0.7171 | 13 / 23 | `933ce4ea2144a06f8c9479a3dc4cb07a59a4315369f8091b7eb547c6246c60aa` |
| E2_EEG_label25_sseed2025_seed0_v01 | 0.7158 | 4 / 14 | `8907ed7a92fba2675041fbed1bfa21e8ccb3da2dab72240fb33cbc2706fe22b2` |
| E2_EEG_label25_sseed2026_seed0_v01 | 0.6915 | 24 / 34 | `b239d55decbe517e9d6f2aa04c127abc722a069fd98f98878962c120c577e368` |
| E2_EOG_label25_sseed2024_seed0_v01 | 0.6559 | 6 / 16 | `343a3530929405e5083ed5d6ad0fe0b3efd4e0e5f4e220dcb097d01297d4b59e` |
| E2_EOG_label25_sseed2025_seed0_v01 | 0.6534 | 14 / 24 | `cd5badd2116f239267275bf01cb7f1f99eb2934524b0a25e2c2e2fef420400c8` |
| E2_EOG_label25_sseed2026_seed0_v01 | 0.6567 | 5 / 15 | `6cd53d967e0759aa4d9f61b00fed7f714e0e8c0b6c00720cc66ac707e5eb0569` |

100% 对照引用 A:`E1_EEG_label100_seed0_v01`(0.7645)、`E1_EOG_label100_seed0_v01`(0.6827)。
每个 run 的 config 与 metrics 小文件在 `reports/runs_E2/`。

## 初步现象(验证集 8 人,仅描述,不作因果结论)
- EEG:25% 比 100% 低约 0.05–0.07 Macro-F1(三个名单种子都低,配对区间不含 0);N1 F1 也下降(0.456 → 0.32–0.38)。
- EOG:25% 比 100% 低约 0.03,但 EOG 的区间很宽,种子 2024 的配对区间包含 0;N1 F1 在 25% 下反而略高(+0.03~+0.05),区间含 0,**更像噪声**,不要解读为"少标注有利于 N1"。
- 换名单:EEG 三个种子 Macro-F1 在 0.69–0.72,EOG 在 0.653–0.657;差异小于 25% 与 100% 的差距,但 n=3、训练种子未重复。
- 逐人看,EEG 8 人全部下降;EOG 有 3 人上升、5 人下降,SC480 降幅最大(−0.13)。
- 多个 run 的 best epoch 很靠前(最早第 4 轮),验证曲线波动大,选模对单次训练的随机性敏感。

## 障碍与限制
- 验证集仅 8 人,区间宽;配对 bootstrap 只重采样这 8 人,"Δ>0 占比 0.00"不代表对新受试者必然如此。
- 100% 基线(CUDA)与 25%(Apple MPS)设备不同;训练种子只有 0 一个。
- 25% 档的归一化统计只用有标签受试者。
- 测试集未使用。

## 下一步建议(待全组决定)
1. 在 CUDA 上重跑 25%(同一份名单),消除设备差异;每个名单多个训练种子,拆开两类随机性。
2. 增加 50% 档(名单已嵌套备好:`label50` 列)画标签量曲线。
3. 评估是否需要跨验证折/合并验证+测试再做最终比较(需全组同意,测试集目前封存)。
4. 把本周 25% 预测表交 D 做融合时,注意 25% 与 100% 的预测表 sample_id 一致(已校验)。
