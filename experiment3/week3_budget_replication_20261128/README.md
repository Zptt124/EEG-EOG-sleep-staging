# Sleep-EDF 开发复验｜固定标注预算下的人员与夜间分配

从相同的 5 人、1,200 段起步，每臂追加 600 段。三臂分别标注原人原夜、原人另一夜和新人新夜，使用匹配夜间三分位的连续 30 分钟块、相同的冻结 EEG/EOG 特征和固定 30 轮线性分类头。30 份名单各做 5 次配对抽块，共 450 次主分析拟合，另做 90 次种子敏感性拟合。

| 比较 | Sleep-EDF 均值与 95% 区间 | Ying 均值与 95% 区间 |
|---|---:|---:|
| 新人新夜 − 原人原夜 | +1.53 [+0.46, +2.61] | +1.79 [+0.58, +3.07] |
| 新人新夜 − 原人另一夜 | +1.09 [−0.25, +2.41] | +1.32 [+0.15, +2.57] |
| 原人另一夜 − 原人原夜 | +0.44 [−0.49, +1.38] | +0.47 [−0.43, +1.47] |

差值是五类 Macro-F1 的百分点，区间来自受试者、名单与配对抽块的重抽样，条件是各自已观察到的训练池。Sleep-EDF 预设主比较为「原人另一夜 − 原人原夜」，判为不确定。加新人是在开发复验中得到的次要发现，随后成为 [Ying 外部复验](../week4_budget_external_ying/README.md) 的预设主比较。

开发评估固定为 16 人，包括已反复用于开发的 dev8 和曾参加骨干无标签预训练的另 8 人，二者单独报告。训练名单之间会共享个体。Sleep-EDF 的有效窗口沿用标签辅助裁剪。历史预留 8 人和最终封存 16 人未进入该轮评估。

## 结果与复算

- [冻结方案](results/plan.json)、[选中块](results/blocks.csv)、[候选块](results/candidate_blocks.csv)、[逐拟合评分](results/scores.csv) 和 [汇总](results/summary.json)。
- `results/confusion_counts.npz` 保存按种子、名单、抽块、方案和评估者汇总的 5×5 混淆计数。运行 `python experiment3/week4_budget_external_ying/verify_results.py` 可用归档计数复算两个队列的汇总与区间。
- 重新拟合需 Python 3.11、NumPy、PyTorch、scikit-learn，以及 [D 发布的 full 缓存](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w6-shared-cache-v01-20261126)。运行 `python experiment3/week3_budget_replication_20261128/run.py --bundle /path/to/full --output output/budget_replication_20261128`。脚本会校验特征指纹并按原始种子重建方案。
