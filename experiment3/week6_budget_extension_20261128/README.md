# Sleep-EDF 两档新增标注预算复验

固定 5 人、1,200 段的训练起点及原来的 16 位开发评估者。低预算追加 600 段，高预算在同一个人、同一夜、同一夜间时段给每一个低预算块配一个额外块，共追加 1,200 段。25、27 号名单分别有候选块不足的时段，按事先定好的资格规则排除。其余 28 份名单各做 5 次配对抽样，三组共新增 420 次线性头拟合，低预算评分复用原实验预测。两档均固定 30 轮，高预算每轮更新次数随样本数增加。

| 比较 | 追加 600 段 | 追加 1,200 段 |
| --- | ---: | ---: |
| 新人新夜 − 原人原夜（主要比较） | +1.51 [+0.40, +2.65] | +1.52 [+0.34, +2.77] |
| 新人新夜 − 原人另一夜 | +1.30 [+0.07, +2.57] | +1.23 [+0.13, +2.42] |
| 原人另一夜 − 原人原夜 | +0.21 [−0.69, +1.04] | +0.29 [−0.68, +1.23] |

数值为五类 Macro-F1 百分点差和 95% 配对区间。主要比较的高预算收益减去低预算收益为 +0.02 点，区间 [−0.64, +0.66]。两个预算下的平均收益均为正，预算是否改变收益大小仍不确定。配对区间以现有训练池和这 28 份名单为条件。

## 归档与复算

- [冻结协议](results/protocol.json)、[新增块与原块](results/extended_blocks.csv)、[逐拟合评分](results/scores.csv) 和 [结果](results/summary.json)。归档混淆计数使用训练名单、抽块、标注分配和评估者四个索引。
- 从仓库根目录用 Python 3.11、NumPy、PyTorch 和 scikit-learn 运行下列命令，可直接用归档计数复算本实验和 [BiGRU 复验](../week5_budget_bigru_20261128/README.md) 的均值与配对区间。

```bash
python experiment3/week6_budget_extension_20261128/verify_followups.py
```

完整复跑需要 [D 的完整特征缓存](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w6-shared-cache-v01-20261126) 和原开发复验的工作目录。首次复跑时，可用 [开发复验入口](../week3_budget_replication_20261128/README.md) 重建含逐名单预测的工作目录，再运行下列命令。使用新输出目录可为本次运行冻结输入指纹。

```bash
python experiment3/week6_budget_extension_20261128/extend_budget.py --bundle /path/to/full --source output/budget_replication_20261128 --output output/budget_extension_20261128
```
