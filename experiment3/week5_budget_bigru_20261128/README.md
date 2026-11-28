# Sleep-EDF BiGRU 复验

沿用开发复验的 30 份训练名单、标注掩码、冻结 EEG/EOG 表征及 16 位开发评估者。从 5 人的 1,200 段起步，三组各追加 600 段。BiGRU 用同一人同一夜的前一段、中心段、后一段特征预测中心标签。训练监督来自原标注掩码中的中心样本。共做 450 次主要拟合和 90 次种子敏感性拟合。

| 比较 | 五类 Macro-F1 差（百分点） | 95% 配对区间 |
| --- | ---: | ---: |
| 新人新夜 − 原人原夜（主要比较） | +0.74 | [−0.32, +1.84] |
| 新人新夜 − 原人另一夜 | +0.02 | [−1.14, +1.16] |
| 原人另一夜 − 原人原夜 | +0.72 | [−0.25, +1.77] |

主要比较在 20/30 份名单上为正，预设低重叠子集均值为 +0.51 点。推进门槛还要求均值至少 +1 点且区间下限高于 0，因此本轮判为不确定。

## 归档与复跑

- [冻结协议](results/protocol.json)、[逐拟合评分](results/scores.csv)、[结果](results/summary.json) 和用于复算的按评估者汇总混淆计数。两项复验共用的复算入口见 [预算复验说明](../week6_budget_extension_20261128/README.md)。
- 发布版训练入口从仓库的 `scripts/` 读取共用的 BiGRU 模块，并用仓库相对路径区分指纹同名项。实际运行脚本的 SHA-256 单独保存在 [原脚本指纹](results/original_runner.sha256)，实验配置与训练计算沿用原版。
- 重新训练使用 CUDA 版 PyTorch（实际为 2.7.1+cu126）、NumPy、pandas、PyYAML、scikit-learn 和 [D 的完整特征缓存](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w6-shared-cache-v01-20261126)。从仓库根目录运行以下命令。前一步可复用完整的本地开发复验工作目录。

```bash
python experiment3/week3_budget_replication_20261128/run.py --bundle /path/to/full --output output/budget_replication_20261128
python experiment3/week5_budget_bigru_20261128/run.py --bundle /path/to/full --source output/budget_replication_20261128 --output output/budget_bigru_20261128
```
