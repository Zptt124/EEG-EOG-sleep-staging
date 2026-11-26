# 第二周预算线｜C 交接

- 新名单按目录 `experiment3/week2_budget_20261127/rosters/名单号/` 查看。`matched_blocks.csv` 记录逐人逐块旧夜和另一夜对照，`training_epochs.csv` 给出按缓存行排序的 1,800 个冻结样本键。
- A、D 先查看 01 和 04，B 查看 02。03 和 10 的配置及日志在 [结果目录](results/README.md)，模型、标准化及开发预测可从结果包下载。所有九份名单可复用同一个训练入口。
- 原人另一夜拟合用 `train_other_night.py`，复用本包内的第一周 trainer。旧头在单独的 `C_week1_heads_for_B_20261127.zip`，请按名单和条件与各自标准化文件配对。
- `results/结果报告.md` 和 `results/audit/` 收录复核数据。预留 8 人评分由 B 统一组织。

[结果报告](results/README.md) · [下载完整交付](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-budget-week2-results-v01-20261127)

取得 [D 的全量共享缓存](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w6-shared-cache-v01-20261126) 后从解压根目录运行。用 Python 3.11+、NumPy、PyTorch、scikit-learn。

```text
python experiment3/week2_budget_20261127/train_other_night.py --plan experiment3/week2_budget_20261127/rosters --old-plan experiment3/week1_budget_20261126/rosters --data data --bundle /path/to/D/full --output runs --feature-info-sha256 392122c602030d8dcfa903a8d5a5f55fd582f0e09dc6d086248c86bac4267bdf --roster 03 10
```
