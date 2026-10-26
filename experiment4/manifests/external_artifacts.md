# 不入库的大文件

| 内容 | 大小 | 在哪里 | 怎么重新得到 |
|---|---|---|---|
| EOG 特征 `eog_cond14_v01`（122 个 `.npy`，float16，每段 `[20, 30, 128]`） | 约 23.67 GB | D 本地 WSL：`~/week1/C_experiment3/features/eog_cond14_v01/` | 用 C 的 `experiment3/scripts/extract_features.py` 提取 EOG（数据下标 1，条件 14）；需要 GPU（D 用的是 RTX 4060 Laptop），耗时较长 |
| EEG 特征 `eeg_cond14_v01` | 约 23.67 GB | C 的交付包 | 见 C 的 `experiment3/manifests/external_artifacts.json` |
| 线性分类头权重（`best_head.pt` 等） | 每个约 100 KB | D 本地各运行文件夹 | 按 `configs/fusion_config.yaml` 重新训练，每次几十秒 |
| 多种子的全部逐段预测（100%：20 份；25%：60 份） | 约 240 MB | D 本地 `runs/D_multiseed_*` | 重跑 `multiseed_heads.py` / `multiseed_budget.py`；汇总结果已在 `reports/runs_E4/` |
| 条件 10 / 18 的 EOG 特征 | 各约 23 GB | 已删除（三条件无差别） | 同第一行，换条件编号 |

`eog_cond14_v01/` 里的 4 个元数据文件（提取设置、逐段索引、资源记录、汇总）已入库，读特征时**必须按 `sample_id` 用 `feature_index.csv` 的 `feature_path` 和 `row_index` 对应**，不能按文件行序拼接。

需要特征的组员，联系 D 通过网盘传输。
