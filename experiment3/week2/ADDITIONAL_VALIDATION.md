# C：EEG-only多名单与按受试者留出验证

2026-11-02更新。这两项是原W2与三个补充实验之后的新探索验证，均已完成并通过本地审计；原始报告和先前结果保留。尚未纳入B的独立全组复评，不替换原公共W2运行登记。

**完整预测和选定模型：[同仓库Release下载](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-additional-validation-v01-20261102)。** 两个ZIP分别约386MB和240MB；下载入口无需切换分支。报告、指标、配置、训练历史、隔离证据、源码和测试直接在main本目录中。

| 实验 | 设计与完成规模 | 主要发现 |
| --- | --- | --- |
| EEG-only＋3份标注名单 | 原54/8/16划分；每名单5/14人标注；S0与3个已有S1；54内层拟合、24最终模型 | 固定8人上18/18组Macro-F1正向，平均+0.136个百分点，幅度很小 |
| 按受试者留出 | 仅原54训练者内3折；每折36人适配、18人留出；各折重训3个种子，共9个新EMA；3名单、4/9人预算；162内层拟合、72最终模型 | 合并54人OOF后4/18组Macro-F1正向、14组负向，平均−0.042个百分点；未确认稳定的跨人适配收益 |

**研究结论：当前末两层、10000步本地扩散继续适配没有在本轮显示稳定跨人收益。** 4人预算S0→S1的Macro-F1均值61.008%→60.939%（−0.069pp），9人预算64.672%→64.656%（−0.016pp）；N1 F1分别−0.331pp、−0.061pp。没有一组正向配对95%区间完全高于零。S0本身已是扩散预训练主干，因此这些结果仅检验继续适配的增量价值。

## 直接阅读报告

- [EEG-only三名单完成报告](reports/EEG_multisubset_20261031_v02/EEG多名单_完成报告.md)、[24模型运行表](reports/EEG_multisubset_20261031_v02/run_registry.csv)、[18组配对差值](reports/EEG_multisubset_20261031_v02/paired_adaptation_deltas.csv)、[完成审计](reports/EEG_multisubset_20261031_v02/completion_audit.json)。
- [按人留出：发现与两轮对照总结](reports/subject_holdout_20261101_v01/实验发现与两轮对照总结.md)、[完整18组结果与区间](reports/subject_holdout_20261101_v01/按受试者留出_完成报告.md)、[54人OOF配对表](reports/subject_holdout_20261101_v01/OOF54_paired_deltas.csv)、[72模型逐折运行表](reports/subject_holdout_20261101_v01/fold_run_registry.csv)。
- [逐折配对](reports/subject_holdout_20261101_v01/fold_pairs.csv)、[逐人配对](reports/subject_holdout_20261101_v01/per_subject_pairs.csv)、[原始锁定方案](reports/subject_holdout_20261101_v01/实验方案.md)、[主干隔离审计](reports/subject_holdout_20261101_v01/adaptation_isolation_audit.json)、[缓存审计](reports/subject_holdout_20261101_v01/feature_cache_audit.json)、[完成审计](reports/subject_holdout_20261101_v01/completion_audit.json)。
- [全部原始文件与Release成员SHA](additional_source_snapshot.json)、[外部产物索引](additional_external_artifacts.json)、[本次发布复核](verification/additional_publication_20261102/)。

两轮评价者、适配训练池（54→36）和标注人数（5/14→4/9）均变化，不能把结果差异全部归因于某一种原因。4人预算相对36人为11.11%，9人为25%。名单、适配种子和折共享受试者/模型，表中描述性均值不是独立重复；按人bootstrap没有完整覆盖训练不确定性。本轮54人此前参与研究开发，S0预训练数据重叠未审计，不能称为全新外部数据验证。原8人开发验证和16人测试者均不纳入新留出实验。

## 下载内容与复核

在上方Release下载两个ZIP：

1. `C_additional_full_predictions_and_indices_v01.zip`：96份最终模型原始预测CSV、24份合并OOF54预测CSV、9份新缓存样本索引。合并OOF是同一批外层预测的重新汇总，不增加独立模型数量。
2. `C_additional_selected_models_v01.zip`：9份新主干选定EMA、312份收敛分类器NPZ（216内层＋96最终），以及单列的首版v01两份历史拟合权重。先验证`SHA256SUMS.txt`；每个成员的路径和SHA见`manifest.json`。

不用解压，以本目录为工作目录，安装NumPy和Pandas后运行：

```bash
python scripts/verify_additional_validation.py --assets /path/to/downloaded/assets
```

脚本校验原始字节、全部96份最终预测身份/概率/指标、24份OOF合并、36组配对bootstrap、312次收敛记录及9份适配日志隔离。还核对314份分类器NPZ的结构和SHA，其中2份属于失败v01诊断，单独计数。若需额外读取9份EMA元数据与有限性，安装PyTorch后加`--inspect-backbones`。复核不训练、不联网、不读取原始信号或测试数据。

9份约1.36GB的池化特征数组保留在本地，未放入Git或本次Release，其SHA与位置见外部索引。初始S0模型、原始信号和旧特征依赖仍按原交接协议恢复；两份原始冻结S0 EEG/EOG NPZ的[既有Release](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-pooled-features-v01-20261031)继续可用。

## 源码与历史记录

- EEG-only：[正式v02脚本](scripts/run_eeg_multisubset_v02.py)、[测试](tests/test_eeg_multisubset_v02.py)。
- 按人留出：[训练/提取/分类/分析流水线](scripts/run_subject_holdout.py)、[测试](tests/test_subject_holdout.py)、[日志字段口径](reports/subject_holdout_20261101_v01/日志字段说明.md)。
- 首版v01因最弱正则模型达到迭代上限而停止，开发验证评价次数为0；保留[失败记录](reports/EEG_multisubset_20261031_v01/summary.json)、[原脚本](scripts/run_eeg_multisubset.py)。v02仅提高求解器迭代/函数调用上限后全部重跑，没有放宽梯度阈值或变更目标、名单、正则网格。

复制的锁、报告、日志和源码均保留计算当时的字节。报告中的“尚未同步GitHub”属于历史状态，当前入口以本页及[publication_status.json](publication_status.json)为准。完整训练重放须恢复固定vendor、原始S0与特征/信号依赖，并派生新运行版本；不要直接覆盖归档路径。本次发布没有启动新训练，也不代表已通过B的独立复评。
