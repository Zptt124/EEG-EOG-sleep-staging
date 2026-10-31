# C：三个补充实验（main 交付入口）

2026-10-31。固定训练预算、类别加权和取层消融均已完成并通过审计，新增 **160 个不重复分类头**。它们是在看过前序开发验证结果后设计的探索实验，原预登记结果保留。测试集未使用。

| 实验 | 新分类头 | 复用对照 | 完成时间（北京时间） |
| --- | ---: | --- | --- |
| 固定50轮、评价最后一轮 | 40 | 原早停结果40个头 | 16:22 |
| 五类逆平方根权重交叉熵 | 40 | 固定50轮40个头 | 17:06 |
| EEG末5层/末2层消融 | 80 | 全部20层固定预算40个头 | 18:09 |

## 结果入口

- [三个补充实验结论汇总](reports/week2_supplement_summary_20261031/三个补充实验结论汇总.md)
- [固定50轮报告](reports/week2_fixed50_20261031_v01/固定50轮_完成报告.md)、[逐运行索引](reports/week2_fixed50_20261031_v01/run_registry.csv)
- [类别加权报告](reports/week2_weighted50_20261031_v01/类别加权_完成报告.md)、[40组损失配对](reports/week2_weighted50_20261031_v01/weighted_minus_unweighted.csv)
- [取层消融报告](reports/week2_layer_ablation_20261031_v01/取层消融_完成报告.md)、[120格含复用对照索引](reports/week2_layer_ablation_20261031_v01/all120_run_registry.csv)、[训练集逐层变化](reports/week2_layer_ablation_20261031_v01/layer_shift_train_only.csv)
- [160个新运行的完整配置/历史/预测/指标](runs/)、[补充源码与证据SHA清单](supplement_source_snapshot.json)
- [本次上传复核](verification/supplement_publication_20261031/)、[当前发布状态](publication_status.json)

原报告、配置、日志与状态文件保留计算当时的字节。其中“尚未同步GitHub”等文字属于历史记录；当前发布状态以本页和 `publication_status.json` 为准。

## 主要信息

1. 原最差格的 N1 F1 适配配对差由 −8.756 变为 +0.223 个百分点，显示对停止/选模规则敏感。但固定50轮的绝对成绩更低，整体适配收益仍接近零。
2. 40/40个加权头的N1 F1均高于对应未加权头。S1的10%/25% N1 F1均值分别28.39%→29.74%、36.21%→39.44%，并有精确率/accuracy下降的权衡。S0也改善，不是适配专属收益。
3. 前18层的训练集池化特征完全相同，变化集中末2层。末5层有一般性分类收益，但六种取层×预算组合的S1平均Macro-F1均未超过匹配S0。末2层25%预算有局部N1优势，尚无稳定整体适配收益证据。

同一8名开发验证者被重复使用，分类种子不是独立适配或受试者。补充方案的Macro-F1和N1 F1均值尚未超过原早停双模态主比较，不替换原正式结论。

## 直接复核上传结果

在本目录下，以Python 3.11及以上、NumPy、Pandas、PyYAML运行：

```bash
python scripts/verify_supplement_predictions.py
```

该入口逐份核对160份预测的SHA、21186段验证身份、五类概率、固定50轮/最后轮规则、训练名单、类别权重、取层配置及B固定v1点指标。它不训练、不选择新模型、不读取测试数据，也不要求GPU或权重。

若已接收本地分类头权重，可以追加检查：

```bash
python scripts/verify_supplement_predictions.py --model-root /path/to/external/runs
```

此模式还需要PyTorch，并核对每个 `run_id/final_head_epoch50.pt` 的SHA、配置、维度和标准化参数。分类头权重、主干权重和大特征数组未入Git，位置及SHA见 [external_artifacts.json](external_artifacts.json)。两份原始冻结S0池化NPZ仍可从[既有Release](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-pooled-features-v01-20261031)获取。

三个原始训练入口与测试均在`scripts/`、`tests/`。严格训练重放还需要按归档配置恢复固定vendor和外部特征；原运行ID和锁不可覆盖，重训应派生新版本。取层头的推理必须先执行标准化后掩码，见[预测复现注意](reports/week2_layer_ablation_20261031_v01/预测复现注意.md)。本次上传没有启动新训练。
