# 实验③第二周：无标签 EEG 睡眠域适配

**C→B 特征下载补充（2026-10-31）：** [EEG/EOG 两份完整池化 NPZ](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-pooled-features-v01-20261031)已发布到 GitHub Release；各154124×2560、float32、未标准化，键为sample_id和x，附样本索引、SHA和B接口核验。EEG为原始冻结S0，EOG为C按D同协议本地重建。

2026-10-31 GitHub交付。C的3轮独立主干适配、40个EEG诊断分类头、40个EEG+EOG主对照分类头，以及N1错误分析已经完成。**B已发布v2名单核验和80份预测复评；D共同EOG来源认可及实际接收验收仍待补。测试集未使用。**

| 标注预算 | Macro-F1：S0 → S1 | N1 F1：S0 → S1 |
| --- | ---: | ---: |
| 10%（5人） | 73.209% → 73.088% | 38.454% → 37.764% |
| 25%（14人） | 74.558% → 74.584% | 40.655% → 40.752% |

S0为5个分类种子均值；S1先按每个适配主干求5种子均值，再平均3个主干。当前配置未显示稳定的双模态收益，不能推论所有域适配无效。EOG为按D原始冻结协议在本地重建；D完整缓存未收到，未验证与其逐字节一致。

- [最终报告与图表](reports/week2_final_C_20261029/实验3_第二周最终报告.md)
- [主比较明细](reports/week2_final_C_20261029/adapt_comparison.csv)、[80个运行索引](reports/week2_final_C_20261029/run_registry.csv)
- [完整运行记录与验证预测](runs/)、[N1错误分析](reports/week2_final_C_20261029/N1错误分析与待检验假设.md)
- [A/B/D依赖与交接清单](reports/week2_final_C_20261029/A_B_D依赖与交接待办.md)
- [配置与原始锁](configs/)、[训练名单及参数清单](manifests/week2_c_adaptation_v01/)、[源码](scripts/)、[测试](tests/)
- [本次发布状态](publication_status.json)、[逐文件源码/证据SHA](source_snapshot.json)、[大文件交接索引](external_artifacts.json)

## 档案与复现

本目录保持原始相对布局及文件字节，独立存放以保留第一周已发布的路径适配源码。`reports/`、`runs/`中的配置、锁、历史状态均为计算当时的原始记录；其中“尚未上传”“训练未启动”等历史文字不表示当前状态，当前交付状态看`publication_status.json`。

80份11列验证预测、配置、历史、指标、资源及SHA均已包含，每份预测21186行。原始信号、模型权重、完整逐窗缓存及S1池化数组仍外置；两份S0 EEG/EOG池化NPZ已发布到上方Release；原artifact清单引用这些外部文件，因此不能在缺少它们时声称已复验全部模型。完整原始实验复现需要按`external_artifacts.json`接收外部文件、恢复固定vendor依赖，并准备WSL/Linux环境；归档脚本不会自动下载这些材料。

固定依赖：A `02f120ee80c1c7a7822c33746ce92017632f1faa`；B `6e7df9978c48c84bbef4d2f7e9d66e9ccd93b923`；D `f00a16ba73bdabed65c819a9bfd2d9e7fe6198ed`；EEGDM `003b5a2a016b2f02788ea5693d8b7cd00b6bd969`。准确vendor路径和SHA保留于`configs/*.lock.json`；B评估器已归档在`evaluation_reference/`，仅供本次预测复评。

仅复评上传预测，无需模型权重、原始数据或GPU：

```bash
python scripts/verify_published_predictions.py
```

需要Python 3.11及以上、NumPy和Pandas。该入口核对80份预测SHA、21186段验证索引及五类概率，调用固定B v1重算点指标/混淆矩阵，与归档结果逐项对照；不重新训练或重新选择权重。完整模型/选模审计为先前本机已完成的`completion_audit.json`，不可混同本次纯预测复评。

固定50轮/最后轮选模的探索性补充实验正在本地执行，本次合并不将其列为已完成结果。原预登记结论保持不变。
