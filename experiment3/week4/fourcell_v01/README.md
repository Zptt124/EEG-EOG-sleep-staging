# W4 实验 C · 五人预算四格结果

2026-11-18 完成。三个 head seed，配对的同一个 backbone seed，权重取 D 第三周发布的 step 0 / 30k EMA。

EEG+EOG、12 层 hidden 64、t=1、非重叠 1 秒 std。5 名有标签受试者共 11,936 段，训练侧按人五折、50 轮 OOF Macro-F1 选轮，再用全部 5 人重训。配置及 α 在最终评分前锁定。开发集为 8 人、21,186 段，封存测试集未使用。

LFT 沿用 W3 C 的时间共享标准化、AdamW lr=5e-4、batch 32；线性头沿用 W3 B 的池化标准化、AdamW lr=1e-3、batch 128。两种读出各自的配方固定用于随机和预训练权重。

| 读出 | 权重 | Macro-F1 | N1 F1 | N1 AP | α 调整后 Macro-F1 | α 调整后 N1 F1 |
| --- | --- | --- | --- | --- | --- | --- |
| lft | pretrained | 70.83% | 33.05% | 29.14% | 65.65% | 35.64% |
| lft | random_init | 67.60% | 29.39% | 28.82% | 64.98% | 35.23% |
| linear | pretrained | 70.18% | 25.38% | 32.10% | 68.47% | 38.85% |
| linear | random_init | 66.98% | 16.18% | 27.01% | 64.87% | 34.84% |

表中为逐 seed 分数的均值。以下差值先合并三个 seed 的预测，再按同一批开发受试者配对 bootstrap，2000 次，95% 区间，单位为百分点。

| 比较 | ΔMacro-F1 [95% CI] | ΔN1 F1 [95% CI] |
| --- | --- | --- |
| linear_pretrained_minus_random | +3.19 [+2.28, +4.26] | +9.20 [+6.11, +12.49] |
| lft_pretrained_minus_random | +3.29 [+2.07, +4.33] | +3.50 [-1.37, +7.22] |
| pretrained_lft_minus_linear | +0.69 [-1.10, +2.02] | +7.47 [-0.88, +12.96] |

预训练带来的整体 Macro-F1 收益在两个读出中均复现。LFT 相对线性头的预训练额外收益（两者预训练−随机差的差）为 +0.10 [-1.33, +1.37] 个百分点，区间跨零。

预训练 LFT 的原始 N1 F1 均值更高，线性头的 N1 AP 均值更高。训练侧 α 约为 8–31，应用到开发集后部分 LFT 的 Macro-F1 下降明显，应同时报告原始和调整后分数。现有证据支持预训练收益，也反映了 N1 排序与判定门槛的差异。LFT 的整体优势和读出对预训练收益的交互仍不明确。

三个 head seed 共用一个 backbone seed 和同一批开发受试者。结果尚未覆盖独立预训练初始化、14 人预算或封存测试。

B 可用 c_eval_registry_w4.csv 独立复评。predictions 路径相对本交付目录；使用 B 的 --extra 前，将预测目录及登记路径映射到其仓库根目录。每个 run 含 11 列开发集预测、metrics.json、config.json、训练侧 OOF 和按受试者指标。LFT 另含全名单重训历史。

配置依据为评分前 A 已提供的 1 秒 std 推荐和本次三 seed 训练侧复验。本轮以 1 秒配置完成最终评分。

2026-11-18 18:03 收到 A 的最终池化对照（PR #24，commit 9e6cade）。在官方 EEGDM、14 人、单 head seed 下，1s/2s 的训练侧 OOF Macro-F1 为 72.22%/72.09%，OOF N1 F1 为 37.96%/34.01%。2s 每轮快约 19%，峰值显存从 916 MiB 降至 504 MiB。A 最终优先推荐 2s 以节省资源，同时认可 1s 为连续性配置。C 本轮锁定和评分早于这次交付，结果按已完成的 1s 配方汇报。A 的成本测量适用于其官方主干实验，C 的小主干需单独测速。

窗口池化后维数相同并不保证特征数值相同。先计算不同窗长的 std 再取均值，通常会改变线性头输入。跨模态三组应在评分前统一窗长。差异显著性应使用按受试者配对差的区间；两组各自区间重叠不能替代该检验。

A 的最终对照 https://github.com/Zptt124/EEG-EOG-sleep-staging/blob/9e6cade7ec4bbd0fc407e34c0d45d7daa407507f/reports/week4/a_w4_pooling_compare.md

[运行登记](c_eval_registry_w4.csv) · [复评和重跑](reproduction.md) · [模型与完整结果下载](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-fourcell-w4-v01-20261119)
