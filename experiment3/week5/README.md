# W5 交付 · C 后层读出的性能与开销

## 本周结论

完成普通 30k／self-cond 30k × 全 12 层／第 10–12 层的 24 组 linear probe。两档使用同一标签名单和 head seed。普通 30k 的后三层读出，在 5 人和 14 人档的 Macro-F1 分别为 **0.7137** 和 **0.7389**。相对普通全层的受试者配对差为 −0.25 和 +0.54 个百分点，两档的 95% CI 都跨 0。固定读后三层时，self-cond 的两档 Macro-F1 点估计均略低，当前对照未观察到额外收益。

默认可用**普通 30k 的第 10–12 层**节省存储和分类头开销。双模态缓存从 **333.39 MB 降到 86.88 MB**，普通模型的 CPU 分类头前向从 **580 µs 降到 160 µs**。普通模型的 GPU backbone 前向（含 gate 收集）分别为 149.04 ms 和 149.26 ms，两组均计算全部 12 层。两种 backbone 各使用一个预训练 seed。下游结论来自同一批 8 名 dev 受试者，封存测试集未使用。

## 交付文件

| 内容 | 文件 |
| --- | --- |
| W4 普通与 self-cond 的四格性能、N1 指标及配对区间 | [实验报告](W5_C_30k_fourcell_v01/README.md) · [汇总](W5_C_30k_fourcell_v01/summary.csv) · [逐 run 指标](W5_C_30k_fourcell_v01/metrics_by_run.csv) |
| 普通 backbone 前向与 pooling、两种模型各自的缓存与分类头开销 | [计时报告](W5_C_readout_cost_v01/README.md) · [GPU 明细](W5_C_readout_cost_v01/gpu_batches.csv) · [写盘明细](W5_C_readout_cost_v01/cache_writes.csv) |
| 补充实验，W3 普通 30k 的全层和后三层 LFT 短程筛查 | [2 轮上限](W5_C_lft_layer_screen_v01/README.md) · [5 轮上限](W5_C_lft_layer_screen_5ep_v01/README.md) |
| 24 组 probe 和 4 组 LFT 的逐段 dev／训练侧 OOF 预测、配置与文件校验 | [GitHub Release](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-w5-readout-v01-20261124) · [压缩包 SHA256](SHA256SUMS.txt) |

逐 run 指标的 `predictions` 列对应 Release 压缩包内路径。压缩包内的 `manifest.json` 提供逐文件校验值。线性 probe 的配置文件保留权重和特征哈希，`features/` 路径相对于对应 backbone 的本地运行目录。

W4 self-cond 的 GPU 前向计时还需要对应 EMA 权重。可选的 W4 self-cond 逐秒 LFT 需要该权重或保留逐秒位置的特征。现有 Release 提供的是在 30 秒段内平均后的特征。[D 的 self-cond 特征说明](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w4-selfcond30k-feat-v01-20261124)

## 复跑入口

按原运行环境使用 Python 3.11、PyTorch、NumPy、pandas、PyYAML，以及 B／D 脚本依赖。把 `W5_C_TEAM_SOURCE` 指向下面两份固定源码组成的目录。

1. 用 Git 的 `core.autocrlf=true` 检出 D 快照 `6590ec9`。这与原运行的 Windows 工作区一致，并保留 D 目录自身的 Git 属性。快照含 B 的 29 号评估、名单及 D 的 04 号特征提取。[D 快照](https://github.com/Zptt124/EEG-EOG-sleep-staging/tree/6590ec90a9fcc4ea6e1d7118556191f618117810)
2. 将 B 快照中的 `scripts/34_b_w4_probe.py` 原样放入上述目录的同一路径。该文件 SHA256 为 `a82b1c052d79677e96d7aeac282836f44524273597e07030b65495596f5980a5`。[B 的 34 号探针脚本](https://raw.githubusercontent.com/Zptt124/EEG-EOG-sleep-staging/fb4370d4b610723653332157be5dab7047fcb710/scripts/34_b_w4_probe.py)

把 `W5_C_ASSETS` 指向原封不动的 `w4_d_main30k_feat_pool1s_v01.zip` 和 `w4_d_selfcond30k_feat_pool1s_v01.zip` 所在目录。`W5_C_OUTPUTS` 可指定运行目录，默认是 `experiment3/runs`。GPU 计时另需 `w4_d_main_ckpt_0to30k_v01.zip`，并设置 `SLEEP_DATA` 和 `EEGDM_VENDOR` 指向预处理记录与原版 EEGDM 源码。[普通分层特征](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-w4-xmodal-shifted30k-v01-20261119) · [普通 30k checkpoint](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/d-pretrain-w4-main30k-v01-20261117)

```sh
python experiment3/scripts/run_week5_c_late_layers.py pilot --backbone plain
python experiment3/scripts/run_week5_c_late_layers.py full --backbone plain
python experiment3/scripts/run_week5_c_late_layers.py pilot --backbone selfcond
python experiment3/scripts/run_week5_c_late_layers.py full --backbone selfcond
python experiment3/scripts/evaluate_week5_c_fourcell.py
python experiment3/scripts/benchmark_week5_c_readout.py
```
