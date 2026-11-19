# A 第四周 · 周末会一页结果（草稿,2026-11-19）

> 口径:14 人预算(label25,sseed2024)+ 8 名验证者;LFT/探针均训练侧五折选轮、开发集只最终
> 评分一次。测试集(16 人)本周未使用。本机 RTX 4060 Laptop。

## 结论一句话

**同窗 EEG 30k 的 EOG-only 表征不输官方 EEGDM 冒烟参考(label25 开发集 Macro 0.6404,
N1 AP 0.310,α 调整后 N1 F1 0.376);跨模态三组判读以 B 的 35 号脚本配对复评为准。
LFT 提速:1s→2s 池化每轮快 ~19%、显存 −45%、分数打平,推荐 2s(团队已锁 1s,作为
连续性选择可接受)。**

## 1 · LFT 提速计时表与推荐配置

入口:`scripts/36_a_w4_window_extract.py` + `scripts/37_a_w4_lft_compare.py`。

| 配置 | 位置数 | 提取耗时 | LFT 每轮 | 全套净训练 | 峰值显存 | OOF Macro-F1 | dev Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1s std | 30 | 27.6–32.4 min/组 | 156–173 s | ~11.5 h | 916 MiB | 0.7222 | 0.7283 |
| 2s std(推荐) | 15 | 28.7–30.6 min/组 | 128–137 s(−19%) | 10.1 h | 504 MiB | 0.7209 | 0.7497 |
| 3s std(提速②) | 10 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 |
| 2s 去 classifier(提速②) | 15 | 同 2s | 待补 | 待补 | 待补 | 待补 | 待补 |
| 2s signal 100Hz(提速②) | 15 | ~15 min/组(冒烟 73–76 段/s,2.2×) | 待补 | 待补 | 待补 | 待补 | 待补 |

- 推荐配置:`--window-sec 2 --reduce std --diffusion-t 1`(2s = 非重叠 400/400,15 位置,
  pooled 同为 2560 维,linear 臂与窗口无关)。1s 复现 C 第三周(OOF 0.7222 vs 0.7204)。
- 提速方法②(更长 pooling/去 classifier/downsampling)为计划中的低优先级项,11-19 起补,
  结果随表更新(commit 见分支 feat/a-week4-pooling-cv)。

## 2 · 同窗 EEG 30k 的 EOG-only 结果

权重 Release `a-w4-xmodal-same-30k-v01-20261119`(SHA256 `ba81448a…`);训练:
cond_dropout 0.2、B 的 8 人 held-out w4_v01、val 固定 2,048 窗,stop_metric 0.05407@30k
(单调下降;D 06 号脚本按团队噪声重算与本机逐位一致,已闭合)。EOG-only 特征(768 维,
EEG 条件置空)zero-eeg 逐位验收通过。

| run | E* | dev Macro-F1 | N1 AP | N1 F1 原始 | α | N1 F1 调整后 |
| --- | --- | --- | --- | --- | --- | --- |
| xm_sync label10 seed0/1/2 | 44/24/40 | 0.5967/0.6013/0.5915 | 0.2475/0.234/0.240 | 0.165/0.177/0.141 | 7.61/6.73/9.04 | 0.306/0.305/0.304 |
| xm_sync label25 seed0 | 44 | 0.6404 | 0.310 | 0.254 | 5.54 | 0.376 |

B 的官方 EEGDM EOG-only 冒烟参考(label10 seed0:Macro 0.583 / N1 AP 0.223 / N1 F1
0.193→0.308)→ **同窗 backbone 不输官方主干**。11 列预测与逐 run 小文件:
`experiment3/week4/pool_compare/probe_results/`。

## 3 · EEG/EOG 相关性与 coherence 图

`experiment3/week4/pool_compare/diag/`(corr_by_subject / coherence / coherence_amp_strata +
window_metrics.csv,供 B 复算):低频 MSC 同窗 0.394 vs 错时 0.359(配对差 +0.035),高 EOG
振幅窗差距更大(+0.057)——眼动伪迹线索存在,幅度待与 C 的判读规则一起看。

## 待办与风险

- 提速方法②三个变体跑完后补计时表(11-19 夜起 GPU/CPU 并行)。
- PR #24 仍为 DRAFT:cross-modal 实现经 D 的文件式 review(PR #23 review_A_xmodal_same_v01.md
  "训练实现通过")后,待团队确认合并路径(PR #21/#22/#23 均 OPEN,main 尚无 experiment4/week4)。
- B 的正式三组汇总(35 号脚本)截至 11-19 上午未出;D 的 PR #23 已有三组交叉核对数值
  (analysis/w4_xmodal_recheck/),A 的同窗结果已由 C/D 独立复评(PR #25)。
