# W4 A · 1s/2s std pooling 对照与跨模态前置

第四周实验 A 的池化对照工作区。协议:14 人预算(label25,sseed2024)+ 8 名验证者,
53,496 段;LFT 训练侧按人五折选轮(与 C 第三周 inner_cv 一致,fold_seed=0),
全名单重训 E* 轮,开发集只最终评分一次。

## 状态(2026-11-17,进行中)

| 项 | 状态 |
|---|---|
| 1s/2s std 特征提取(EEG/EOG 四组,`--save-full`) | 完成;耗时 27.6/30.6/32.4/28.7 min/组(43 记录,53,496 段) |
| LFT 对照(1s 与 2s 同机并行,seed0,5 折 × 50 轮 + 重训) | 训练中(`runs/lft_w1s_seed0_v01`、`runs/lft_w2s_seed0_v01`) |
| 跨模态诊断图(同窗 vs 错时相关性与低频 coherence) | 完成,见 `diag/` |
| 同窗 EEG 30k 预训练 | 待 LFT 完成后启动(脚本 `experiment4/week4/src/01_pretrain.py --cross-modal same_window`) |

## 目录

- `configs/w4_lft_compare.json` — LFT 配置(与 C 第三周一致;window_sec 运行时写入)
- `diag/` — 跨模态诊断三图 + 逐窗 CSV + 口径说明
- `features/` — 四组特征的小文件索引(样本索引、配置、逐记录耗时、SHA;NPZ/npy 大文件按仓库规则外置)
- `runs/` — 两组 LFT run(逐轮 history/progress/plan/smoke;权重外置)
- `_compat/fcntl.py` — Windows 运行 C 的脚本所需的最小垫片(不改动 C 任何源文件)
- `b_checkout/` — 本机 junction,不入库

## 脚本(仓库 `scripts/`)

- `36_a_w4_window_extract.py` — 提取,`--window-sec {1,2}`;基于 B 的 26 号,仅窗口参数化
- `37_a_w4_lft_compare.py` — LFT 对照;基于 C 的 inner_cv,仅 n_pool 参数化 + Windows 适配
- `a_w4_readout.py` — 37 号用的窗口参数化 readout 变体
- `38_a_w4_crossmodal_diag.py` — 诊断图(依赖 B 的 31/33 号脚本)
