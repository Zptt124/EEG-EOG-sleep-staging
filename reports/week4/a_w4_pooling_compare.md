# A 第四周:1s/2s std pooling 对照 · 进行中快照(2026-11-17)

协议:14 人预算(label25,sseed2024)+ 8 名验证者,53,496 段;LFT 训练侧按人五折选轮
(与 C 第三周 inner_cv 一致,fold_seed=0,每折 50 轮,OOF Macro-F1 选 E*,全名单重训,
开发集只最终评分一次)。两窗口只差时间位置数(1s=30,2s=15);pooled 特征同为 2560 维,
线性头与窗口无关。

## 特征提取(完成)

提取入口 `scripts/36_a_w4_window_extract.py`(基于 B 26 号,仅窗口参数化)。
2 秒口径:非重叠 2s 窗(400/400,官方 reducer 只在 L=6000 native 模式合法,15 位置)。

| 配置 | 记录/段数 | 耗时 | 峰值显存 | 吞吐 |
| --- | --- | --- | --- | --- |
| EEG 1s std t=1 | 43 / 53,496 | 27.6 min | 591 MB | 31–33 段/s |
| EEG 2s std t=1 | 43 / 53,496 | 30.6 min | 591 MB | 28–33 段/s |
| EOG 1s std t=1 | 43 / 53,496 | 32.4 min | 591 MB | 28–33 段/s |
| EOG 2s std t=1 | 43 / 53,496 | 28.7 min | 591 MB | 28–33 段/s |

两窗口提取耗时基本一致(瓶颈在主干 5s 前向,窗口只改 reducer)。本机 RTX 4060 Laptop,
冒烟结果与 B 在 C 机器上的记录一致(1863 段、吞吐、峰值显存逐项相同)。

## LFT 对照(训练中)

入口 `scripts/37_a_w4_lft_compare.py`(基于 C inner_cv,仅 n_pool 参数化 + Windows 适配,
fcntl 垫片在 `experiment3/week4/pool_compare/_compat/`)。1s 与 2s 同机并行,
`experiment3/week4/pool_compare/runs/{lft_w1s,lft_w2s}_seed0_v01`。

| 项 | 1s | 2s |
| --- | --- | --- |
| LFT 每轮耗时(同机并行,首 15 轮) | ~170 s | ~138 s |
| 峰值显存 | 916 MiB | 504 MiB |
| 首轮 loss(fold1) | 0.8165 | 0.8270 |
| 5 折 + 重训 ETA | ~13 h | ~11 h |

参考:C 机器 W3 的 1s 每折 50 轮 ≈ 3.2 h,seed0 OOF Macro-F1 0.7204(E*=16)。
本机 1s fold1 首轮 loss 0.8165 vs C 的 0.8354(torch 2.13 vs 2.7.1 数值差异,口径可用)。

完成后补:inner_cv 曲线、选轮、OOF Macro-F1/N1、重训后开发集指标、逐折计时。

## 跨模态前置(完成)

- D 公共预训练脚本 `experiment4/week4/`(来自 D 的 feat/d-week4-pretrain-script)。
- 预训练缓存:`experiment4/week4/cache/pretrain_train_v01/`,SHA256 与团队基准
  **逐字节一致**(signals `baa935f7…`,index `e3632bce…`)。本机需 numpy==1.26.4
  (numpy 2.x 的 float32 ufunc 舍入不同);团队环境 = py3.11 + torch 2.7.1+cu126 +
  numpy 1.26.4 + scipy 1.15.3 + lightning 2.5.1 + diffusers 0.33.1 + ema-pytorch 0.7.7。
- B 的 held-out 名单(`manifests/pretrain_split_w4_v01.csv`,PR #21)与验证窗清单就位。
- 待 LFT 完成后启动:`01_pretrain.py --cross-modal same_window --no-early-stop --max-steps 30000`。

## 诊断图(完成)

`experiment3/week4/pool_compare/diag/`:同窗 vs 错时 Fpz-Cz/EOG 相关性与 0.3–4 Hz
magnitude-squared coherence(Welch nperseg=500,Δf=0.4 Hz),20,000 窗按受试者分层抽样。
低频 MSC:同窗 0.394 vs 错时 0.359(配对差 +0.035);高 EOG 振幅窗差距更大(+0.057)。
口径与图注见 `diag/README.md`。
