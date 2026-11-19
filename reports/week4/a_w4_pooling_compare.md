# A 第四周:1s/2s std pooling 对照 · 结果(2026-11-18)

协议:14 人预算(label25,sseed2024)+ 8 名验证者,53,496 段;LFT 训练侧按人五折选轮
(与 C 第三周 inner_cv 一致,fold_seed=0,每折 50 轮,OOF Macro-F1 选 E*,全名单重训,
开发集只最终评分一次)。两窗口只差时间位置数(1s=30,2s=15);pooled 特征同为 2560 维,
线性头与窗口无关。本机 RTX 4060 Laptop;1s 与 2s 同机并行训练(公平口径)。

## 结论一句话

**训练侧 OOF 打平(差 0.0013),单 seed 开发集 2s 更高(+2.1pp Macro / +5.6pp N1,区间跨零),
成本 2s 明显占优(每轮快 ~19%、显存 −45%、full 缓存减半)→ 推荐四格筛查用 2s std;
1s 作为连续性备选完全可接受。**

## 特征提取(完成)

入口 `scripts/36_a_w4_window_extract.py`(基于 B 26 号,仅窗口参数化)。
2 秒口径:非重叠 2s 窗(window/stride=400/400,官方 reducer 只在 L=6000 native 模式合法,15 位置)。

| 配置 | 记录/段数 | 耗时 | 峰值显存 |
| --- | --- | --- | --- |
| EEG 1s std t=1 | 43 / 53,496 | 27.6 min | 591 MB |
| EEG 2s std t=1 | 43 / 53,496 | 30.6 min | 591 MB |
| EOG 1s std t=1 | 43 / 53,496 | 32.4 min | 591 MB |
| EOG 2s std t=1 | 43 / 53,496 | 28.7 min | 591 MB |

两窗口提取耗时基本一致(瓶颈在主干 5s 前向,窗口只改 reducer)。冒烟结果与 B 在
C 机器上的记录一致(1863 段、吞吐 31–33 段/s、峰值显存逐项相同)。

## LFT 对照(完成)

入口 `scripts/37_a_w4_lft_compare.py`(基于 C inner_cv,仅 n_pool 参数化 + Windows 适配)。
结果目录 `experiment3/week4/pool_compare/runs/{lft_w1s,lft_w2s}_seed0_v01`
(逐轮 history / 逐折 prediction / inner_cv 曲线 / oof_train / 开发集 11 列预测)。

### 主表(seed 0)

| 指标 | 1s | 2s | 差(2s−1s) |
| --- | --- | --- | --- |
| 选轮 E* | 16 | 15 | — |
| **OOF Macro-F1(训练侧,选轮指标)** | 0.7222 | 0.7209 | −0.0013 |
| OOF N1 F1 | 0.3796 | 0.3401 | −0.0395 |
| 开发集 Macro-F1 | 0.7283 [0.6755, 0.7643] | 0.7497 [0.7002, 0.7826] | +0.0214 |
| 开发集 N1 F1 | 0.3452 [0.2678, 0.4236] | 0.4012 [0.3153, 0.4909] | +0.0560 |
| 开发集 acc | 0.8010 | 0.8082 | +0.0072 |

开发集方括号为按受试者 bootstrap 95% 区间(2000 次,官方 evaluate.py)。两个区间的
重叠很大——单 seed 下开发集差异不显著;OOF 的 0.0013 差在 32,310 段的噪声水平内。
**结论:分数上两窗口无可靠差异。**

合规说明(共同规则第 6 条):1s 与 2s 两个候选均在 C 锁定前各做了一次开发集评分
(W3 协议"开发集只最终评分一次"的延续口径),推荐选择依据为训练侧 OOF 打平 + 计时
成本;开发集数字仅作参考并在此披露,是否构成规则 6 的例外请团队在周末会裁定
(C 的锁定依据为训练侧复验,未使用本对照的开发集数字)。

### 交叉验证

- 本机 1s 复现 C 第三周 1s:OOF 0.7222 vs 0.7204,E* 16 vs 16 一致;开发集
  0.7283 vs 0.7403(torch 2.13 vs 2.7.1 数值差异,单 seed 噪声内)。
- 1s fold1 首轮 loss 0.8165 vs C 的 0.8354——特征端到端口径一致(非逐位)。

### 成本

| 项 | 1s | 2s |
| --- | --- | --- |
| 每轮(干净折) | 156–173 s | 128–137 s(快 ~19%) |
| 全套净训练 | ~11.5 h(5 折+16 轮重训) | 10.1 h(5 折+15 轮重训) |
| 峰值显存 | 916 MiB | 504 MiB |
| full 特征缓存(单模态 float16) | 8.2 GB | 4.1 GB |

注:1s 全套挂钟 29 h——fold5 两个 epoch 跨两次笔记本休眠(QPC 计入休眠时间,
训练数值不受影响,loss 曲线连续);净计算 ~11.5 h。

## 给 C 的推荐

- **推荐 2s std**:OOF 打平、开发集单 seed 更高、成本全面占优(每轮 −19%、显存 −45%)。
  四格筛查(linear/LFT × pretrained/random-init)直接可用 `--window-sec 2` 特征;
  linear 臂与窗口无关(2560 维),只有 LFT 臂受影响。
- **前提**:与 C 的四格特征口径对齐——本对照的 2s = 非重叠窗 400/400(15 位置)。
  若 C 用了别的切法,按 C 的口径重跑对照。
- **备选**:若团队要维持 W3 连续性(下游探针/三组跨模态默认 t=1),1s 也完全可接受
  ——OOF 无差异,只是成本高 ~19%。下游三组若改 2s,按 C 的规则三组一起锁定后再评分。
- 命令:
  - 提取:`python scripts/36_a_w4_window_extract.py --modality {eeg,eog} --window-sec {1,2} --reduce std --diffusion-t 1 --save-full --cache-kernels`
  - LFT:`python scripts/37_a_w4_lft_compare.py --window-sec {1,2} --output experiment3/week4/pool_compare/runs/lft_w{1,2}s_seed0_v01`

## 跨模态(进行中)

- 预训练缓存与团队基准**逐字节一致**(signals `baa935f7…`;本机需 numpy==1.26.4,
  numpy 2.x 的 float32 ufunc 舍入不同);团队环境 = py3.11 + torch 2.7.1+cu126 +
  numpy 1.26.4 + scipy 1.15.3 + lightning 2.5.1 + diffusers 0.33.1 + ema-pytorch 0.7.7。
- D 公共脚本冒烟测试通过(same_window 0.54 s/step,val MSE 0.77→0.16);30k 全量
  预计 ~4.5 h,完成后补 EOG-only 探针(34 号)与三组复评。
- 诊断图(完成):`experiment3/week4/pool_compare/diag/`——低频 MSC 同窗 0.394 vs
  错时 0.359(配对差 +0.035),高 EOG 振幅窗差距更大(+0.057);口径见 diag/README.md。

## 跨模态 · 同窗 EEG 30k 完成(2026-11-19 01:47)

- **checkpoint**:`experiment4/week4/big/runs/W4_A_xmodal_same_30k_seed0_v01/ckpt_step30000.pt`
  EMA,SHA256 `ba81448a1ff6e08402173bd76d784fd05f6e8948161f97ae0c2024dbcdb0fd70`;
  Release `a-w4-xmodal-same-30k-v01-20261119`(zip 含运行小文件 + SHA256SUMS)。
- **val(训练内,D 正式口径 2,048 窗)**:stop_metric 0.05407@30k(EEG 0.0525 / EOG 0.0556,t∈[5,25)),
  全程单调下降,best=30k。噪声口径已闭合:D 的 06 号脚本按团队噪声(3465eadc…)重算,
  与本机 val_mse.csv 的 ema 两行逐位一致(2026-11-19 核验,0 列差异),跨组比较可直接用本表。
- **EOG-only 下游特征**:768 维(12 层×64,std t=1,空 EEG 条件),提取 43 记录/53,496 段 8.6 min;
  zero-eeg 验收通过(NPZ SHA 与 EEG 置零重跑逐位一致 `c0930d41…`)。
- **探针(B 34 号,训练侧五折 50 轮 OOF 选轮 + OOF 上选 α)**:

| run | E* | 开发集 Macro-F1 | N1 AP | N1 F1 原始 | α | N1 F1 调整后 | 调整后 Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xm_sync label10 seed0 | 44 | 0.5967 | 0.2475 | 0.1652 | 7.61 | 0.3062 | 0.5743 |
| xm_sync label10 seed1 | 24 | 0.6013 | 0.234 | 0.177 | 6.73 | 0.305 | 0.5691 |
| xm_sync label10 seed2 | 40 | 0.5915 | 0.240 | 0.141 | 9.04 | 0.304 | 0.5701 |
| xm_sync label25 seed0 | 44 | 0.6404 | 0.310 | 0.254 | 5.54 | 0.376 | 0.6331 |

对照 B 的官方 EEGDM EOG-only 冒烟参考(label10 seed0:Macro 0.583 / N1 AP 0.223 /
N1 F1 0.193→0.308):W4 同窗 backbone 的 EOG-only 表征不输官方 TUEV 主干。
三组配对(同窗−无 EEG、错时−无 EEG、同窗−错时)由 B 的 35 号脚本统一复评。
