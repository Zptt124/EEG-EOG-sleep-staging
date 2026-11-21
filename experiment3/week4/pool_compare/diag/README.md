# W4 A · 跨模态诊断图(同窗 vs 错时 Fpz-Cz/EOG)

生成脚本:`scripts/38_a_w4_crossmodal_diag.py`(2026-11-17)。依据 C 的
`experiment3/week4/cross_modal_protocol.md` 诊断图条款。

## 口径

| 项 | 值 |
|---|---|
| 数据范围 | 46 人预训练池(`manifests/pretrain_split_w4_v01.csv`,`pretrain_role=pretrain_train`),8 名 held-out 不进入 |
| 目标窗 | 20,000 个 5 秒 EOG 窗,按受试者分层抽样,seed=20261118 |
| 条件窗 | 同窗 = 同一 (epoch,k) 的 Fpz-Cz 窗;错时 = B 33 号脚本 `pair_windows`(seed=20261118,60≤\|Δ\|≤300 s);覆盖率 20,000/20,000 = 100% |
| 预处理 | 与预训练管线一致:100→200 Hz `resample_poly(2,1)`(相关性/coherence 对幅度缩放不变,未做 mu-law) |
| 相关 | Pearson r,200 Hz 全带 |
| Coherence | Welch MSC:fs=200 Hz,nperseg=500(2.5 s),Δf=0.4 Hz,hann,noverlap=250,detrend=constant;诊断频带 bin 中心覆盖 0.4–4.0 Hz(协议 0.3–4 Hz) |
| 分层 | 目标窗 EOG RMS 上四分位(q=26.10 µV@200Hz)vs 其余 |

## 文件

- `corr_by_subject.png` — 46 受试者同窗/错时平均 r(配对点,按同窗均值升序)
- `coherence.png` — 0.3–4 Hz MSC 曲线(窗均值 ±95%CI)
- `coherence_amp_strata.png` — 高 EOG 振幅窗 / 其余窗,同口径分层
- `window_metrics.csv` — 逐窗 r / band MSC / EOG RMS / 配对状态,供 B 复算

## 观察(诊断口径,不是三组比较结论)

- 低频 MSC:同窗 0.394 vs 错时 0.359,配对差 +0.035(同窗更高);
- 高 EOG 振幅窗的同窗优势更大(+0.057 vs 其余窗 +0.028),与眼动伪迹线索一致;
- 全带 r:同窗 −0.029 vs 错时 +0.023(配对差 −0.052)——全带相关由高频噪声主导,
  低频相干才是协议主诊断;此差异仅作记录,不作解读。
