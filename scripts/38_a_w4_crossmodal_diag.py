"""Step W4-A3: 同窗 vs 错时 EEG 条件的 Fpz-Cz/EOG 相关性与低频 coherence 诊断图。

依据 C 的跨模态三组对照规则(experiment3/week4/cross_modal_protocol.md):
- 目标窗 = 46 人预训练池的 5 秒 EOG 窗;条件窗 = 同记录、间隔 60–300 秒的 Fpz-Cz 窗,
  用 B 的 33 号脚本同一抽样(pair_windows,seed=20261118),同窗 = 同一 (epoch,k) 的 Fpz-Cz 窗。
- 每窗计算 200 Hz 重采样后(与预训练管线一致)的 Pearson 相关与 Welch magnitude-squared
  coherence;诊断频带 0.3–4 Hz,bin 中心覆盖 0.4–4.0 Hz(nperseg=500,Δf=0.4 Hz)。
- 按受试者汇总;另按目标窗 EOG RMS 分层(高振幅 = 上四分位)复算同口径曲线。
- 抽样 20,000 个目标窗,按受试者分层,seed=20261118;全部参数写入图注与 CSV。

输出(experiment3/week4/pool_compare/diag/):
  corr_by_subject.png   按受试者同窗/错时平均相关(配对点)
  coherence.png         0–4 Hz 两条 MSC 曲线(均值±95%CI)
  coherence_amp_strata.png  高 EOG 振幅窗 vs 其余窗,同口径分层
  window_metrics.csv    逐窗 r / MSC / EOG RMS(供 B 复算)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

REPO = Path(__file__).resolve().parents[1]
DATA = Path("D:/sleepedf/processed")
OUT = REPO / "experiment3/week4/pool_compare/diag"
SPLIT_CSV = REPO / "manifests/pretrain_split_w4_v01.csv"

FS = 200
NPERSEG = 500
NOVERLAP = 250
BAND_LO, BAND_HI = 0.3, 4.0
N_TARGETS = 20000
SAMPLE_SEED = 20261118

# 调色板:dataviz 参考调色板 1/2 号槽(已通过相邻对 CVD 校验)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"


def load_shift_module():
    spec = importlib.util.spec_from_file_location("b33", REPO / "scripts/33_b_w4_shift_pairs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.PROCESSED = DATA
    return m


def window_signal(record_id, epoch_index, k, channel):
    with np.load(DATA / f"{record_id}.npz", allow_pickle=False) as z:
        sig = z["signal"]                                    # (n, 2, 3000) uV @100Hz
        row = int(np.where(z["epoch_index"] == epoch_index)[0][0])
    x = sig[row, channel, k * 500:(k + 1) * 500]              # 5 s @100 Hz
    return signal.resample_poly(x, 2, 1).astype(np.float64)   # 1000 @200 Hz


def msc_band(x, y):
    f, c = signal.coherence(x, y, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                            window="hann", detrend="constant")
    band = (f > BAND_LO) & (f <= BAND_HI)
    return f, c, float(c[band].mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shift = load_shift_module()
    split = pd.read_csv(SPLIT_CSV)
    pool = split[split.pretrain_role == "pretrain_train"]     # 46 人
    records = []
    for row in pool.itertuples():
        for rec in row.record_ids.split(";"):
            records.append((rec, row.subject_id))
    rec_of = dict(records)

    # 目标窗清单(全池),按受试者分层抽样
    targets = []
    for rec, subj in records:
        with np.load(DATA / f"{rec}.npz", allow_pickle=False) as z:
            ei = z["epoch_index"].astype(np.int64)
        for e in ei:
            for k in range(6):
                targets.append((rec, subj, int(e), k))
    rng = np.random.default_rng(SAMPLE_SEED)
    df = pd.DataFrame(targets, columns=["record_id", "subject_id", "epoch_index", "window_k"])
    sampled = df.groupby("subject_id", group_keys=False, sort=False).apply(
        lambda g: g.sample(n=max(1, int(round(N_TARGETS * len(g) / len(df)))), random_state=rng)
    ).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    if len(sampled) > N_TARGETS:
        sampled = sampled.sample(n=N_TARGETS, random_state=rng).reset_index(drop=True)
    print(f"sampled {len(sampled)} target windows from {len(df)}", flush=True)

    rows = []
    for i, t in enumerate(sampled.itertuples()):
        eog = window_signal(t.record_id, t.epoch_index, t.window_k, 1)
        eeg_same = window_signal(t.record_id, t.epoch_index, t.window_k, 0)
        p = shift.pair_windows(t.record_id, [t.epoch_index], [t.window_k])
        has = int(p["n_candidates"][0]) > 0
        r_same = float(np.corrcoef(eog, eeg_same)[0, 1])
        _, _, msc_same = msc_band(eog, eeg_same)
        if has:
            ce, ck = int(p["cond_epoch_index"][0]), int(p["cond_window_k"][0])
            eeg_shift = window_signal(t.record_id, ce, ck, 0)
            r_shift = float(np.corrcoef(eog, eeg_shift)[0, 1])
            _, _, msc_shift = msc_band(eog, eeg_shift)
        else:
            r_shift = msc_shift = np.nan
        rows.append(dict(subject_id=t.subject_id, record_id=t.record_id, epoch_index=t.epoch_index,
                         window_k=t.window_k, eog_rms=float(np.sqrt((eog ** 2).mean())),
                         r_same=r_same, r_shift=r_shift, msc_same=msc_same, msc_shift=msc_shift,
                         has_candidate=has))
        if i and i % 2000 == 0:
            print(f"{i}/{len(sampled)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "window_metrics.csv", index=False)

    def msc_curves(sub):
        # 逐窗重算 band 内曲线(主循环只存了 band 均值;重复计算但内存可控)
        f = np.linspace(0, FS / 2, NPERSEG // 2 + 1)
        band = (f > BAND_LO) & (f <= BAND_HI)
        curves = {"same": [], "shift": []}
        for t in sub.itertuples():
            eog = window_signal(t.record_id, t.epoch_index, t.window_k, 1)
            eeg_same = window_signal(t.record_id, t.epoch_index, t.window_k, 0)
            _, c, _ = msc_band(eog, eeg_same)
            curves["same"].append(c[band])
            if t.has_candidate:
                p = shift.pair_windows(t.record_id, [t.epoch_index], [t.window_k])
                eeg_shift = window_signal(t.record_id, int(p["cond_epoch_index"][0]),
                                          int(p["cond_window_k"][0]), 0)
                _, c, _ = msc_band(eog, eeg_shift)
                curves["shift"].append(c[band])
        return f[band], curves

    # ---- Fig 1: 按受试者平均相关 ----
    per_subj = d.groupby("subject_id").agg(r_same=("r_same", "mean"), r_shift=("r_shift", "mean"),
                                           n=("subject_id", "size")).reset_index()
    per_subj = per_subj.sort_values("r_same").reset_index(drop=True)
    x = np.arange(len(per_subj))
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    ax.plot([x, x], [per_subj.r_same, per_subj.r_shift], color=GRID, lw=1.0, zorder=1)
    ax.scatter(x, per_subj.r_same, s=34, color=BLUE, zorder=3, label="同窗 EEG(目标窗同位置 Fpz-Cz)")
    ax.scatter(x, per_subj.r_shift, s=34, color=ORANGE, zorder=3, label="错时 EEG(同夜 60–300 s)")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(per_subj.subject_id, rotation=90, fontsize=6.5, color=MUTED)
    ax.set_ylabel("Fpz-Cz / EOG 平均相关 r(200 Hz)")
    ax.set_xlabel("预训练池受试者(按同窗均值升序)")
    ax.set_ylim(-0.35, 0.55)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("同窗 vs 错时:Fpz-Cz/EOG 相关(按受试者)", color=INK, fontsize=12, loc="left")
    note = (f"n={len(sampled)} 个 5 s 窗(按受试者分层抽样,seed={SAMPLE_SEED});46 人预训练池;"
            f"错时配对按 B 33 号脚本(seed=20261118,|Δ|∈[60,300] s);r 为 200 Hz 重采样后的 Pearson 相关。")
    fig.text(0.01, 0.015, note, fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(OUT / "corr_by_subject.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: 0–4 Hz MSC 曲线 ----
    f_band, curves = msc_curves(d)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    for key, color, label in (("same", BLUE, "同窗 EEG"), ("shift", ORANGE, "错时 EEG")):
        c = np.array(curves[key])
        m, se = c.mean(axis=0), c.std(axis=0, ddof=1) / np.sqrt(len(c))
        ax.plot(f_band, m, color=color, lw=2, label=label)
        ax.fill_between(f_band, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.16, lw=0)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("Magnitude-squared coherence")
    ax.set_xlim(0, 4.05)
    ax.set_ylim(0, None)
    ax.grid(color=GRID, lw=0.7, alpha=0.7)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("同窗 vs 错时:Fpz-Cz/EOG 低频 coherence(0.3–4 Hz 诊断频带)", color=INK,
                 fontsize=12, loc="left")
    note = (f"Welch:fs={FS} Hz,nperseg={NPERSEG}(2.5 s),Δf={FS / NPERSEG} Hz,hann,noverlap={NOVERLAP},detrend=constant;"
            f"诊断频带 bin 中心覆盖 0.4–4.0 Hz;每条曲线 n={len(d)} 窗均值 ±95%CI(窗内样本级 SE);"
            f"同/错时各 {len(curves['same'])}/{len(curves['shift'])} 个有效窗。")
    fig.text(0.01, 0.015, note, fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(OUT / "coherence.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: 高 EOG 振幅分层 ----
    q = d.eog_rms.quantile(0.75)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150, sharey=True)
    for ax, (name, sub) in zip(axes, (("高 EOG 振幅窗(上四分位)", d[d.eog_rms >= q]),
                                      ("其余窗", d[d.eog_rms < q]))):
        _, c2 = msc_curves(sub)
        for key, color, label in (("same", BLUE, "同窗 EEG"), ("shift", ORANGE, "错时 EEG")):
            c = np.array(c2[key])
            m, se = c.mean(axis=0), c.std(axis=0, ddof=1) / np.sqrt(len(c))
            ax.plot(f_band, m, color=color, lw=2, label=label)
            ax.fill_between(f_band, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.16, lw=0)
        ax.set_title(name, fontsize=10, color=INK)
        ax.set_xlabel("频率 (Hz)")
        ax.set_xlim(0, 4.05)
        ax.grid(color=GRID, lw=0.7, alpha=0.7)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("Magnitude-squared coherence")
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    note = (f"分层 = 目标窗 EOG RMS 上四分位(q={q:.2f} µV@200Hz);曲线为窗均值 ±95%CI;"
            f"其余口径与左图相同(0.3–4 Hz 诊断频带,同/错时配对)。")
    fig.text(0.01, 0.015, note, fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(OUT / "coherence_amp_strata.png", bbox_inches="tight")
    plt.close(fig)

    print("diag summary:")
    print(per_subj[["subject_id", "r_same", "r_shift"]].describe().round(3))
    print(d.groupby("has_candidate").size())


if __name__ == "__main__":
    main()
