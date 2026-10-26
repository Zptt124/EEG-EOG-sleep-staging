#plot_cases.py —— 把典型片段的EEG和EOG原始波形画出来，每个类别一张图
#在Ubuntu的eegdm环境里运行（需要读A的npz数据）
#用法：python plot_cases.py 典型片段表.csv npz数据文件夹 输出图片文件夹

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  #不弹出窗口，直接存成图片文件（Ubuntu里没有图形界面）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cases_csv, npz_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

EEG_COLOR = "#2a78d6"  #蓝
EOG_COLOR = "#eb6834"  #橙
INK = "#333333"        #文字颜色
GRID = "#dddddd"       #网格颜色（淡，不抢眼）
FS = 100               #采样率100Hz，3000个点 = 30秒

cases = pd.read_csv(cases_csv)
print(f"Loaded {len(cases)} cases")

#---------- 读出每段的原始信号 ----------
#同一晚的npz只读一次，存在字典里
npz_cache = {}
signals = []
for _, row in cases.iterrows():
    rec = row["record_id"]
    if rec not in npz_cache:
        npz_cache[rec] = np.load(npz_dir / f"{rec}.npz")
    d = npz_cache[rec]
    #npz里的epoch_index数组记录了每一行对应整晚的第几个30秒网格，用它找到这一段在第几行
    rows = np.where(d["epoch_index"] == row["epoch_index"])[0]
    if len(rows) != 1:
        print(f"WARNING: {row['sample_id']} found {len(rows)} times in {rec}.npz")
        sys.exit(1)
    i = rows[0]
    #核对真实标签和预测表里的一致
    if int(d["labels"][i]) != int(row["y_true"]):
        print(f"WARNING: label mismatch for {row['sample_id']}")
        sys.exit(1)
    signals.append((d["signal"][i, 0], d["signal"][i, 1]))  #第0通道EEG，第1通道EOG，单位µV
print("OK: all cases found in npz files, labels match")

t = np.arange(3000) / FS  #横轴：0到30秒

#---------- 每个类别一张图 ----------
for category, group in cases.groupby("category", sort=False):
    idx = list(group.index)
    n = len(idx)
    fig, axes = plt.subplots(2 * n, 1, figsize=(12, 2.3 * n), sharex=True)
    axes = np.atleast_1d(axes)

    #同一张图里，EEG用同一个纵轴范围，EOG用同一个纵轴范围，这样不同片段之间能直接比较幅值
    #用99%分位数定范围，避免被个别尖峰撑得太大
    eeg_lim = max(np.percentile(np.abs(signals[i][0]), 99) for i in idx) * 1.2
    eog_lim = max(np.percentile(np.abs(signals[i][1]), 99) for i in idx) * 1.2

    for k, i in enumerate(idx):
        row = cases.loc[i]
        eeg, eog = signals[i]
        ax_eeg, ax_eog = axes[2 * k], axes[2 * k + 1]

        ax_eeg.plot(t, eeg, color=EEG_COLOR, linewidth=0.7)
        ax_eog.plot(t, eog, color=EOG_COLOR, linewidth=0.7)
        ax_eeg.set_ylim(-eeg_lim, eeg_lim)
        ax_eog.set_ylim(-eog_lim, eog_lim)
        ax_eeg.set_ylabel("EEG (µV)", color=INK, fontsize=8)
        ax_eog.set_ylabel("EOG (µV)", color=INK, fontsize=8)

        title = (f"{row['sample_id']}   true={row['true_stage']}   "
                 f"CNN_EEG={row['pred_CNN_EEG']}  CNN_EOG={row['pred_CNN_EOG']}  CNN_avg={row['pred_CNN_avg']}   "
                 f"EEGDM_EEG={row['pred_EEGDM_EEG']}  EEGDM_concat={row['pred_EEGDM_EEGEOG']}   "
                 f"[25%] EEG={row['pred_EEGDM_EEG_25']}  concat={row['pred_EEGDM_EEGEOG_25']}")
        ax_eeg.set_title(title, fontsize=8, loc="left", color=INK)

        for ax in (ax_eeg, ax_eog):
            ax.grid(True, color=GRID, linewidth=0.5)
            ax.tick_params(labelsize=7, colors=INK)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    axes[-1].set_xlabel("Time (s)", color=INK)
    axes[-1].set_xlim(0, 30)
    fig.suptitle(category, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = out_dir / f"{category}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")