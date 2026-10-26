#把EEG和EOG两边的概率取平均，算出融合后的预测结果

import pandas as pd
import sys
from pathlib import Path

#数据文件夹：本脚本上一级文件夹里的real_data（不管从哪个目录运行都能找到）
DATA = Path(__file__).resolve().parents[1] / "real_data"

#读入align.py导出的合并表
merged = pd.read_csv(DATA / "paired_predictions_val.csv")

print(f"Loaded rows: {len(merged)}")

stages = ["W", "N1", "N2", "N3", "REM"]

#按行取平均
for s in stages:
    col_eeg = f"p_{s}_eeg"
    col_eog = f"p_{s}_eog"
    merged[f"p_{s}_fused"] = (merged[col_eeg] + merged[col_eog]) / 2

#核对每行的5个概率加起来是否约等于1
fused_cols = [f"p_{s}_fused" for s in stages]
fused_sum = merged[fused_cols].sum(axis=1)
bad_rows = (fused_sum < 0.98) | (fused_sum > 1.02)
if bad_rows.sum() > 0:
    print(f"WARNING: fused has {bad_rows.sum()} rows where probabilities do not sum to 1")
    sys.exit(1)
else:
    print("OK: fused probabilities sum to 1 for all rows")

#取出融合后的五个概率列，找每一行里数值最大的那一列
best_col = merged[fused_cols].idxmax(axis=1)

#建一个"列名 -> 数字标签"的对照表，比如 "p_W_fused" -> 0, "p_N1_fused" -> 1
col_to_label = {f"p_{s}_fused": i for i, s in enumerate(stages)}

merged["y_pred_fused"] = best_col.map(col_to_label)

print("OK: fused prediction (y_pred_fused) computed")

#快速计算整体准确率
eeg_acc = (merged["y_pred_eeg"] == merged["y_true"]).mean()
eog_acc = (merged["y_pred_eog"] == merged["y_true"]).mean()
fused_acc = (merged["y_pred_fused"] == merged["y_true"]).mean()

print(f"EEG overall accuracy: {eeg_acc:.4f}")
print(f"EOG overall accuracy: {eog_acc:.4f}")
print(f"Fused overall accuracy: {fused_acc:.4f}")

merged.to_csv(DATA / "fused_predictions_val.csv", index=False)

print("Saved: real_data/fused_predictions_val.csv")