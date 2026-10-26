#验证EEG EOG两张表是否对齐，并导出合并的表

#起别名pd--专门处理表格数据的工具包
import pandas as pd
import sys
from pathlib import Path

#数据文件夹：本脚本上一级文件夹里的real_data（不管从哪个目录运行都能找到）
DATA = Path(__file__).resolve().parents[1] / "real_data"

#读入两张表
df_eeg = pd.read_csv(DATA / "predictions_val_EEG.csv") #相对路径
df_eog = pd.read_csv(DATA / "predictions_val_EOG.csv")

print(f"EEG rows: {len(df_eeg)}")
print(f"EOG rows: {len(df_eog)}")

#核对两张表的 sample_id 是否完全一致
#取列变集合（集合特点：不考虑顺序自动去重）
eeg_ids = set(df_eeg["sample_id"])
eog_ids = set(df_eog["sample_id"])

only_in_eeg = eeg_ids - eog_ids
only_in_eog = eog_ids - eeg_ids

if only_in_eeg:
    print(f"WARNING: {len(only_in_eeg)} sample_id found only in EEG table")
    sys.exit(1)
if only_in_eog:
    print(f"WARNING: {len(only_in_eog)} sample_id found only in EOG table")
    sys.exit(1)
if not only_in_eeg and not only_in_eog:
    print("OK: sample_id sets match exactly between EEG and EOG tables")

#按 si 拼接两张表
merged = pd.merge(
    df_eeg,
    df_eog,
    on="sample_id", #按这个id对齐
    suffixes=("_eeg","_eog"),  #同名列分别加什么后缀
    how="inner",
)

print(f"Merged rows: {len(merged)}")

#核对合并后的y_true是否一致
mismatched = merged[merged["y_true_eeg"] != merged["y_true_eog"]]

if len(mismatched) > 0:
    print(f"WARNING: {len(mismatched)} rows have mismatched y_true, please check the data")
    sys.exit(1)
else:
    print("OK: y_true matches exactly between EEG and EOG tables")

#核对每行的5个概率加起来是否等于1
#依次把 "_eeg" 和 "_eog" 赋值给 suffix
for suffix in ["_eeg","_eog"]:
    prob_cols = [f"p_{s}{suffix}" for s in ["W", "N1", "N2", "N3", "REM"]]
    prob_sum = merged[prob_cols].sum(axis=1) #取出这五列按行求和
    bad_rows = (prob_sum < 0.98) | (prob_sum > 1.02)
    if bad_rows.sum() > 0:
        print(f"WARNING: {suffix} has {bad_rows.sum()} rows where probabilities do not sum to 1")
        sys.exit(1)

#整理保存并输出文件
merged["y_true"] = merged["y_true_eeg"]
merged = merged.drop(columns=["y_true_eeg", "y_true_eog"]) #删掉重复的列

#subject_id、record_id、epoch_index两边内容完全一样，只保留一份，去掉重复后缀列
for col in ["subject_id", "record_id", "epoch_index"]:
    merged[col] = merged[f"{col}_eeg"]
    merged = merged.drop(columns=[f"{col}_eeg", f"{col}_eog"])

merged.to_csv(DATA / "paired_predictions_val.csv", index=False)

print("Saved: real_data/paired_predictions_val.csv")