#fuse_avg.py —— 通用版概率平均融合：给定任意两份预测表，按sample_id对齐，把五个概率取平均
#输出是全组统一格式的预测表（11列），可以直接放进confusion.py比较
#用法：python src\fuse_avg.py --a 预测表A.csv --b 预测表B.csv --out 输出文件.csv

import argparse
import sys

import pandas as pd

STAGES = ["W", "N1", "N2", "N3", "REM"]
PROB_COLS = [f"p_{s}" for s in STAGES]
INFO_COLS = ["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]

#argparse：让脚本能从命令行接收参数
#required=True表示这个参数必须写，不写会直接报错并提示用法
parser = argparse.ArgumentParser()
parser.add_argument("--a", required=True)
parser.add_argument("--b", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

df_a = pd.read_csv(args.a)
df_b = pd.read_csv(args.b)
print(f"A rows: {len(df_a)}  ({args.a})")
print(f"B rows: {len(df_b)}  ({args.b})")

#---------- 核对1：两份表都要有需要的列 ----------
for name, df in (("A", df_a), ("B", df_b)):
    missing = [c for c in INFO_COLS + PROB_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: table {name} is missing columns: {missing}")
        sys.exit(1)

#---------- 核对2：sample_id集合完全一致、没有重复 ----------
if set(df_a["sample_id"]) != set(df_b["sample_id"]) or df_a["sample_id"].duplicated().any() or df_b["sample_id"].duplicated().any():
    print("WARNING: sample_id sets differ or contain duplicates")
    sys.exit(1)
print("OK: sample_id sets match exactly")

#---------- 按sample_id合并 ----------
#B表只取sample_id、y_true和五个概率，避免subject_id等列重复（这次提前避开了align.py当时遇到的后缀问题）
merged = pd.merge(
    df_a[INFO_COLS + PROB_COLS],
    df_b[["sample_id", "y_true"] + PROB_COLS],
    on="sample_id",
    suffixes=("_a", "_b"),
    how="inner",
)

#---------- 核对3：真实标签一致 ----------
n_mismatch = int((merged["y_true_a"] != merged["y_true_b"]).sum())
if n_mismatch > 0:
    print(f"WARNING: {n_mismatch} rows have different y_true")
    sys.exit(1)
print("OK: y_true matches exactly")

#---------- 核对4：两边每行概率加起来约等于1 ----------
for suffix in ("_a", "_b"):
    s = merged[[c + suffix for c in PROB_COLS]].sum(axis=1)
    bad = int(((s < 0.98) | (s > 1.02)).sum())
    if bad > 0:
        print(f"WARNING: {bad} rows in table {suffix[1:]} have probabilities not summing to 1")
        sys.exit(1)
print("OK: probabilities sum to 1 in both tables")

#---------- 融合：五个概率逐个取平均 ----------
out = pd.DataFrame()
out["sample_id"] = merged["sample_id"]
out["subject_id"] = merged["subject_id"]
out["record_id"] = merged["record_id"]
out["epoch_index"] = merged["epoch_index"]
out["y_true"] = merged["y_true_a"]
for c in PROB_COLS:
    out[c] = (merged[c + "_a"] + merged[c + "_b"]) / 2

#找每行概率最大的那一列，再把列名翻译成数字标签（和fuse.py一样的做法）
col_to_label = {c: i for i, c in enumerate(PROB_COLS)}
out["y_pred"] = out[PROB_COLS].idxmax(axis=1).map(col_to_label)

#按全组统一的列顺序排列：sample_id, subject_id, record_id, epoch_index, y_true, y_pred, p_W ... p_REM
out = out[INFO_COLS + ["y_pred"] + PROB_COLS]
out.to_csv(args.out, index=False)

acc = (out["y_pred"] == out["y_true"]).mean()
print(f"Fused accuracy: {acc:.4f}")
print(f"Saved: {args.out}")