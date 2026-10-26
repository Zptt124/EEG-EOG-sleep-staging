#subject_breakdown.py —— 按受试者拆开看：每个人的官方指标，以及融合前后每个人每个阶段的"帮忙/帮倒忙"
#输出到 real_data/subject/：
#   per_subject_metrics.csv  每个模型 × 每个人：Macro-F1、N1-F1、N1段数（官方evaluate.py计算）
#   per_subject_flips.csv    每组(参照→融合) × 每个人 × 每个阶段：段数、参照准确率、融合后准确率、帮倒忙数、帮忙数

import importlib.util
import sys
from pathlib import Path

import pandas as pd

#---------- 加载B的官方evaluate.py（和official_eval_all.py一样的方式） ----------
#在仓库里parents[2]是仓库根目录。按顺序找官方evaluate.py，用找到的第一个：
#   1. 仓库根目录的src/evaluation/evaluate.py（B的分支合并之后就在这里）
#   2. real_data/_upstream/evaluate.py（prepare_inputs.py从B的分支取出来的）
#   3. 本地week1/inputs/B_feat-b-evaluation/里的那份（D自己电脑上的老位置）
BASE = Path(__file__).resolve().parents[2]
EVAL_CANDIDATES = [
    BASE / "src" / "evaluation" / "evaluate.py",
    Path(__file__).resolve().parents[1] / "real_data" / "_upstream" / "evaluate.py",
    BASE / "inputs" / "B_feat-b-evaluation" / "src" / "evaluation" / "evaluate.py",
]
EVAL_PATH = next((p for p in EVAL_CANDIDATES if p.exists()), None)
if EVAL_PATH is None:
    print("WARNING: official evaluate.py not found. Run prepare_inputs.py first. Looked in:")
    for p in EVAL_CANDIDATES:
        print(f"  {p}")
    sys.exit(1)
print(f"Using official evaluate.py: {EVAL_PATH}")
spec = importlib.util.spec_from_file_location("official_evaluate", EVAL_PATH)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

#数据文件夹：本脚本上一级文件夹里的real_data（不管从哪个目录运行都能找到）
DATA = Path(__file__).resolve().parents[1] / "real_data"
OUT = DATA / "subject"
OUT.mkdir(exist_ok=True)
STAGES = ["W", "N1", "N2", "N3", "REM"]

MODELS = {
    "CNN_EEG":                        "predictions_val_EEG.csv",
    "CNN_EOG":                        "predictions_val_EOG.csv",
    "CNN_fused_avg":                  "predictions_val_CNN_fusedavg.csv",
    "EEGDM_EEG_cond14":               "predictions_val_EEGDM_EEG_cond14.csv",
    "EEGDM_EOG_cond14":               "predictions_val_EEGDM_EOG_cond14.csv",
    "EEGDM_fused_avg_cond14":         "predictions_val_EEGDM_fusedavg_cond14.csv",
    "EEGDM_EEGEOG_cond14":            "predictions_val_EEGDM_EEGEOG_cond14.csv",
    "CNN_EEG_label25":                "predictions_val_CNN_EEG_label25.csv",
    "CNN_fused_avg_label25":          "predictions_val_CNN_fusedavg_label25.csv",
    "EEGDM_EEG_cond14_label25":       "predictions_val_EEGDM_EEG_cond14_label25.csv",
    "EEGDM_fused_avg_cond14_label25": "predictions_val_EEGDM_fusedavg_cond14_label25.csv",
    "EEGDM_EEGEOG_cond14_label25":    "predictions_val_EEGDM_EEGEOG_cond14_label25.csv",
}

#要比较的(参照, 融合后)。第二组就是之前缺的"以EOG为参照"
FLIP_PAIRS = [
    ("CNN_EEG", "CNN_fused_avg"),
    ("CNN_EOG", "CNN_fused_avg"),
    ("EEGDM_EEG_cond14", "EEGDM_fused_avg_cond14"),
    ("EEGDM_EEG_cond14", "EEGDM_EEGEOG_cond14"),
    ("CNN_EEG_label25", "CNN_fused_avg_label25"),
    ("EEGDM_EEG_cond14_label25", "EEGDM_fused_avg_cond14_label25"),
    ("EEGDM_EEG_cond14_label25", "EEGDM_EEGEOG_cond14_label25"),
]

#---------- 读入所有预测表（官方函数会顺便校验格式），按sample_id排好顺序 ----------
tables = {}
for name, fn in MODELS.items():
    df = ev.load_predictions(str(DATA / fn))
    tables[name] = df.sort_values("sample_id").reset_index(drop=True)

#核对：所有表的sample_id和y_true必须完全一致，这样后面才能按行直接比较
ref = tables["CNN_EEG"]
for name, df in tables.items():
    if not (df["sample_id"].equals(ref["sample_id"]) and df["y_true"].equals(ref["y_true"])):
        print(f"WARNING: {name} sample_id or y_true differs from CNN_EEG")
        sys.exit(1)
print(f"OK: {len(tables)} tables aligned, {len(ref)} rows each")

#---------- 第一部分：每个模型、每个人的官方指标 ----------
metric_rows = []
for name, df in tables.items():
    #官方函数per_subject_table：按subject_id分组，每组算一遍Macro-F1、N1-F1等
    t = ev.per_subject_table(df)
    t.insert(0, "model", name)
    metric_rows.append(t)
per_subject = pd.concat(metric_rows, ignore_index=True)
per_subject.to_csv(OUT / "per_subject_metrics.csv", index=False)

#转成"行=人、列=模型"的宽表，方便看谁在所有模型上都差
#pivot：把长表变宽表，index是行、columns是列、values是格子里的数
wide = per_subject.pivot(index="subject_id", columns="model", values="macro_f1").round(3)
wide = wide[list(MODELS.keys())]  #列按清单顺序排列

#---------- 第二部分：融合前后，每个人每个阶段的"帮忙/帮倒忙" ----------
flip_rows = []
for ref_name, new_name in FLIP_PAIRS:
    a, b = tables[ref_name], tables[new_name]
    ref_ok = a["y_pred"] == a["y_true"]   #参照模型判对没有
    new_ok = b["y_pred"] == b["y_true"]   #融合后判对没有
    for (subj, label), idx in a.groupby(["subject_id", "y_true"]).groups.items():
        r, n = ref_ok[idx], new_ok[idx]
        flip_rows.append({
            "reference": ref_name,
            "fused": new_name,
            "subject_id": subj,
            "stage": STAGES[label],
            "n_segments": len(idx),
            "ref_accuracy": round(r.mean(), 4),
            "fused_accuracy": round(n.mean(), 4),
            "hurt": int((r & ~n).sum()),     #参照对、融合错
            "helped": int((~r & n).sum()),   #参照错、融合对
        })
flips = pd.DataFrame(flip_rows)
flips.to_csv(OUT / "per_subject_flips.csv", index=False)

#---------- 打印重点 ----------
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n=== Per-subject Macro-F1 (rows=subjects, cols=models) ===")
print(wide.to_string())

for stage in ["N1", "REM"]:
    print(f"\n=== {stage}: accuracy before -> after fusion, per subject ===")
    s = flips[flips["stage"] == stage].copy()
    s["change"] = (s["fused_accuracy"] - s["ref_accuracy"]).round(3)
    s["pair"] = s["reference"] + " -> " + s["fused"]
    #每组一行、每个人一列，格子里是准确率变化
    print(s.pivot(index="pair", columns="subject_id", values="change").to_string())

print(f"\nSaved: {OUT / 'per_subject_metrics.csv'}")
print(f"Saved: {OUT / 'per_subject_flips.csv'}")