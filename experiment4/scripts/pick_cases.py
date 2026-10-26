#pick_cases.py —— 挑典型片段的sample_id，供下一步画波形
#输出：real_data/cases/typical_cases.csv（每行一段：类别、sample_id、受试者、记录、网格序号、真实标签、相关模型的预测和概率）

import sys
from pathlib import Path

import pandas as pd

#数据文件夹：本脚本上一级文件夹里的real_data（不管从哪个目录运行都能找到）
DATA = Path(__file__).resolve().parents[1] / "real_data"
OUT = DATA / "cases"
OUT.mkdir(exist_ok=True)
STAGES = ["W", "N1", "N2", "N3", "REM"]
N_PER_CLASS = 6      #每类挑几段
MAX_PER_SUBJECT = 2  #每个人最多几段，让片段分散到不同的人

FILES = {
    "CNN_EEG":             "predictions_val_EEG.csv",
    "CNN_EOG":             "predictions_val_EOG.csv",
    "CNN_avg":             "predictions_val_CNN_fusedavg.csv",
    "EEGDM_EEG":           "predictions_val_EEGDM_EEG_cond14.csv",
    "EEGDM_EOG":           "predictions_val_EEGDM_EOG_cond14.csv",
    "EEGDM_avg":           "predictions_val_EEGDM_fusedavg_cond14.csv",
    "EEGDM_EEGEOG":        "predictions_val_EEGDM_EEGEOG_cond14.csv",
    "EEGDM_EEG_25":        "predictions_val_EEGDM_EEG_cond14_label25.csv",
    "EEGDM_EEGEOG_25":     "predictions_val_EEGDM_EEGEOG_cond14_label25.csv",
}

#---------- 读入并合成一张宽表：每行一段，每个模型两列（预测、对真实阶段的概率） ----------
base = None
for name, fn in FILES.items():
    df = pd.read_csv(DATA / fn).sort_values("sample_id").reset_index(drop=True)
    if base is None:
        base = df[["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]].copy()
    elif not (df["sample_id"].equals(base["sample_id"]) and df["y_true"].equals(base["y_true"])):
        print(f"WARNING: {name} not aligned with the first table")
        sys.exit(1)
    base[f"pred_{name}"] = df["y_pred"]
    #每段在"真实阶段"上的概率：比如真实是N1，就取p_N1。用来衡量模型对正确答案有多大把握
    prob = df[[f"p_{s}" for s in STAGES]].to_numpy()
    base[f"ptrue_{name}"] = prob[range(len(df)), df["y_true"].to_numpy()].round(3)
print(f"OK: {len(FILES)} tables aligned, {len(base)} rows")

def correct(name):
    #返回一列True/False：这个模型判对没有
    return base[f"pred_{name}"] == base["y_true"]

def pick(mask, score_col, category):
    #在满足条件的段里，按score_col从高到低排，每人最多MAX_PER_SUBJECT段，取前N_PER_CLASS段
    #kind="mergesort"是稳定排序：分数相同时保持原来的顺序（按sample_id），换电脑、换pandas版本挑出来的结果都一样
    cand = base[mask].sort_values(score_col, ascending=False, kind="mergesort")
    cand = cand.groupby("subject_id").head(MAX_PER_SUBJECT).head(N_PER_CLASS).copy()
    cand.insert(0, "category", category)
    print(f"{category}: {int(mask.sum())} candidates, picked {len(cand)}")
    return cand

N1, REM = 1, 4
cases = pd.concat([
    #A：SC462的N1，CNN-EEG对、CNN平均错；按CNN-EEG对正确答案的把握排序
    pick((base["subject_id"] == "SC462") & (base["y_true"] == N1) & correct("CNN_EEG") & ~correct("CNN_avg"),
         "ptrue_CNN_EEG", "A_N1_hurt_by_avg_SC462"),
    #B：真实REM，25%只用EEG错、25%拼接对；按拼接的把握排序
    pick((base["y_true"] == REM) & ~correct("EEGDM_EEG_25") & correct("EEGDM_EEGEOG_25"),
         "ptrue_EEGDM_EEGEOG_25", "B_REM_rescued_by_EOG_25"),
    #C：真实N1，100%只用EEG错、拼接对
    pick((base["y_true"] == N1) & ~correct("EEGDM_EEG") & correct("EEGDM_EEGEOG"),
         "ptrue_EEGDM_EEGEOG", "C_N1_rescued_by_EOG"),
    #D：真实N1，6个100%模型全错；按"最有把握的错误"排序（真实阶段概率最低 = 错得最肯定）
    pick((base["y_true"] == N1) & ~correct("CNN_EEG") & ~correct("CNN_EOG") & ~correct("CNN_avg")
         & ~correct("EEGDM_EEG") & ~correct("EEGDM_EOG") & ~correct("EEGDM_EEGEOG"),
         "ptrue_EEGDM_EEGEOG", "D_N1_all_models_wrong"),
])

#为了方便阅读，把数字标签翻译成阶段名
cases["true_stage"] = cases["y_true"].map(dict(enumerate(STAGES)))
for name in FILES:
    cases[f"pred_{name}"] = cases[f"pred_{name}"].map(dict(enumerate(STAGES)))

cases.to_csv(OUT / "typical_cases.csv", index=False)
pd.set_option("display.width", 250)
print(cases[["category", "sample_id", "true_stage", "pred_CNN_EEG", "pred_CNN_avg",
             "pred_EEGDM_EEG", "pred_EEGDM_EEGEOG"]].to_string(index=False))
print(f"\nSaved: {OUT / 'typical_cases.csv'}")