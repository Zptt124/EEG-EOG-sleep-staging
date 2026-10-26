#confusion.py —— 对所有模型算混淆矩阵和各阶段指标，并汇总"真实N1被判成了什么"
#输出：
#   real_data/models_summary.csv          每个模型一行：Macro-F1、准确率、各阶段F1、N1精确率/召回率
#   real_data/n1_confusion_all_models.csv 每个模型一行：真实N1被判成五个阶段各多少段、占比
#   real_data/confusion/confusion_模型名.csv  每个模型完整的5x5混淆矩阵

import sys
from pathlib import Path

import pandas as pd
#sklearn里现成的函数：confusion_matrix算混淆矩阵，precision_recall_fscore_support算每一类的精确率/召回率/F1
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

#五个阶段，顺序和全组约定一致：0=W,1=N1,2=N2,3=N3,4=REM
STAGES = ["W", "N1", "N2", "N3", "REM"]
LABELS = [0, 1, 2, 3, 4]

#数据文件夹：本脚本上一级文件夹里的real_data（不管从哪个目录运行都能找到）
DATA = Path(__file__).resolve().parents[1] / "real_data"

#---------- 模型清单：名字 -> (文件名, 预测在哪一列) ----------
#以后新增模型（比如B的25%标签结果），只要在这里加一行
MODELS = {
    #---------- 100% 标签 ----------
    "CNN_EEG":                      ("predictions_val_EEG.csv",                          "y_pred"),
    "CNN_EOG":                      ("predictions_val_EOG.csv",                          "y_pred"),
    "CNN_fused_avg":                ("predictions_val_CNN_fusedavg.csv",                 "y_pred"),
    "EEGDM_EEG_cond14":             ("predictions_val_EEGDM_EEG_cond14.csv",             "y_pred"),
    "EEGDM_EOG_cond14":             ("predictions_val_EEGDM_EOG_cond14.csv",             "y_pred"),
    # "EEGDM_EOG_cond10":           ("predictions_val_EEGDM_EOG_cond10.csv",             "y_pred"),
    "EEGDM_fused_avg_cond14":       ("predictions_val_EEGDM_fusedavg_cond14.csv",        "y_pred"),
    "EEGDM_EEGEOG_cond14":          ("predictions_val_EEGDM_EEGEOG_cond14.csv",          "y_pred"),
    #---------- 25% 标签（主名单 sseed2024，训练种子0） ----------
    "CNN_EEG_label25":              ("predictions_val_CNN_EEG_label25.csv",              "y_pred"),
    "CNN_EOG_label25":              ("predictions_val_CNN_EOG_label25.csv",              "y_pred"),
    "CNN_fused_avg_label25":        ("predictions_val_CNN_fusedavg_label25.csv",         "y_pred"),
    "EEGDM_EEG_cond14_label25":     ("predictions_val_EEGDM_EEG_cond14_label25.csv",     "y_pred"),
    "EEGDM_EOG_cond14_label25":     ("predictions_val_EEGDM_EOG_cond14_label25.csv",     "y_pred"),
    "EEGDM_fused_avg_cond14_label25": ("predictions_val_EEGDM_fusedavg_cond14_label25.csv", "y_pred"),
    "EEGDM_EEGEOG_cond14_label25":  ("predictions_val_EEGDM_EEGEOG_cond14_label25.csv",  "y_pred"),
}

#---------- 第一部分：读入参照表（A的EEG预测表），作为对齐和核对的标准 ----------
#所有模型的sample_id集合和真实标签都要和它一致，否则说明拿错了文件
ref = pd.read_csv(DATA / "predictions_val_EEG.csv")
#set_index("sample_id")：把sample_id这一列设为"行名"，之后可以按sample_id直接查某一行
ref_true = ref.set_index("sample_id")["y_true"]
print(f"Reference rows: {len(ref_true)}")

#建一个文件夹存每个模型的混淆矩阵；exist_ok=True表示已经存在也不报错
(DATA / "confusion").mkdir(exist_ok=True)

summary_rows = []   #收集每个模型的总体指标
n1_rows = []        #收集每个模型"真实N1那一行"

for name, (filename, pred_col) in MODELS.items():
    path = DATA / filename
    #文件不存在就直接停，提示是哪个文件
    if not path.exists():
        print(f"WARNING: file not found for {name}: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    #检查需要的列在不在
    for col in ["sample_id", "y_true", pred_col]:
        if col not in df.columns:
            print(f"WARNING: column '{col}' not found in {filename}")
            sys.exit(1)

    #---------- 核对1：sample_id集合必须和参照表完全一致 ----------
    if set(df["sample_id"]) != set(ref_true.index) or df["sample_id"].duplicated().any():
        print(f"WARNING: {name} sample_id set differs from reference or has duplicates")
        sys.exit(1)

    #---------- 核对2：真实标签必须和参照表完全一致 ----------
    #reindex(...)：按参照表的sample_id顺序重新排列这张表，保证一行对一行
    df = df.set_index("sample_id").reindex(ref_true.index)
    n_mismatch = int((df["y_true"] != ref_true).sum())
    if n_mismatch > 0:
        print(f"WARNING: {name} has {n_mismatch} rows with different y_true from reference")
        sys.exit(1)

    y_true = ref_true.values
    y_pred = df[pred_col].values

    #---------- 计算混淆矩阵和指标 ----------
    #cm[i][j] = 真实是第i类、被判成第j类的段数
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in STAGES], columns=[f"pred_{s}" for s in STAGES])
    cm_df.to_csv(DATA / "confusion" / f"confusion_{name}.csv")

    #p、r、f都是长度为5的数组，依次对应W、N1、N2、N3、REM
    #zero_division=0：万一某类一次都没被预测到，精确率记为0，不报错
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)

    summary = {
        "model": name,
        "macro_F1": round(f.mean(), 4),               #Macro-F1 = 五个阶段F1的简单平均
        "accuracy": round((y_true == y_pred).mean(), 4),
    }
    for j, s in enumerate(STAGES):
        summary[f"F1_{s}"] = round(f[j], 4)
    summary["N1_precision"] = round(p[1], 4)          #p[1]是N1那一类
    summary["N1_recall"] = round(r[1], 4)
    summary_rows.append(summary)

    #cm[1]是第1行，也就是"真实N1"这一行
    n1_row = cm[1]
    n_true_n1 = n1_row.sum()
    record = {"model": name, "n_true_N1": int(n_true_n1)}
    for j, s in enumerate(STAGES):
        record[f"N1_to_{s}"] = int(n1_row[j])
    for j, s in enumerate(STAGES):
        record[f"N1_to_{s}_pct"] = round(n1_row[j] / n_true_n1 * 100, 2)
    #顺手记一下"别的阶段被误判成N1"的数量（第1列减去对角线），用来看N1的误报从哪来
    record["false_N1_from_W"] = int(cm[0][1])
    record["false_N1_from_N2"] = int(cm[2][1])
    record["false_N1_from_REM"] = int(cm[4][1])
    n1_rows.append(record)

    print(f"OK: {name}  macro_F1={summary['macro_F1']}  N1_F1={summary['F1_N1']}")

#---------- 第二部分：保存汇总表 ----------
summary_df = pd.DataFrame(summary_rows)
n1_df = pd.DataFrame(n1_rows)

summary_df.to_csv(DATA / "models_summary.csv", index=False)
n1_df.to_csv(DATA / "n1_confusion_all_models.csv", index=False)

#to_string(index=False)：把整张表打印成文本，不打印行号
print("\n=== Overall and per-stage F1 ===")
print(summary_df.to_string(index=False))

print("\n=== Where true N1 goes (counts) ===")
print(n1_df[["model", "n_true_N1"] + [f"N1_to_{s}" for s in STAGES]].to_string(index=False))

print("\n=== Where true N1 goes (percent) ===")
print(n1_df[["model"] + [f"N1_to_{s}_pct" for s in STAGES]].to_string(index=False))

print("\n=== Where false N1 comes from ===")
print(n1_df[["model", "false_N1_from_W", "false_N1_from_N2", "false_N1_from_REM"]].to_string(index=False))

print("\nSaved: real_data/models_summary.csv")
print("Saved: real_data/n1_confusion_all_models.csv")
print("Saved: real_data/confusion/confusion_<model>.csv")