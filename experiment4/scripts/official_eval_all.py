#official_eval_all.py —— 用B的官方evaluate.py统一评估所有模型，并做关键的配对比较
#输出到 real_data/official/：
#   official_metrics.csv      每个模型的Macro-F1、N1-F1、准确率 及 按受试者bootstrap的95%区间
#   official_paired.csv       关键模型对的配对差（A减B）及95%区间
#   metrics_official_<模型>.json  每个模型的官方完整指标

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

#---------- 找到并加载B的官方evaluate.py ----------
#__file__是本脚本的位置：week1/exp4_eog_fusion/src/official_eval_all.py
#parents[0]=src，parents[1]=exp4_eog_fusion，parents[2]=week1
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

#importlib：按"文件路径"加载一个Python文件当作模块用（因为它不在本项目的src文件夹里，不能直接import）
spec = importlib.util.spec_from_file_location("official_evaluate", EVAL_PATH)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
print(f"Official evaluate.py loaded, version {ev.EVALUATE_VERSION}")

#数据文件夹：本脚本上一级文件夹里的real_data（不管从哪个目录运行都能找到）
DATA = Path(__file__).resolve().parents[1] / "real_data"
OUT = DATA / "official"
OUT.mkdir(exist_ok=True)
N_BOOT = 1000  #bootstrap次数，和B报告里一样

#---------- 模型清单：名字 -> 预测表文件（必须是标准11列格式） ----------
MODELS = {
    "CNN_EEG":                        "predictions_val_EEG.csv",
    "CNN_EOG":                        "predictions_val_EOG.csv",
    "CNN_fused_avg":                  "predictions_val_CNN_fusedavg.csv",
    "EEGDM_EEG_cond14":               "predictions_val_EEGDM_EEG_cond14.csv",
    "EEGDM_EOG_cond14":               "predictions_val_EEGDM_EOG_cond14.csv",
    "EEGDM_fused_avg_cond14":         "predictions_val_EEGDM_fusedavg_cond14.csv",
    "EEGDM_EEGEOG_cond14":            "predictions_val_EEGDM_EEGEOG_cond14.csv",
    "CNN_EEG_label25":                "predictions_val_CNN_EEG_label25.csv",
    "CNN_EOG_label25":                "predictions_val_CNN_EOG_label25.csv",
    "CNN_fused_avg_label25":          "predictions_val_CNN_fusedavg_label25.csv",
    "EEGDM_EEG_cond14_label25":       "predictions_val_EEGDM_EEG_cond14_label25.csv",
    "EEGDM_EOG_cond14_label25":       "predictions_val_EEGDM_EOG_cond14_label25.csv",
    "EEGDM_fused_avg_cond14_label25": "predictions_val_EEGDM_fusedavg_cond14_label25.csv",
    "EEGDM_EEGEOG_cond14_label25":    "predictions_val_EEGDM_EEGEOG_cond14_label25.csv",
}

#---------- 要做配对比较的模型对：(A, B)，算 A 减 B ----------
PAIRS = [
    ("EEGDM_EEGEOG_cond14", "EEGDM_EEG_cond14"),
    ("EEGDM_EEGEOG_cond14", "EEGDM_fused_avg_cond14"),
    ("CNN_fused_avg", "CNN_EEG"),
    ("EEGDM_EEGEOG_cond14", "CNN_EEG"),
    ("EEGDM_EEGEOG_cond14", "CNN_fused_avg"),
    ("EEGDM_EEGEOG_cond14_label25", "EEGDM_EEG_cond14_label25"),
    ("EEGDM_EEGEOG_cond14_label25", "EEGDM_fused_avg_cond14_label25"),
    ("CNN_fused_avg_label25", "CNN_EEG_label25"),
    ("EEGDM_EEGEOG_cond14_label25", "CNN_EEG_label25"),
    ("EEGDM_EEGEOG_cond14_label25", "CNN_fused_avg_label25"),
    ("EEGDM_EEGEOG_cond14_label25", "EEGDM_EEG_cond14"),
]

#confusion.py算出的总表，用来核对官方结果是否一致
summary = pd.read_csv(DATA / "models_summary.csv").set_index("model")

#---------- 第一部分：逐个模型官方评估 ----------
metric_rows = []
for name, filename in MODELS.items():
    path = DATA / filename
    if not path.exists():
        print(f"WARNING: file not found for {name}: {path}")
        sys.exit(1)
    #ev.evaluate会先用官方规则校验预测表格式，不合格直接抛出ValueError
    try:
        m = ev.evaluate(str(path), n_boot=N_BOOT, seed=0)
    except ValueError as e:
        print(f"WARNING: {name} failed official validation: {e}")
        sys.exit(1)

    #核对：官方Macro-F1和confusion.py的结果必须一致（confusion.py保留了4位小数，所以允许1e-4的误差）
    if name in summary.index and abs(summary.loc[name, "macro_F1"] - m["macro_f1"]) > 1e-4:
        print(f"WARNING: {name} official macro_F1 {m['macro_f1']:.4f} != confusion.py {summary.loc[name, 'macro_F1']}")
        sys.exit(1)

    (OUT / f"metrics_official_{name}.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

    b = m["bootstrap_subjects"]
    metric_rows.append({
        "model": name,
        "macro_F1": round(m["macro_f1"], 4),
        "macro_F1_lo": round(b["macro_f1"]["lo"], 4),
        "macro_F1_hi": round(b["macro_f1"]["hi"], 4),
        "N1_F1": round(m["n1_f1"], 4),
        "N1_F1_lo": round(b["n1_f1"]["lo"], 4),
        "N1_F1_hi": round(b["n1_f1"]["hi"], 4),
        "accuracy": round(m["accuracy"], 4),
        "REM_F1": round(m["per_class_f1"]["REM"], 4),
        "N3_F1": round(m["per_class_f1"]["N3"], 4),
    })
    print(f"OK: {name}")

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(OUT / "official_metrics.csv", index=False)
print("OK: all official macro_F1 values match confusion.py")

#---------- 第二部分：配对比较 ----------
#先把所有预测表用官方函数读进来（读的同时会做格式校验）
loaded = {name: ev.load_predictions(str(DATA / fn)) for name, fn in MODELS.items()}

pair_rows = []
for a, b in PAIRS:
    d = ev.paired_bootstrap_delta(loaded[a], loaded[b], n_boot=N_BOOT, seed=0)
    row = {"A": a, "B": b}
    for key in ("macro_f1", "n1_f1"):
        row[f"d_{key}"] = round(d[key]["point"], 4)
        row[f"d_{key}_lo"] = round(d[key]["lo"], 4)
        row[f"d_{key}_hi"] = round(d[key]["hi"], 4)
        row[f"d_{key}_frac_positive"] = round(d[key]["frac_boot_positive"], 3)
    pair_rows.append(row)

pairs_df = pd.DataFrame(pair_rows)
pairs_df.to_csv(OUT / "official_paired.csv", index=False)

#---------- 打印 ----------
pd.set_option("display.width", 200)
print("\n=== Official metrics (95% CI, bootstrap over subjects) ===")
print(metrics_df.to_string(index=False))
print("\n=== Paired differences A - B (95% CI) ===")
print(pairs_df[["A", "B", "d_macro_f1", "d_macro_f1_lo", "d_macro_f1_hi", "d_macro_f1_frac_positive",
                "d_n1_f1", "d_n1_f1_lo", "d_n1_f1_hi"]].to_string(index=False))
print(f"\nSaved: {OUT / 'official_metrics.csv'}")
print(f"Saved: {OUT / 'official_paired.csv'}")