"""evaluate.py — 全组公共评估入口(B 维护,官方版 v1)。

输入:逐段验证集预测表(列顺序见 predictions_val.template.csv)
    sample_id, subject_id, record_id, epoch_index, y_true, y_pred, p_W, p_N1, p_N2, p_N3, p_REM
输出:Macro-F1(主指标)、每类 precision/recall/F1/support、N1 指标、
    混淆矩阵(行真实、列预测)、accuracy;可选按受试者聚类的 bootstrap 区间。

标签顺序固定:0=W, 1=N1, 2=N2, 3=N3, 4=REM。
指标只用 numpy 由混淆矩阵计算(不依赖 sklearn),便于手工核对;
tests/test_evaluate.py 用人工可算的假预测和 sklearn 交叉检查。

用法:
  python src/evaluation/evaluate.py runs/<run_id>/predictions_val.csv
  python src/evaluation/evaluate.py pred.csv --out metrics.json --bootstrap 1000
  python src/evaluation/evaluate.py pred_a.csv --compare pred_b.csv   # 配对 bootstrap:a 减 b

变更说明:本脚本取代 A 的临时版 scripts/evaluate.py。与 A 的 metrics.py 数值一致
(见 reports/b_recompute_check.md);若以后修改指标定义,需注明哪些结果要重算。
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES = 5
PROB_COLS = ["p_W", "p_N1", "p_N2", "p_N3", "p_REM"]
REQUIRED_COLS = ["sample_id", "subject_id", "record_id", "epoch_index",
                 "y_true", "y_pred"] + PROB_COLS
EVALUATE_VERSION = "v1"


# --------------------------------------------------------------------------- 读取与校验
def validate_predictions(df: pd.DataFrame) -> None:
    """格式不符直接抛 ValueError,不静默修复。"""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")
    if len(df) == 0:
        raise ValueError("预测表为空")
    if df[REQUIRED_COLS].isna().any().any():
        raise ValueError("预测表含缺失值")
    if not df["sample_id"].is_unique:
        raise ValueError(f"sample_id 不唯一: {int(df['sample_id'].duplicated().sum())} 个重复")
    for col in ("y_true", "y_pred"):
        v = df[col]
        if not np.issubdtype(v.dtype, np.integer) or v.min() < 0 or v.max() >= N_CLASSES:
            raise ValueError(f"{col} 必须是 0..{N_CLASSES - 1} 的整数标签")
    p = df[PROB_COLS].to_numpy(dtype=float)
    if (p < 0).any() or (p > 1).any():
        raise ValueError("概率应在 [0,1]")
    psum = p.sum(1)
    if (np.abs(psum - 1) > 0.01).any():
        raise ValueError(f"概率行和应约为 1,实际范围 {psum.min():.4f}..{psum.max():.4f}")
    if not (p.argmax(1) == df["y_pred"].to_numpy()).all():
        raise ValueError("y_pred 必须等于 (p_W..p_REM) 的 argmax")


def load_predictions(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_predictions(df)
    return df


# --------------------------------------------------------------------------- 指标
def confusion(y_true, y_pred) -> np.ndarray:
    """行真实、列预测,固定 5 类。"""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_div(a, b):
    """分母为 0 时返回 0(与 sklearn zero_division=0 一致)。"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    pred_n = cm.sum(0)
    prec = _safe_div(tp, pred_n)
    rec = _safe_div(tp, support)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    return {
        "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class_f1": {n: float(f1[i]) for i, n in enumerate(STAGE_NAMES)},
        "per_class_precision": {n: float(prec[i]) for i, n in enumerate(STAGE_NAMES)},
        "per_class_recall": {n: float(rec[i]) for i, n in enumerate(STAGE_NAMES)},
        "per_class_support": {n: int(support[i]) for i, n in enumerate(STAGE_NAMES)},
        "n1_precision": float(prec[1]),
        "n1_recall": float(rec[1]),
        "n1_f1": float(f1[1]),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def compute_metrics(y_true, y_pred) -> dict:
    return metrics_from_confusion(confusion(y_true, y_pred))


def per_subject_confusions(df: pd.DataFrame):
    """返回 (subject_ids, 形状 (S,5,5) 的混淆矩阵数组)。"""
    subs = sorted(df["subject_id"].unique())
    cms = np.stack([confusion(g["y_true"], g["y_pred"])
                    for _, g in df.groupby("subject_id", sort=True)])
    return subs, cms


def per_subject_table(df: pd.DataFrame) -> pd.DataFrame:
    subs, cms = per_subject_confusions(df)
    rows = []
    for s, cm in zip(subs, cms):
        m = metrics_from_confusion(cm)
        rows.append({"subject_id": s, "n_epochs": int(cm.sum()),
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "n1_f1": m["n1_f1"], "n1_support": m["per_class_support"]["N1"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 不确定性
def bootstrap_subjects(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """按受试者重采样(有放回)的 bootstrap:同一人的片段相关,不能按片段重采样。

    返回 macro_f1 / n1_f1 / accuracy 的点估计与百分位区间。受试者很少(验证集 8 人)
    时区间会很宽,这是真实不确定性,不是缺陷。
    """
    subs, cms = per_subject_confusions(df)
    rng = np.random.default_rng(seed)
    S = len(subs)
    draws = rng.integers(0, S, size=(n_boot, S))
    stats = {"macro_f1": [], "n1_f1": [], "accuracy": []}
    for d in draws:
        m = metrics_from_confusion(cms[d].sum(0))
        for k in stats:
            stats[k].append(m[k])
    point = metrics_from_confusion(cms.sum(0))
    out = {"n_boot": n_boot, "seed": seed, "n_subjects": S, "level": 1 - alpha}
    for k, v in stats.items():
        lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = {"point": float(point[k]), "lo": float(lo), "hi": float(hi)}
    return out


def paired_bootstrap_delta(df_a: pd.DataFrame, df_b: pd.DataFrame,
                           n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """同一批验证受试者上 a 相对 b 的指标差(a - b)的配对 bootstrap。

    要求两张表 sample_id 集合与 y_true 完全一致。
    """
    a = df_a.sort_values("sample_id").reset_index(drop=True)
    b = df_b.sort_values("sample_id").reset_index(drop=True)
    if len(a) != len(b) or not (a["sample_id"].to_numpy() == b["sample_id"].to_numpy()).all():
        raise ValueError("两张预测表的 sample_id 集合不一致,无法配对比较")
    if not (a["y_true"].to_numpy() == b["y_true"].to_numpy()).all():
        raise ValueError("两张预测表同一 sample_id 的 y_true 不一致")
    subs_a, cms_a = per_subject_confusions(a)
    subs_b, cms_b = per_subject_confusions(b)
    assert subs_a == subs_b
    rng = np.random.default_rng(seed)
    S = len(subs_a)
    draws = rng.integers(0, S, size=(n_boot, S))
    keys = ["macro_f1", "n1_f1", "accuracy"]
    deltas = {k: [] for k in keys}
    for d in draws:
        ma, mb = metrics_from_confusion(cms_a[d].sum(0)), metrics_from_confusion(cms_b[d].sum(0))
        for k in keys:
            deltas[k].append(ma[k] - mb[k])
    pa, pb = metrics_from_confusion(cms_a.sum(0)), metrics_from_confusion(cms_b.sum(0))
    out = {"n_boot": n_boot, "seed": seed, "n_subjects": S, "level": 1 - alpha}
    for k in keys:
        lo, hi = np.percentile(deltas[k], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = {"point": float(pa[k] - pb[k]), "lo": float(lo), "hi": float(hi),
                  "frac_boot_positive": float(np.mean(np.array(deltas[k]) > 0))}
    return out


# --------------------------------------------------------------------------- 输出
def metrics_table(m: dict) -> str:
    lines = [f"accuracy : {m['accuracy']:.4f}",
             f"macro-F1 : {m['macro_f1']:.4f}   <- main metric"]
    for n in STAGE_NAMES:
        lines.append(f"  {n:<4} P {m['per_class_precision'][n]:.4f}  "
                     f"R {m['per_class_recall'][n]:.4f}  F1 {m['per_class_f1'][n]:.4f}  "
                     f"support {m['per_class_support'][n]}")
    lines.append(f"N1 precision: {m['n1_precision']:.4f}   N1 recall: {m['n1_recall']:.4f}"
                 f"   N1 F1: {m['n1_f1']:.4f}")
    lines.append("confusion matrix (rows=true, cols=pred):")
    lines.append("         " + " ".join(f"{n:>6}" for n in STAGE_NAMES))
    for i, n in enumerate(STAGE_NAMES):
        lines.append(f"  {n:<7}" + " ".join(f"{v:>6}" for v in m["confusion_matrix"][i]))
    return "\n".join(lines)


def evaluate(path, n_boot: int = 0, seed: int = 0) -> dict:
    df = load_predictions(path)
    m = compute_metrics(df["y_true"], df["y_pred"])
    m["n_epochs"] = int(len(df))
    m["n_subjects"] = int(df["subject_id"].nunique())
    m["class_order"] = STAGE_NAMES
    m["evaluate_version"] = EVALUATE_VERSION
    m["source"] = str(path)
    if n_boot:
        m["bootstrap_subjects"] = bootstrap_subjects(df, n_boot, seed)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("predictions_csv")
    ap.add_argument("--out", default=None, help="写出 metrics.json")
    ap.add_argument("--bootstrap", type=int, default=0, help="按受试者 bootstrap 次数(0=不做)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap 种子")
    ap.add_argument("--compare", default=None, help="另一张预测表,输出 本表-对方 的配对差")
    args = ap.parse_args(argv)

    try:
        m = evaluate(args.predictions_csv, args.bootstrap, args.seed)
        print(metrics_table(m))
        if "bootstrap_subjects" in m:
            b = m["bootstrap_subjects"]
            print(f"\n按受试者 bootstrap(n={b['n_boot']}, {b['n_subjects']} 人, "
                  f"{int(b['level'] * 100)}% 区间):")
            for k in ("macro_f1", "n1_f1", "accuracy"):
                print(f"  {k:<9} {b[k]['point']:.4f}  [{b[k]['lo']:.4f}, {b[k]['hi']:.4f}]")
        if args.compare:
            d = paired_bootstrap_delta(load_predictions(args.predictions_csv),
                                       load_predictions(args.compare),
                                       args.bootstrap or 1000, args.seed)
            m["paired_delta_vs"] = {"other": args.compare, **d}
            print(f"\n配对差(本表 - {args.compare}):")
            for k in ("macro_f1", "n1_f1", "accuracy"):
                print(f"  Δ{k:<9} {d[k]['point']:+.4f}  [{d[k]['lo']:+.4f}, {d[k]['hi']:+.4f}]")
    except ValueError as e:
        sys.exit(f"错误: {e}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
