"""Metric computation shared by training and the evaluate.py CLI.

Team doc section 二: Macro-F1 is the main metric; also per-class F1,
N1 precision/recall, confusion matrix, accuracy.
"""
import json

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, names=None) -> dict:
    names = names or [f"c{i}" for i in range(5)]
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(names)))
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels,
                                   zero_division=0)),
        "per_class_f1": {names[c]: float(f)
                         for c, f in enumerate(f1_score(y_true, y_pred, average=None,
                                                        labels=labels, zero_division=0))},
        "per_class_precision": {names[c]: float(p)
                                for c, p in enumerate(precision_score(y_true, y_pred, average=None,
                                                                      labels=labels, zero_division=0))},
        "per_class_recall": {names[c]: float(r)
                             for c, r in enumerate(recall_score(y_true, y_pred, average=None,
                                                                labels=labels, zero_division=0))},
        "per_class_support": {names[c]: int((y_true == c).sum()) for c in labels},
        "n1_precision": float(precision_score(y_true, y_pred, labels=[1],
                                              average="macro", zero_division=0)),
        "n1_recall": float(recall_score(y_true, y_pred, labels=[1],
                                        average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    return m


def metrics_table(m: dict, names=None) -> str:
    names = names or ["W", "N1", "N2", "N3", "REM"]
    lines = [f"accuracy : {m['accuracy']:.4f}",
             f"macro-F1 : {m['macro_f1']:.4f}   <- main metric"]
    for c, n in enumerate(names):
        lines.append(f"  F1 {n:<4}: {m['per_class_f1'][n]:.4f}")
    lines.append(f"N1 precision: {m['n1_precision']:.4f}   "
                 f"N1 recall: {m['n1_recall']:.4f}")
    lines.append("confusion matrix (rows=true, cols=pred):")
    cm = np.array(m["confusion_matrix"])
    header = "         " + " ".join(f"{n:>6}" for n in names)
    lines.append(header)
    for c, n in enumerate(names):
        lines.append(f"  {n:<7}" + " ".join(f"{v:>6}" for v in cm[c]))
    return "\n".join(lines)


def save_metrics(m: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
