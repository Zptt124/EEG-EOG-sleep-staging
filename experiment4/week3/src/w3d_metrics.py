"""评估指标：Macro-F1、各阶段 F1、N1 precision/recall/AP、5x5 混淆矩阵、逐人指标。"""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
LABELS = list(range(5))


def compute_metrics(y, pred, prob, subj):
    y, pred, subj = np.asarray(y), np.asarray(pred), np.asarray(subj)
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    out = dict(n=int(len(y)), macro_f1=float(f1_score(y, pred, labels=LABELS, average='macro', zero_division=0)),
               accuracy=float((y == pred).mean()),
               per_class_f1={s: float(f[i]) for i, s in enumerate(STAGES)},
               per_class_precision={s: float(p[i]) for i, s in enumerate(STAGES)},
               per_class_recall={s: float(r[i]) for i, s in enumerate(STAGES)},
               support={s: int(sup[i]) for i, s in enumerate(STAGES)},
               n1_ap=float(average_precision_score((y == 1).astype(int), np.asarray(prob)[:, 1])),
               confusion_matrix=confusion_matrix(y, pred, labels=LABELS).tolist(),
               confusion_axes='rows=true W,N1,N2,N3,REM; cols=pred', per_subject={})
    for s in sorted(set(subj)):
        m = subj == s
        fs = f1_score(y[m], pred[m], labels=LABELS, average=None, zero_division=0)
        out['per_subject'][s] = dict(n=int(m.sum()),
                                     macro_f1=float(f1_score(y[m], pred[m], labels=LABELS, average='macro',
                                                             zero_division=0)),
                                     n1_f1=float(fs[1]))
    return out
