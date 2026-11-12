"""w3_epoch_selection.py — 第三周训练侧选轮共用逻辑(照抄 B 的 27_b_w3_linear_probe.py)。

第三周选模协议:开发集只用于最终评估一次;轮次选择全部在训练侧完成。
  - 折分配:FOLD_SEED=0,对排序后的受试者 id 做固定置换再 array_split。
    折分配与下游 seed、pooling、模型无关,保证不同方法/模态/预算共用同一
    折划分(配对比较的前提),与 B 的 27 号脚本一致。
  - 选轮:各折固定训练 MAX_EPOCHS 轮,逐轮对留出折预测;拼接留出预测(OOF)
    后按 Macro-F1 取最高轮次 E*(并列取最早)。
  - 最终:在全部预算内训练者上重新标准化、同 seed 训练 E* 轮,对开发集只
    预测一次。
"""
from __future__ import annotations

import numpy as np

FOLD_SEED = 0            # 与 B 的 27_b_w3_linear_probe.py 一致:固定,与下游 seed 无关


def subject_folds(subjects: list[str], k: int) -> list[list[str]]:
    """按受试者分 k 折;实现与 27_b_w3_linear_probe.py 逐位一致。"""
    order = np.random.default_rng(FOLD_SEED).permutation(sorted(subjects))
    return [sorted(f.tolist()) for f in np.array_split(order, k)]


def pick_epoch(curve: list[dict]) -> dict:
    """curve 每项含 epoch / oof_macro_f1 / oof_n1_f1;返回宏 F1 最高(并列取最早)。"""
    return max(curve, key=lambda r: (r["oof_macro_f1"], -r["epoch"]))


def oof_curve(y_true: np.ndarray, oof_probs: np.ndarray, macro_f1_fn) -> list[dict]:
    """拼接留出预测逐轮选轮曲线。oof_probs:(epochs, N, 5)。"""
    curve = []
    for e in range(oof_probs.shape[0]):
        m = macro_f1_fn(y_true, oof_probs[e].argmax(1))
        curve.append({"epoch": e + 1, "oof_macro_f1": m})
    return curve
