#label_guard.py —— 标签隐藏工具（第二周 实验④ 半监督）
#规则：只有名单里的 5/14 名有标签训练者可以使用真标签；其余 49/40 名训练者的 y 一律置为 -1
#所有训练脚本都通过这里拿训练标签，不直接读 feature_index.csv 里的 y_true
#用法：from label_guard import load_roles, hide_labels, assert_hidden

import csv
import sys

import numpy as np

HIDDEN = -1  #无标签样本的标签值


class LabelLeakError(Exception):
    #发现无标签样本带着真标签时抛出（调用方应 sys.exit(1)）
    pass


def load_roles(path, budget):
    #读 make_label_roles.py 生成的 labeled_unlabeled_subjects.csv
    #budget 取 "label10" 或 "label25"；返回 (有标签训练者集合, 无标签训练者集合)
    col = f"role_{budget}"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or col not in rows[0]:
        print(f"ERROR: column {col} not found in {path}")
        sys.exit(1)
    labeled = {r["subject_id"] for r in rows if r[col] == "labeled"}
    unlabeled = {r["subject_id"] for r in rows if r[col] == "unlabeled"}
    if labeled & unlabeled:
        print(f"ERROR: subjects are both labeled and unlabeled: {sorted(labeled & unlabeled)}")
        sys.exit(1)
    return labeled, unlabeled


def hide_labels(y, subject, split, labeled, unlabeled):
    #输入：全部样本的 y、subject_id、split（numpy 数组，顺序一致）
    #输出：y_train —— 训练集样本里，有标签者保留真标签，无标签者置 -1；验证集样本保持原样（只用于评估）
    #返回的 is_labeled / is_unlabeled 是按样本的布尔掩码
    y = np.asarray(y).copy()
    subject = np.asarray(subject)
    split = np.asarray(split)
    is_train = split == "train"
    is_labeled = is_train & np.isin(subject, list(labeled))
    is_unlabeled = is_train & np.isin(subject, list(unlabeled))
    #训练集里每个人都必须有明确角色，不能漏
    orphan = is_train & ~is_labeled & ~is_unlabeled
    if orphan.any():
        raise LabelLeakError(f"training subjects without a role: {sorted(set(subject[orphan]))}")
    y[is_unlabeled] = HIDDEN
    return y, is_labeled, is_unlabeled


def assert_hidden(y_train, is_unlabeled, is_labeled=None):
    #断言：无标签样本的 y 必须全是 -1；有标签样本的 y 必须在 0..4
    #每个训练脚本在开始训练前、以及生成伪标签前都要调用
    y_train = np.asarray(y_train)
    leaked = int((y_train[is_unlabeled] != HIDDEN).sum())
    if leaked:
        raise LabelLeakError(f"{leaked} unlabeled samples still carry a true label")
    if is_labeled is not None:
        bad = y_train[is_labeled]
        if ((bad < 0) | (bad > 4)).any():
            raise LabelLeakError("labeled samples have labels outside 0..4")
