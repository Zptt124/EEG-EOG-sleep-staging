"""D：把 C 的阶段构成（逐次抽块）与 F1 差对接，看构成差能解释多少窗口效应（探索性，非预设）。

对每个 (窗口, 名单, 抽块)：
  y  = Macro-F1(新人新夜) − Macro-F1(X)，X = 原人原夜 / 随机（16 位开发评估者合并）
  dW, dN2, dREM, dN3 = 新人组与 X 组尝试标注段里各阶段占比之差（百分点），来自 C 的 stage_composition.xlsx
回归 1：y ~ 窗口
回归 2：y ~ 窗口 + 构成差
比较窗口系数在加入构成差前后的变化。标准误按名单聚类（名单级 bootstrap 2000 次，rng 0）。
只做描述：构成与 F1 的关联不等于因果。

用法：python 06_composition_vs_f1.py --c-src <dir> --fits-old <dir> --fits-lights <dir> --stage-csv <逐次抽块.csv> --out <dir>
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ARM_CN = {"原人原夜": "deepen", "原人另一夜": "other_night", "新人新夜": "add_people", "D 混合随机": "random"}
WIN_CN = {"旧窗口": 0, "关灯窗口": 1}
STAGES = ("W", "N1", "N2", "N3", "REM")


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def macro(conf):
    m = conf.reshape(*conf.shape[:-1], 5, 5)
    den = m.sum(-1) + m.sum(-2)
    diag = np.diagonal(m, axis1=-2, axis2=-1)
    return np.divide(200.0 * diag, den, out=np.zeros(den.shape), where=den > 0).mean(-1)


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def main():
    ap = argparse.ArgumentParser()
    for n in ("c-src", "fits-old", "fits-lights", "stage-csv", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.c_src / "experiment3/window_budget_c_20261128/results/paired_confusions.npz")
    rosters = [int(r) for r in z["rosters"]]
    f1 = np.zeros((len(rosters), 5, 2, 4))
    f1[:, :, :, :3] = macro(z["counts"].sum(-2))
    for j, ro in enumerate(rosters):
        f1[j, :, 0, 3] = macro(np.load(a.fits_old / f"roster_{ro:02d}.npz")["counts"][0].sum(-2))
        f1[j, :, 1, 3] = macro(np.load(a.fits_lights / f"roster_{ro:02d}.npz")["counts"][0].sum(-2))
    comp = {}
    with open(a.stage_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    hdr = rows[3]
    for r in rows[4:]:
        if not r or r[0] not in WIN_CN:
            continue
        d = dict(zip(hdr, r))
        key = (WIN_CN[d["窗口"]], int(d["名单号"]), int(d["抽块序号"]), ARM_CN[d["组别"]])
        comp[key] = np.array([100 * float(d[f"{s} %"]) for s in STAGES])
    if len(comp) != 2 * len(rosters) * 5 * 4:
        fail(f"unexpected composition rows: {len(comp)}")
    arm_ix = {"deepen": 0, "other_night": 1, "add_people": 2, "random": 3}
    out_rows = []
    for X in ("deepen", "random"):
        Y, W, D, G = [], [], [], []
        for wi in (0, 1):
            for j, ro in enumerate(rosters):
                for d in range(5):
                    Y.append(f1[j, d, wi, 2] - f1[j, d, wi, arm_ix[X]])
                    W.append(wi)
                    D.append(comp[wi, ro, d + 1, "add_people"] - comp[wi, ro, d + 1, X])
                    G.append(j)
        Y, W, D, G = np.array(Y), np.array(W, float), np.array(D), np.array(G)
        # 只放 W、N2、REM 三个构成差（五项占比和为 100，避免共线；N1、N3 占比小且波动大，作为敏感性另报）
        sets = {"window_only": [], "plus_W_N2_REM": [0, 2, 4], "plus_all_but_N1": [0, 2, 3, 4]}
        rng = np.random.default_rng(0)
        boots = {k: [] for k in sets}
        def fit(idx, cols):
            Xm = np.column_stack([np.ones(len(idx)), W[idx]] + [D[idx, c] for c in cols])
            return ols(Xm, Y[idx])
        allidx = np.arange(len(Y))
        point = {k: fit(allidx, c) for k, c in sets.items()}
        groups = [np.flatnonzero(G == j) for j in range(len(rosters))]
        for _ in range(2000):
            pick = rng.integers(len(rosters), size=len(rosters))
            idx = np.concatenate([groups[p] for p in pick])
            for k, c in sets.items():
                boots[k].append(fit(idx, c))
        for k, c in sets.items():
            b = np.array(boots[k])
            names = ["intercept", "window_lights_minus_old"] + [f"d_{STAGES[i]}_share_pp" for i in c]
            for i, nm in enumerate(names):
                lo, hi = np.percentile(b[:, i], [2.5, 97.5])
                out_rows.append(dict(contrast=f"new_minus_{X}", model=k, term=nm, coef=round(float(point[k][i]), 4),
                                     ci95_lo=round(float(lo), 4), ci95_hi=round(float(hi), 4)))
        # 构成差在两窗口的平均
        for wi, wn in ((0, "old"), (1, "lights_off")):
            m = D[W == wi].mean(0)
            out_rows.append(dict(contrast=f"new_minus_{X}", model="mean_share_diff", term=wn,
                                 coef=" ".join(f"{s}:{v:+.2f}" for s, v in zip(STAGES, m)), ci95_lo="", ci95_hi=""))
    with open(a.out / "composition_regression.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    for r in out_rows:
        if r["term"] != "intercept":
            print(r, flush=True)


if __name__ == "__main__":
    main()
