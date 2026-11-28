"""D：窗口追问（共同 26 份名单），只用已归档的混淆计数，不重新拟合。

(a) 逐阶段 F1：四组（原人原夜、原人另一夜、新人新夜、随机）× 两个窗口的五类各自 F1；
    以及 新人−原夜、新人−随机 的逐阶段差在两个窗口各是多少、跨窗口变了多少（配对区间）。
(b) 跨窗口配对差值区间：(关灯 新人−X) − (旧 新人−X)，X = 原人原夜 / 随机。
    原人原夜这一条用来对照 C 已报告的 −1.10 [−2.10, −0.12]（检验本脚本与 C 的口径一致）。
(c) 扩跑精度估计：把关灯窗口"新人−随机"的方差拆成 名单间 / 名单内抽块 / 评估者 三部分，
    估算抽块从 5 次增到 10、20、50 次时 95% 区间的半宽。

重抽样与 C 相同：名单有放回、名单内抽块序号有放回（两个窗口共用同一抽块序号）、两个 8 人评估队列内
multinomial(8) 权重（两个窗口共用），10,000 次，default_rng(0)。
主口径 = 固定尝试 600 段（全部抽块）；敏感性 = 严格等预算（关灯窗口里该比较涉及的两组都拿到 600 个标签）。

用法：python 05_window_followup.py --c-src <dir> --fits-old <dir> --fits-lights <dir> --out <dir>
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

N_BOOT = 10000
ARMS = ("deepen", "other_night", "add_people", "random")
STAGES = ("W", "N1", "N2", "N3", "REM")


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wr(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def class_f1(conf):
    m = conf.reshape(*conf.shape[:-1], 5, 5)
    den = m.sum(-1) + m.sum(-2)
    diag = np.diagonal(m, axis1=-2, axis2=-1)
    return np.divide(200.0 * diag, den, out=np.zeros(den.shape), where=den > 0)


def macro(conf):
    return class_f1(conf).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    for n in ("c-src", "fits-old", "fits-lights", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    wdir = a.c_src / "experiment3/window_budget_c_20261128/results"
    z = np.load(wdir / "paired_confusions.npz")
    rosters = [int(r) for r in z["rosters"]]
    R = len(rosters)
    cc_old = np.load(a.c_src / "experiment3/week3_budget_replication_20261128/results/confusion_counts.npz")["counts"][0]
    # C 在关灯归档里保存的旧窗口计数应与开发复验归档一致
    if not np.array_equal(z["counts"][:, :, 0], cc_old[[r - 1 for r in rosters]]):
        fail("C's old-window counts in the lights-off archive differ from the replication archive")
    # all[R, draw5, window2, arm4, 16, 25]
    allc = np.zeros((R, 5, 2, 4, 16, 25), np.int64)
    allc[:, :, :, :3] = z["counts"]
    good = {}
    sc = rd(wdir / "scores.csv")
    full = {(int(r["roster"]), int(r["draw"]), r["arm"]): int(r["scorable_additions"]) == 600 for r in sc}
    g_r = np.zeros((R, 5), bool)
    for j, ro in enumerate(rosters):
        fo = np.load(a.fits_old / f"roster_{ro:02d}.npz")
        fl = np.load(a.fits_lights / f"roster_{ro:02d}.npz")
        allc[j, :, 0, 3] = fo["counts"][0]
        allc[j, :, 1, 3] = fl["counts"][0]
        g_r[j] = fl["scorable_added"] == 600
    g = {arm: np.array([[full[ro, d + 1, arm] for d in range(5)] for ro in rosters]) for arm in ARMS[:3]}
    g["random"] = g_r

    cohorts = (("combined", slice(None)), ("dev8", slice(0, 8)), ("pretrain_eval8", slice(8, 16)))

    def boot_stats(stat_fn, mask, seed=0, n_boot=N_BOOT, resample=("roster", "draw", "eval")):
        """stat_fn(conf[...,R',5,2,4,25]) -> array [..., R', 5, K]；返回 点估计[K], 区间[2,K], 名单均值[R',K]"""
        idx = [np.flatnonzero(m) for m in mask]
        keep = [j for j in range(R) if len(idx[j])]
        sub = allc[keep]
        Rk = len(keep)
        sizes = np.array([len(idx[j]) for j in keep])
        pad = np.stack([np.pad(idx[j], (0, 5 - len(idx[j]))) for j in keep])
        cell = stat_fn(sub.sum(-2))
        rmeans = np.stack([cell[i][idx[j]].mean(0) for i, j in enumerate(keep)])
        point = rmeans.mean(0)
        flat = sub.reshape(-1, 16, 25).transpose(1, 0, 2).reshape(16, -1)
        rng = np.random.default_rng(seed)
        out = []
        for lo in range(0, n_boot, 50):
            n = min(50, n_boot - lo)
            rs = rng.integers(Rk, size=(n, Rk)) if "roster" in resample else np.tile(np.arange(Rk), (n, 1))
            nr = sizes[rs][..., None]
            if "draw" in resample:
                ds = pad[rs[..., None], (rng.random((n, Rk, 5)) * nr).astype(int)]
            else:
                ds = pad[rs]
            valid = np.arange(5) < nr
            if "eval" in resample:
                w = rng.multinomial(8, [1 / 8] * 8, size=(n, 2)).reshape(n, 16).astype(float)
            else:
                w = np.ones((n, 16))
            wc = (w @ flat).reshape(n, *sub.shape[:4], 25)
            c = stat_fn(wc)
            v = c[np.arange(n)[:, None, None], rs[..., None], ds]
            out.append(((v * valid[..., None]).sum(2) / nr).mean(1))
        b = np.concatenate(out)
        return point, np.percentile(b, [2.5, 97.5], axis=0), rmeans, b

    # ---------- (b) 跨窗口配对差 ----------
    def dd_stat(X):  # X arm index; returns [old, lights, lights-old]
        def f(conf):
            m = macro(conf)                       # [..., R,5,2,4]
            d = m[..., 2] - m[..., X]             # [..., R,5,2]
            return np.stack([d[..., 0], d[..., 1], d[..., 1] - d[..., 0]], -1)
        return f

    rows_b = []
    for X, xname in ((0, "deepen"), (3, "random")):
        for label, mask in (("fixed_600_attempts", np.ones((R, 5), bool)),
                            ("exact_600", g["add_people"] & g[ARMS[X]])):
            pt, ci, rm, _ = boot_stats(dd_stat(X), mask)
            for k, nm in enumerate(("old_window", "lights_off_window", "lights_minus_old")):
                rows_b.append(dict(contrast=f"new_minus_{xname}", analysis=label, quantity=nm, rosters=len(rm),
                                   draw_pairs=int(mask[[j for j in range(R) if mask[j].any()]].sum()),
                                   mean_pp=round(float(pt[k]), 4), ci95_lo=round(float(ci[0, k]), 4),
                                   ci95_hi=round(float(ci[1, k]), 4),
                                   positive_rosters=f"{int((rm[:, k] > 0).sum())}/{len(rm)}"))
    wr(a.out / "cross_window_paired.csv", rows_b)

    # ---------- (a) 逐阶段 F1 ----------
    def stage_stat(conf):
        f = class_f1(conf)                        # [..., R,5,2,4,5]
        nd = f[..., 2, :] - f[..., 0, :]           # new-deepen  [...,R,5,2,5]
        nr_ = f[..., 2, :] - f[..., 3, :]          # new-random
        parts = [nd[..., 0, :], nd[..., 1, :], nd[..., 1, :] - nd[..., 0, :],
                 nr_[..., 0, :], nr_[..., 1, :], nr_[..., 1, :] - nr_[..., 0, :]]
        return np.concatenate(parts, -1)           # 30
    pt, ci, rm, _ = boot_stats(stage_stat, np.ones((R, 5), bool))
    names = [(c, w) for c in ("new_minus_deepen", "new_minus_random") for w in ("old_window", "lights_off_window", "lights_minus_old")]
    rows_a = []
    for i, (c, w) in enumerate(names):
        for s, st in enumerate(STAGES):
            k = i * 5 + s
            rows_a.append(dict(contrast=c, quantity=w, stage=st, mean_pp=round(float(pt[k]), 3),
                               ci95_lo=round(float(ci[0, k]), 3), ci95_hi=round(float(ci[1, k]), 3)))
    wr(a.out / "per_stage_contrasts.csv", rows_a)
    f = class_f1(allc.sum(-2)).mean((0, 1))        # [2,4,5]
    rows_lvl = [dict(window=wn, arm=arm, **{st: round(float(f[wi, ai, s]), 2) for s, st in enumerate(STAGES)},
                     macro=round(float(f[wi, ai].mean()), 2))
                for wi, wn in enumerate(("old", "lights_off")) for ai, arm in enumerate(ARMS)]
    wr(a.out / "per_stage_f1_levels.csv", rows_lvl)

    # ---------- (c) 扩跑精度 ----------
    def nr_lights(conf):
        m = macro(conf)
        return (m[..., 1, 2] - m[..., 1, 3])[..., None]
    allmask = np.ones((R, 5), bool)
    pt0, ci0, rm0, b_full = boot_stats(nr_lights, allmask)
    _, _, _, b_eval = boot_stats(nr_lights, allmask, resample=("eval",), n_boot=4000)
    _, _, _, b_draw = boot_stats(nr_lights, allmask, resample=("draw",), n_boot=4000)
    cell = nr_lights(allc.sum(-2))[..., 0]          # [R,5]
    s2_within = float(cell.var(axis=1, ddof=1).mean())
    s2_rmeans = float(cell.mean(1).var(ddof=1))
    s2_between = max(s2_rmeans - s2_within / 5, 0.0)
    v_full, v_eval, v_draw = float(b_full.var()), float(b_eval.var()), float(b_draw.var())
    rows_c = [dict(component="bootstrap_total", variance=round(v_full, 5), note="roster+draw+evaluator resampling, 5 draws"),
              dict(component="evaluator_only", variance=round(v_eval, 5), note="rosters and draws fixed"),
              dict(component="draw_only", variance=round(v_draw, 5), note="within-roster draw resampling only"),
              dict(component="within_roster_draw_var_per_draw", variance=round(s2_within, 5), note="pooled var of draw-level deltas"),
              dict(component="between_roster_var", variance=round(s2_between, 5), note="var(roster means) - within/5")]
    proj = []
    for nd_ in (5, 10, 20, 50):
        # 抽块项按 5/nd 缩放，其余不变
        # 两级重抽样里抽块波动既出现在"名单内抽块"里，也混在"名单均值"里；
        # 保守估计只减一次，乐观估计减两次（上界的收益）
        v = v_full - v_draw * (1 - 5 / nd_)
        v_opt = v_full - 2 * v_draw * (1 - 5 / nd_)
        hw = 1.96 * np.sqrt(max(v, 0))
        hw_opt = 1.96 * np.sqrt(max(v_opt, 0))
        proj.append(dict(draws_per_roster=nd_, projected_var=round(v, 5), ci95_halfwidth_pp=round(float(hw), 3),
                         projected_ci_if_mean_unchanged=f"[{pt0[0] - hw:+.2f}, {pt0[0] + hw:+.2f}]",
                         optimistic_halfwidth_pp=round(float(hw_opt), 3),
                         optimistic_ci_if_mean_unchanged=f"[{pt0[0] - hw_opt:+.2f}, {pt0[0] + hw_opt:+.2f}]"))
    wr(a.out / "precision_components.csv", rows_c)
    wr(a.out / "precision_projection.csv", proj)

    for r in rows_b:
        print(r, flush=True)
    for r in rows_a:
        if r["quantity"] == "lights_minus_old":
            print(r, flush=True)
    print("current lights-off new-random", round(float(pt0[0]), 3), ci0[:, 0].round(3).tolist())
    for r in rows_c + proj:
        print(r, flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
