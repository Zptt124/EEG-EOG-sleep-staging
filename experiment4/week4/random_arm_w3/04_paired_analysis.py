"""D：随机组的配对比较与覆盖统计。

主对照：新人新夜 − 随机（五类 Macro-F1，百分点）；另报 随机 − 原人原夜、随机 − 原人另一夜 作参考。
- 三组（C）的混淆计数来自 C 的归档（old：confusion_counts.npz；lights_off：paired_confusions.npz 的关灯窗口），
  随机组的计数来自 02 号输出；同一名单、同一抽块序号配对，评估者相同（16 人）。
- 点估计：每个 (名单, 抽块) 算差值，名单内平均，再按名单等权平均（与 C 相同）。
- 95% 区间：沿用 C 的重抽样（名单有放回、名单内抽块有放回、两个 8 人评估队列内受试者 multinomial 权重），
  10,000 次，numpy default_rng(0)。
- lights_off 另报"严格等预算"口径：新人新夜与随机组都实际取得 600 个五类标签的抽块。
- old 窗口另报 seed 0/1/2（名单 1–3）的敏感性。

用法：python 04_paired_analysis.py --window old|lights_off --c-src <dir> --fits <02 输出> --sampler-out <01 输出> --out <目录>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

N_BOOT = 10000
ARM_IX = {"deepen": 0, "other_night": 1, "add_people": 2}


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


def macro(conf):
    m = conf.reshape(*conf.shape[:-1], 5, 5)
    den = m.sum(-1) + m.sum(-2)
    diag = np.diagonal(m, axis1=-2, axis2=-1)
    return np.divide(200.0 * diag, den, out=np.zeros(den.shape), where=den > 0).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=("old", "lights_off"), required=True)
    for n in ("c-src", "fits", "sampler-out", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    slock = json.loads((a.sampler_out / "sampler_lock.json").read_text())
    rosters = slock["rosters"]
    R = len(rosters)
    if a.window == "old":
        cc = np.load(a.c_src / "experiment3/week3_budget_replication_20261128/results/confusion_counts.npz")["counts"]
        c_counts = cc[0][[r - 1 for r in rosters]]                      # [R,5,3,16,25] seed 0
        good_c = np.ones((R, 5), bool)
    else:
        z = np.load(a.c_src / "experiment3/window_budget_c_20261128/results/paired_confusions.npz")
        if list(z["rosters"]) != rosters:
            fail("roster order differs from C")
        c_counts = z["counts"][:, :, 1]                                  # lights-off window
        sc = rd(a.c_src / "experiment3/window_budget_c_20261128/results/scores.csv")
        good_c = np.zeros((R, 5), bool)
        full = {(int(r["roster"]), int(r["draw"]), r["arm"]): int(r["scorable_additions"]) == 600 for r in sc}
        for j, ro in enumerate(rosters):
            for d in range(5):
                good_c[j, d] = full[ro, d + 1, "add_people"]
    rand = np.zeros((R, 5, 16, 25), np.int64)
    good_r = np.zeros((R, 5), bool)
    seed_rand = {}
    for j, ro in enumerate(rosters):
        f = np.load(a.fits / f"roster_{ro:02d}.npz")
        rand[j] = f["counts"][0]
        good_r[j] = f["scorable_added"] == 600
        if len(f["seeds"]) == 3:
            seed_rand[ro] = f["counts"]
    # 统一数组 [R,5,4,16,25]：0 deepen,1 other,2 add,3 random
    allc = np.concatenate([c_counts, rand[:, :, None]], axis=2)

    contrasts = [("new_minus_random", 2, 3), ("random_minus_deepen", 3, 0), ("random_minus_other_night", 3, 1)]
    cohorts = (("combined", slice(None)), ("dev8", slice(0, 8)), ("pretrain_eval8", slice(8, 16)))

    def summarize(mask, label):
        idx = [np.flatnonzero(m) for m in mask]
        keep = [j for j in range(R) if len(idx[j])]
        if not keep:
            return []
        res = []
        rng = np.random.default_rng(0)
        flat = allc[keep].reshape(-1, 16, 25).transpose(1, 0, 2).reshape(16, -1)
        Rk = len(keep)
        sizes = np.array([len(idx[j]) for j in keep])
        pad = np.stack([np.pad(idx[j], (0, 5 - len(idx[j]))) for j in keep])
        boot = np.empty((N_BOOT, len(cohorts), len(contrasts)))
        for lo in range(0, N_BOOT, 100):
            n = min(100, N_BOOT - lo)
            rs = rng.integers(Rk, size=(n, Rk))
            u = rng.random((n, Rk, 5))
            nr = sizes[rs][..., None]
            ds = pad[rs[..., None], (u * nr).astype(int)]
            valid = np.arange(5) < nr
            w = rng.multinomial(8, [1 / 8] * 8, size=(n, 2)).reshape(n, 16)
            for ci, (_, sl) in enumerate(cohorts):
                wc = (w[:, sl] @ flat[sl]).reshape(n, Rk, 5, 4, 25)
                f1 = macro(wc)
                for k, (_, hi, lo_) in enumerate(contrasts):
                    v = (f1[..., hi] - f1[..., lo_])[np.arange(n)[:, None, None], rs[..., None], ds]
                    boot[lo:lo + n, ci, k] = ((v * valid).sum(2) / nr[..., 0]).mean(1)
        for ci, (cname, sl) in enumerate(cohorts):
            f1 = macro(allc[keep][..., sl, :].sum(-2))
            for k, (name, hi, lo_) in enumerate(contrasts):
                cell = f1[..., hi] - f1[..., lo_]
                rv = np.array([cell[i][idx[j]].mean() for i, j in enumerate(keep)])
                lo95, hi95 = np.percentile(boot[:, ci, k], [2.5, 97.5])
                res.append(dict(window=a.window, analysis=label, cohort=cname, contrast=name,
                                rosters=Rk, draw_pairs=int(sizes.sum()), mean_pp=round(float(rv.mean()), 4),
                                ci95_lo=round(float(lo95), 4), ci95_hi=round(float(hi95), 4),
                                positive_rosters=f"{int((rv > 0).sum())}/{Rk}"))
        return res

    rows = summarize(np.ones((R, 5), bool), "all_draws" if a.window == "old" else "fixed_600_attempts")
    if a.window == "lights_off":
        rows += summarize(good_c & good_r, "exact_600_new_and_random")
    wr(a.out / "paired_summary.csv", rows)

    # 逐名单
    f1 = macro(allc.sum(-2))
    per = []
    for j, ro in enumerate(rosters):
        per.append(dict(roster=ro, **{f"{n}_pp": round(float((f1[j, :, h] - f1[j, :, l]).mean()), 4) for n, h, l in contrasts},
                        random_scorable_600=f"{int(good_r[j].sum())}/5"))
    wr(a.out / "per_roster.csv", per)

    # seed 敏感性（old，名单 1–3）
    if seed_rand:
        cc = np.load(a.c_src / "experiment3/week3_budget_replication_20261128/results/confusion_counts.npz")["counts"]
        srows = []
        for s in range(3):
            v = [float((macro(cc[s, ro - 1, :, 2].sum(-2)) - macro(seed_rand[ro][s].sum(-2))).mean()) for ro in sorted(seed_rand)]
            srows.append(dict(head_seed=s, rosters="1-3", new_minus_random_mean_pp=round(float(np.mean(v)), 4)))
        wr(a.out / "seed_check.csv", srows)

    # 覆盖统计
    cov = rd(a.sampler_out / "random_coverage.csv")
    keys = ("people", "nights", "from_deepen", "from_other_night", "from_add_people")
    agg = {k: [int(c[k]) for c in cov] for k in keys}
    cov_sum = [dict(window=a.window, statistic=k, mean=round(float(np.mean(v)), 3), min=min(v), max=max(v)) for k, v in agg.items()]
    wr(a.out / "coverage_summary.csv", cov_sum)
    for r in rows:
        if r["cohort"] == "combined":
            print(r, flush=True)
    for c in cov_sum:
        print(c, flush=True)


if __name__ == "__main__":
    main()
