"""D：独立复现成员 C 的 Ying EDF 记录起点敏感性分析（A/B × 新人/混合随机，580 次拟合）。

D 自己按任务书规则重写选块（不调用 C 的 make_plan），用同一份冻结特征和第一周 trainer 重跑全部拟合，
再用 D 自己的配对重抽样计算区间，最后逐项与 C 的归档比对：
  1. 选块：与 C 的 selected_blocks.csv 逐块一致；
  2. 混淆计数：与 C 的 paired_confusions.npz 逐格一致；
  3. 点估计与区间：与 C 的 summary.json 对照（区间只要求接近）。

规则（与 C 协议一致）：
  - 窗口：EDF 第 0 段起最多 1,080 段（9 小时），不超过完整信号段数；三个时段为第 0–359、360–719、720–1079 段；
    每 60 段切一块，只保留完整落在窗口内、且与该名单已标起点块相距 ≥ 60 段的块。
  - B：只保留完整落在 [首个共识睡眠段 − 60, 末个共识睡眠段 + 61) ∩ [0, 共识长度) 内的块。
  - 新人组只要有一格 B 无块，该名单退出。
  - 优先级：numpy default_rng([2026112903, 折, 名单, 重复, 1]).random(n)，块键按 "记录:起始段" 字符串排序；小者优先。
  - 新人组每格取 1 块；混合随机按新人组的时段配额，在该时段三种来源合并去重后取最前的块。
  - 训练 = 起点 1,200 段 + 新增 600 段（按 (记录, 起始段) 排序拼接），去掉无法评分段；head seed 0、30 轮。
  - 评估 = 该折 5 名评估者两晚的全部可评分段。
  - 点估计按折等权；区间：每折内名单、重复、评估者（multinomial(5)）三层重抽样，10,000 次，default_rng(0)。

用法：python 02_independent_replay.py --pkg <C 交接解压根目录> --out <目录>
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

SEED = 2026112903
BLOCK = 60
N_BOOT = 10000


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def macro(c):
    m = c.reshape(*c.shape[:-1], 5, 5)
    den = m.sum(-1) + m.sum(-2)
    diag = np.diagonal(m, axis1=-2, axis2=-1)
    return np.divide(200.0 * diag, den, out=np.zeros(den.shape), where=den > 0).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    src = a.pkg / "output/budget_external_ying"
    res = a.pkg / "output/budget_external_ying_edf_proxy_20261128"
    sys.path.insert(0, str(a.pkg / "experiment3/week1_budget_20261126"))
    from train import fit
    torch.set_num_threads(4)

    folds = json.loads((src / "protocol.json").read_text())["evaluation_folds"]
    timing = {f"{r['subject_id']}_Day{r['day']}": r for r in
              rd(a.pkg / "output/budget_screening_cd_20261128/C_to_D_verification/ying_day12_recording_zero.csv")}
    data = {}
    for f in sorted((src / "features").glob("*.npz")):
        z = np.load(f)
        data[f.stem] = (z["x"], z["y"].astype(int), int(timing[f.stem]["complete_signal_epochs"]))
    if len(data) != 38:
        fail("expected 38 nights")
    lim = {}
    for rec, (x, y, phys) in data.items():
        win = min(1080, phys)
        sl = np.flatnonzero(y > 0)
        lim[rec] = (win, max(0, int(sl[0]) - BLOCK), min(len(y), int(sl[-1]) + BLOCK + 1))
    groups = defaultdict(list)
    for r in rd(src / "blocks.csv"):
        groups[int(r["fold"]), int(r["roster"])].append(r)

    picks, retained, feas = {}, [], []
    for (fo, ro), rows in sorted(groups.items()):
        anchor = sorted({(r["record"], int(r["start"])) for r in rows if r["arm"] == "base"})
        cells = sorted({(r["arm"], r["record"], int(r["third"])) for r in rows if r["arm"] != "base"})
        if len(anchor) != 20 or len(cells) != 30:
            fail(f"fold {fo} roster {ro}: bad historical design")
        cand = {}
        for arm, rec, th in cells:
            win, lo, hi = lim[rec]
            base = [e for r_, e in anchor if r_ == rec]
            A = [(rec, e) for e in range((th - 1) * 360, th * 360, BLOCK)
                 if e + BLOCK <= win and all(abs(e - k) >= BLOCK for k in base)]
            B = [(rec, e) for _, e in A if lo <= e and e + BLOCK <= hi]
            cand[arm, rec, th] = {"A": A, "B": B}
        empty = [f"{rec}:third{th}" for (arm, rec, th), v in cand.items() if arm == "add_people" and not v["B"]]
        feas.append(dict(fold=fo, roster=ro, eligible=int(not empty), reason="|".join(empty)))
        if empty:
            continue
        retained.append((fo, ro, anchor))
        quota = Counter(th for arm, _, th in cells if arm == "add_people")
        keys = sorted({k for v in cand.values() for k in v["A"]}, key=lambda k: f"{k[0]}:{k[1]}")
        for d in range(1, 6):
            u = np.random.default_rng([SEED, fo, ro, d, 1]).random(len(keys))
            pr = {k: (float(u[i]), f"{k[0]}:{k[1]}") for i, k in enumerate(keys)}
            for cond in ("A", "B"):
                new = [min(cand[c][cond], key=lambda k: pr[k]) for c in cells if c[0] == "add_people"]
                mix = []
                for th, q in sorted(quota.items()):
                    pool = sorted({k for c, v in cand.items() if c[2] == th for k in v[cond]}, key=lambda k: pr[k])
                    if len(pool) < q:
                        fail("mixed quota infeasible")
                    mix += pool[:q]
                for st, ch in (("add_people", new), ("random", mix)):
                    if len(set(ch)) != 10 or set(ch) & set(anchor):
                        fail(f"invalid allocation {fo} {ro} {d} {cond} {st}")
                    picks[fo, ro, d, cond, st] = sorted(ch)
    if Counter(f for f, _, _ in retained) != Counter({1: 8, 2: 6, 3: 8, 4: 7}):
        fail(f"retained rosters by fold differ: {Counter(f for f, _, _ in retained)}")

    # 1. 选块比对
    c_sel = defaultdict(list)
    for r in rd(res / "selected_blocks.csv"):
        c_sel[int(r["fold"]), int(r["roster"]), int(r["draw"]), r["condition"], r["strategy"]].append((r["record"], int(r["first_epoch"])))
    sel_same = set(c_sel) == set(picks) and all(sorted(c_sel[k]) == picks[k] for k in picks)
    print(f"selections identical to C: {sel_same} ({len(picks)} units)", flush=True)

    # 2. 重跑拟合
    counts = np.zeros((len(retained), 5, 2, 2, 5, 25), dtype=np.int64)
    scor = np.zeros((len(retained), 5, 2, 2), dtype=np.int64)
    order = []
    for f_i, test in enumerate(folds, 1):
        ex, ey, eg = [], [], []
        for j, s in enumerate(test):
            for rec, (x, y, _) in sorted(data.items()):
                if rec.split("_")[0] == s:
                    ok = np.flatnonzero(y >= 0)
                    ex.append(x[ok]); ey.append(y[ok]); eg += [j] * len(ok)
        ex, ey, eg = np.concatenate(ex), np.concatenate(ey), np.array(eg)
        for gi, (fo, ro, anchor) in enumerate(retained):
            if fo != f_i:
                continue
            order.append((fo, ro))
            for d in range(1, 6):
                for ci, cond in enumerate("AB"):
                    for si, st in enumerate(("add_people", "random")):
                        bl = picks[fo, ro, d, cond, st]
                        scor[gi, d - 1, ci, si] = sum(int((data[r][1][e:e + BLOCK] >= 0).sum()) for r, e in bl)
                        inp = sorted(anchor + bl)
                        x = np.concatenate([data[r][0][e:e + BLOCK] for r, e in inp])
                        y = np.concatenate([data[r][1][e:e + BLOCK] for r, e in inp])
                        model, mu, sd, _ = fit(x[y >= 0], y[y >= 0], 0, 30)
                        with torch.inference_mode():
                            p = model(torch.from_numpy((ex - mu) / sd)).argmax(1).numpy()
                        counts[gi, d - 1, ci, si] = np.bincount(eg * 25 + ey * 5 + p, minlength=125).reshape(5, 25)
            print(f"fold {fo} roster {ro:02d} done", flush=True)
    order_idx = [order.index((fo, ro)) for fo, ro, _ in retained]  # retained already fold-sorted
    cz = np.load(res / "paired_confusions.npz")
    c_groups = [tuple(int(v) for v in g) for g in cz["groups"]]
    mine = {(fo, ro): counts[i] for i, (fo, ro, _) in enumerate(retained)}
    cnt_same = all(np.array_equal(mine[g], cz["counts"][i]) for i, g in enumerate(c_groups))
    scor_mine = {(fo, ro): scor[i] for i, (fo, ro, _) in enumerate(retained)}
    scor_same = all(np.array_equal(scor_mine[g], cz["scorable"][i]) for i, g in enumerate(c_groups))
    print(f"confusion counts identical to C: {cnt_same}; scorable counts identical: {scor_same}", flush=True)

    # 3. 区间（D 自己的重抽样）
    C_ = np.stack([mine[g] for g in c_groups])
    S_ = np.stack([scor_mine[g] for g in c_groups])
    fold_of = np.array([g[0] for g in c_groups])

    def stat(f1):                      # [..., R,5,2,2]
        d = f1[..., 0] - f1[..., 1]
        return np.stack([d[..., 0], d[..., 1], d[..., 1] - d[..., 0]], -1)
    out_rows = []
    rng = np.random.default_rng(0)
    for label, good in (("fixed_600_attempts", np.ones(S_.shape[:2], bool)), ("exact_600", (S_ == 600).all(axis=(2, 3)))):
        est, boot = [], np.zeros((N_BOOT, 3))
        npairs = 0
        for fo in (1, 2, 3, 4):
            ix = np.flatnonzero((fold_of == fo) & good.any(1))
            sub, msk = C_[ix], good[ix]
            npairs += int(msk.sum())
            cell = stat(macro(sub.sum(-2)))
            est.append(np.stack([cell[i][msk[i]].mean(0) for i in range(len(ix))]).mean(0))
            idx = [np.flatnonzero(m) for m in msk]
            sizes = np.array([len(i) for i in idx])
            pad = np.stack([np.pad(i, (0, 5 - len(i))) for i in idx])
            flat = sub.reshape(-1, 5, 25).transpose(1, 0, 2).reshape(5, -1)
            n = len(ix)
            for lo in range(0, N_BOOT, 100):
                rs = rng.integers(n, size=(100, n))
                nr = sizes[rs][..., None]
                ds = pad[rs[..., None], (rng.random((100, n, 5)) * nr).astype(int)]
                w = rng.multinomial(5, [0.2] * 5, size=100).astype(float)
                c = stat(macro((w @ flat).reshape(100, *sub.shape[:4], 25)))
                v = c[np.arange(100)[:, None, None], rs[..., None], ds]
                boot[lo:lo + 100] += ((v * (np.arange(5) < nr)[..., None]).sum(2) / nr).mean(1) / 4
        est = np.stack(est)
        for k, name in enumerate(("delta_A", "delta_B", "delta_B_minus_A")):
            lo95, hi95 = np.percentile(boot[:, k], [2.5, 97.5])
            out_rows.append(dict(analysis=label, contrast=name, rosters=int(good.any(1).sum()), draw_pairs=npairs,
                                 mean_pp=round(float(est[:, k].mean()), 4), ci95_lo=round(float(lo95), 4),
                                 ci95_hi=round(float(hi95), 4),
                                 fold_means=";".join(f"{v:+.3f}" for v in est[:, k])))
    cs = json.loads((res / "summary.json").read_text())
    for r in out_rows:
        c = cs[r["analysis"]]["contrasts"][r["contrast"]]
        r["C_mean_pp"] = round(c["mean_pp"], 4)
        r["C_ci95"] = f"[{c['ci95_pp'][0]:.4f}, {c['ci95_pp'][1]:.4f}]"
        print(r, flush=True)
    with open(a.out / "independent_replay_D.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    with open(a.out / "roster_feasibility_D.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(feas[0]))
        w.writeheader()
        w.writerows(feas)
    (a.out / "checks.json").write_text(json.dumps(dict(selections_identical=sel_same, counts_identical=cnt_same,
                                                       scorable_identical=scor_same, units=len(picks),
                                                       retained_by_fold=dict(Counter(f for f, _, _ in retained))), indent=2) + "\n")
    if not (sel_same and cnt_same and scor_same):
        fail("D replay differs from C archive; see checks.json")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
