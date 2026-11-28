"""D：独立复算实验一（线性头）和实验二（BiGRU）的主比较（只用成员 C 归档的逐评估者混淆计数，不调用 C 的统计代码）。

counts 轴：名单, 重复, 条件(A,B,C | BiGRU 只有 A,B), 策略(新人, 混合随机), 评估者(16), 25 格(5×真值+预测)。
Δ = 新人 Macro-F1 − 混合随机 Macro-F1（16 位评估者合并，百分点）；名单内平均重复，名单等权。
区间：D 自己实现的三层重抽样（名单有放回、名单内重复有放回、两个 8 人队列内 multinomial(8) 评估者权重），
10,000 次，numpy default_rng(0)；所有比较共用同一组重抽样。随机数实现与 C 不同，区间端点只要求接近。
严格口径：scored_additions 在该比较涉及的全部单元都等于 600 的 (名单, 重复)。
另报：条件间实现的相关（重抽样分布中 ΔA 与 ΔB 的相关），用于对照开跑前的精度估计。

用法：python 05_independent_recompute.py --pkg <C 结果包解压目录> --out <目录>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

N_BOOT = 10000


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def macro(conf):
    m = conf.reshape(*conf.shape[:-1], 5, 5)
    den = m.sum(-1) + m.sum(-2)
    diag = np.diagonal(m, axis1=-2, axis2=-1)
    return np.divide(200.0 * diag, den, out=np.zeros(den.shape), where=den > 0).mean(-1)


def boot(counts, mask, stat, seed=0):
    """counts [R,5,...,16,25]; mask [R,5]; stat(f1 array [...,R,5,...]) -> [..., R, 5, K]"""
    keep = [j for j in range(counts.shape[0]) if mask[j].any()]
    sub = counts[keep]
    idx = [np.flatnonzero(mask[j]) for j in keep]
    Rk = len(keep)
    sizes = np.array([len(i) for i in idx])
    pad = np.stack([np.pad(i, (0, 5 - len(i))) for i in idx])
    cell = stat(macro(sub.sum(-2)))
    rm = np.stack([cell[i][idx[i]].mean(0) for i in range(Rk)])
    point = rm.mean(0)
    flat = sub.reshape(-1, 16, 25).transpose(1, 0, 2).reshape(16, -1)
    rng = np.random.default_rng(seed)
    out = []
    for lo in range(0, N_BOOT, 100):
        n = min(100, N_BOOT - lo)
        rs = rng.integers(Rk, size=(n, Rk))
        nr = sizes[rs][..., None]
        ds = pad[rs[..., None], (rng.random((n, Rk, 5)) * nr).astype(int)]
        valid = np.arange(5) < nr
        w = rng.multinomial(8, [1 / 8] * 8, size=(n, 2)).reshape(n, 16).astype(float)
        wc = (w @ flat).reshape(n, *sub.shape[:-2], 25)
        c = stat(macro(wc))
        v = c[np.arange(n)[:, None, None], rs[..., None], ds]
        out.append(((v * valid[..., None]).sum(2) / nr).mean(1))
    b = np.concatenate(out)
    return point, np.percentile(b, [2.5, 97.5], axis=0), b, Rk, int(sizes.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    L = np.load(a.pkg / "primary_linear/paired_confusions.npz")
    G = np.load(a.pkg / "bigru_ab/paired_confusions.npz")
    if list(L["rosters"]) != list(G["rosters"]):
        fail("roster order differs between linear and BiGRU archives")
    lc, gc = L["counts"], G["counts"]
    if lc.shape[:2] != (26, 5) or lc.shape[2:4] != (3, 2) or gc.shape[2:4] != (2, 2):
        fail(f"unexpected shapes {lc.shape} {gc.shape}")
    rows = []

    def lin_stat(f1):                     # f1 [...,R,5,3,2]
        d = f1[..., 0] - f1[..., 1]       # Δ per condition
        A, B, C = d[..., 0], d[..., 1], d[..., 2]
        return np.stack([A, B, C, B - A, B - C, C - A], -1)
    names = ["delta_A", "delta_B", "delta_C", "delta_B_minus_A", "delta_B_minus_C", "delta_C_minus_A"]
    for label, mask in (("fixed_600_attempts", np.ones((26, 5), bool)),
                        ("exact_600_all_groups", (L["scored_additions"] == 600).all(axis=(2, 3)))):
        pt, ci, b, Rk, npairs = boot(lc, mask, lin_stat)
        for k, n in enumerate(names):
            rows.append(dict(model="linear", analysis=label, contrast=n, rosters=Rk, draw_pairs=npairs,
                             mean_pp=round(float(pt[k]), 4), ci95_lo=round(float(ci[0, k]), 4), ci95_hi=round(float(ci[1, k]), 4)))
        if label == "fixed_600_attempts":
            rho = float(np.corrcoef(b[:, 0], b[:, 1])[0, 1])
            rows.append(dict(model="linear", analysis=label, contrast="realized_corr_deltaA_deltaB_bootstrap",
                             rosters=Rk, draw_pairs=npairs, mean_pp=round(rho, 4), ci95_lo="", ci95_hi=""))

    both = np.concatenate([gc[:, :, :, :, None], lc[:, :, :2, :, None]], axis=-1)  # [...,cond2,strat2,model2,16,25]? fix axes
    # 组合成 [R,5,2(cond),2(strat),2(model),16,25]
    both = np.stack([gc, lc[:, :, :2]], axis=4)

    def gru_stat(f1):                     # f1 [...,R,5,2,2,2] (cond, strat, model)
        d = f1[..., 0, :] - f1[..., 1, :]  # Δ per cond, model -> [...,R,5,2(cond),2(model)]
        eff = d[..., 1, :] - d[..., 0, :]  # ΔB−ΔA per model
        return np.stack([d[..., 0, 0], d[..., 1, 0], eff[..., 0], eff[..., 1], eff[..., 0] - eff[..., 1]], -1)
    gnames = ["bigru_delta_A", "bigru_delta_B", "bigru_delta_B_minus_A", "linear_delta_B_minus_A",
              "bigru_minus_linear_effect"]
    gmask_exact = (G["scored_additions"] == 600).all(axis=(2, 3))
    for label, mask in (("fixed_600_attempts", np.ones((26, 5), bool)), ("exact_600_AB_groups", gmask_exact)):
        pt, ci, b, Rk, npairs = boot(both, mask, gru_stat)
        for k, n in enumerate(gnames):
            rows.append(dict(model="bigru_vs_linear", analysis=label, contrast=n, rosters=Rk, draw_pairs=npairs,
                             mean_pp=round(float(pt[k]), 4), ci95_lo=round(float(ci[0, k]), 4), ci95_hi=round(float(ci[1, k]), 4)))
    with open(a.out / "independent_recompute_D.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # 对照 C 的汇总
    cs = json.loads((a.pkg / "primary_linear/summary.json").read_text())
    gs = json.loads((a.pkg / "bigru_ab/summary.json").read_text())
    comp = []
    for r in rows:
        if r["model"] == "linear" and r["analysis"] in cs and r["contrast"] in cs[r["analysis"]]["contrasts"]:
            c = cs[r["analysis"]]["contrasts"][r["contrast"]]
            comp.append(dict(source="linear", analysis=r["analysis"], contrast=r["contrast"], D_mean=r["mean_pp"],
                             C_mean=round(c["mean_pp"], 4), D_ci=f"[{r['ci95_lo']}, {r['ci95_hi']}]",
                             C_ci=f"[{c['ci95_pp'][0]:.4f}, {c['ci95_pp'][1]:.4f}]", mean_abs_diff=abs(r["mean_pp"] - c["mean_pp"])))
    with open(a.out / "compare_with_C.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0]))
        w.writeheader()
        w.writerows(comp)
    for r in rows:
        print(r, flush=True)
    print("max |mean diff| vs C (linear):", max(c["mean_abs_diff"] for c in comp))
    print("C BiGRU summary keys:", list(gs.keys())[:8])


if __name__ == "__main__":
    main()
