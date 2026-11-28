"""D 第 9 天：在预留 8 人上独立重算表一、表二（不读 B/A 的分数和预测，自己推理全部 29 个头）。

- 样本：共享缓存中 role/split 为 held-out 的 8 人全部行；
- Macro-F1：D 自己的（加权）混淆矩阵实现；
- 名单内差值 → 按名单等权平均；
- 区间：1000 次 Poisson(1) 受试者权重，同一受试者在所有名单、所有策略共用权重（与预定规则一致；随机数实现与 B 不同，所以区间端点只要求接近）；
- 逐人剔除（LOSO）范围；
- 与 B 的 table1.csv / table2.csv 逐名单对比点估计，并对比 B 的预测表（y_pred 逐段一致）。

用法：python 04_independent_heldout_tables.py --cache <共享缓存> --head <名单:策略=目录> ... \
        --b-table1 <csv> --b-table2 <csv> --b-pred-dir <B heldout8 预测目录，可选> --out <输出目录>
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HELD = ["SC414", "SC416", "SC431", "SC450", "SC457", "SC463", "SC467", "SC477"]
R10 = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
R9 = [r for r in R10 if r != "05"]


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


def wf1(y, p, w):
    cm = np.zeros((5, 5))
    np.add.at(cm, (y, p), w)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--head", nargs="+", required=True)
    ap.add_argument("--b-table1", type=Path, required=True)
    ap.add_argument("--b-table2", type=Path, required=True)
    ap.add_argument("--b-pred-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    idx = rd(a.cache / "index.csv")
    subj = np.array([r["subject_id"] for r in idx])
    ids = np.flatnonzero(np.isin(subj, HELD))
    if sorted(set(subj[ids])) != HELD:
        fail("held-out subjects missing from cache")
    roles = {idx[i]["role"] for i in ids}
    if roles != {"heldout_val"}:
        fail(f"unexpected role for held-out rows: {roles}")
    y = np.array([int(idx[i]["y_true"]) for i in ids])
    sid = [idx[i]["sample_id"] for i in ids]
    s_of = subj[ids]
    x = np.concatenate([np.load(a.cache / f"{m}_std.npy", mmap_mode="r")[ids] for m in ("eeg", "eog")], axis=1).astype(np.float32)
    print(f"held-out rows {len(ids)}", flush=True)

    pred = {}
    for spec in a.head:
        key, d = spec.split("=", 1)
        ro, arm = key.split(":")
        d = Path(d)
        nz = np.load(d / "normalization.npz")
        m = torch.nn.Linear(x.shape[1], 5)
        m.load_state_dict(torch.load(d / "head.pt", map_location="cpu", weights_only=True))
        with torch.inference_mode():
            pred[ro, arm] = m(torch.from_numpy((x - nz["mean"]) / nz["std"])).argmax(1).numpy()
    need = [(r, a_) for r in R10 for a_ in ("deepen", "add_people")] + [(r, "other_night") for r in R9]
    miss = [k for k in need if k not in pred]
    if miss:
        fail(f"missing heads: {miss}")
    print(f"heads {len(pred)}", flush=True)

    # 与 B 的预测逐段比对
    pred_check = []
    if a.b_pred_dir:
        for (ro, arm), p in sorted(pred.items()):
            f = a.b_pred_dir / f"heldout8_{ro}_{arm}.csv"
            b = rd(f)
            if [r["sample_id"] for r in b] != sid:
                fail(f"sample order differs in {f.name}")
            bp = np.array([int(r["y_pred"]) for r in b])
            pred_check.append(dict(roster=ro, arm=arm, n=len(b), argmax_equal=int((bp == p).sum()),
                                   all_equal=bool((bp == p).all())))
        wr(a.out / "heldout_pred_vs_B.csv", pred_check)

    def scores(w):
        return {k: 100 * wf1(y, p, w) for k, p in pred.items()}

    contrasts = {
        "table1": [("add_minus_deepen", "add_people", "deepen", R10)],
        "table2": [("other_minus_deepen", "other_night", "deepen", R9),
                   ("add_minus_other", "add_people", "other_night", R9)],
    }

    def means(sc):
        out = {}
        for t, cs in contrasts.items():
            for name, hi, lo, rs in cs:
                out[name] = (float(np.mean([sc[r, hi] - sc[r, lo] for r in rs])), {r: sc[r, hi] - sc[r, lo] for r in rs})
        return out

    base_w = np.ones(len(ids))
    sc0 = scores(base_w)
    m0 = means(sc0)
    rng = np.random.default_rng(0)
    boots = {k: [] for k in m0}
    subj_index = {s: np.flatnonzero(s_of == s) for s in HELD}
    for _ in range(1000):
        sw = rng.poisson(1.0, size=len(HELD)).astype(float)
        w = np.zeros(len(ids))
        for s, v in zip(HELD, sw):
            w[subj_index[s]] = v
        if w.sum() == 0:
            continue
        mb = means(scores(w))
        for k in boots:
            boots[k].append(mb[k][0])
    loso = {k: [] for k in m0}
    for s in HELD:
        w = np.where(s_of == s, 0.0, 1.0)
        ml = means(scores(w))
        for k in loso:
            loso[k].append(ml[k][0])

    b1 = {r["roster"]: r for r in rd(a.b_table1)}
    b2 = {r["roster"]: r for r in rd(a.b_table2)}
    bcol = {"add_minus_deepen": (b1, "add_minus_deepen_pp"), "other_minus_deepen": (b2, "other_minus_deepen_pp"),
            "add_minus_other": (b2, "add_minus_other_pp")}
    per, summ = [], []
    for k, (mean, perro) in m0.items():
        bt, col = bcol[k]
        for r, v in perro.items():
            bv = float(bt[r][col])
            per.append(dict(contrast=k, roster=r, d_pp=f"{v:.4f}", b_pp=f"{bv:.4f}", abs_diff=f"{abs(v - bv):.4f}"))
            if abs(v - bv) > 0.01:
                fail(f"{k} roster {r}: D {v:.4f} vs B {bv:.4f}")
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        bm = bt["mean"]
        vals = list(perro.values())
        summ.append(dict(contrast=k, n_rosters=len(vals), d_mean_pp=f"{mean:+.4f}", d_ci95=f"[{lo:+.3f}, {hi:+.3f}]",
                         b_mean_pp=bm[col], b_ci95=f"[{float(bm[col.replace('_pp', '_lo')]):+.3f}, {float(bm[col.replace('_pp', '_hi')]):+.3f}]",
                         positive=f"{sum(v > 0 for v in vals)}/{len(vals)}",
                         loso_range=f"[{min(loso[k]):+.3f}, {max(loso[k]):+.3f}]"))
    wr(a.out / "heldout_per_roster_D.csv", per)
    wr(a.out / "heldout_summary_D.csv", summ)
    wr(a.out / "heldout_macro_f1_D.csv", [dict(roster=r, arm=arm_, macro_f1_pct=f"{v:.4f}") for (r, arm_), v in sorted(sc0.items())])
    for s in summ:
        print(s, flush=True)
    if pred_check:
        print(f"predictions identical to B: {sum(c['all_equal'] for c in pred_check)}/{len(pred_check)}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
