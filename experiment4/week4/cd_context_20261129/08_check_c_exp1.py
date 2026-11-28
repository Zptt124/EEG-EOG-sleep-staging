"""D：复核 C 的实验一（K5 − K1）。

1. 跨机重跑：名单 1、第 1 次重复的 K5、K10、K5+B × add_people/random 共 6 个单元，
   用 C 的 train_bigru.py 中的 fit/infer、C 的输入 physical_context.npz，逐段与 C 归档的 pred 比对（验收 ≥ 99.9%）。
2. 独立重算：只读 C 归档的逐评估者混淆计数，不调用 C 的 analysis.py，自写重抽实现：
   名单等权；10,000 次；共用名单、两个子队列内评估者（各 8 人有放回）、同一选块重复；P/R 等权的 K5 − K1。

用法：python 08_check_c_exp1.py --c-src <context_readout_20261129 目录> --input physical_context.npz \
        --c-results <C 的单元结果目录> --out <输出目录>
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

UNITS = [(1, 1, s, k, b) for k, b in ((5, False), (10, False), (5, True)) for s in ("add_people", "random")]


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def tag(r, d, s, k, seed=0, bal=False):
    return f"{r:02d}_{d:02d}_{s}_k{k}_seed{seed}{'_balanced' if bal else ''}"


def macro(conf):
    den = conf.sum(-2) + conf.sum(-1)
    num = 200.0 * np.diagonal(conf, axis1=-2, axis2=-1)
    return np.divide(num, den, out=np.zeros(den.shape), where=den > 0).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    for n in ("c-src", "input", "c-results", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    ap.add_argument("--skip-replay", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.c_src))
    import torch
    import train_bigru as tb
    z = np.load(a.input, allow_pickle=False)
    data = {k: z[k] for k in z.files if k != "old_eval_windows"}
    radius = tb.validate(data)
    report = {}

    if not a.skip_replay:
        dev = tb.configure("cuda")
        keys = list(zip(data["roster"].tolist(), data["draw"].tolist(), [str(s) for s in data["strategy"]]))
        rows = []
        for r, d, s, k, bal in UNITS:
            ref_path = a.c_results / f"{tag(r, d, s, k, 0, bal)}.npz"
            if not ref_path.exists():
                print(f"skip {ref_path.name}: not yet released by C", flush=True)
                continue
            ref = np.load(ref_path)["pred"]
            i = keys.index((r, d, s))
            sup = data["unit_mask"][i][data["unit_mask"][i] >= 0]
            span = slice(radius - k, radius + k + 1)
            model, mean, std = tb.fit(data["x"][data["train_windows"][sup, span]], data["train_y"][sup], 0, bal, dev)
            pred, _ = tb.infer(model, mean, std, data["x"], data["eval_windows"][:, span], dev)
            dis = int((pred != ref).sum())
            mf = lambda p: float(macro(np.bincount(data["truth"] * 5 + p, minlength=25).reshape(5, 5)))
            rows.append(dict(unit=tag(r, d, s, k, 0, bal), eval_centers=len(ref), disagreements=dis,
                             agreement=1 - dis / len(ref), macro_f1_diff_pp=mf(pred) - mf(ref),
                             passes=bool(1 - dis / len(ref) >= 0.999)))
            print(json.dumps(rows[-1]), flush=True)
        report["replay"] = dict(units=rows, all_pass=all(r["passes"] for r in rows), torch=str(torch.__version__),
                                environment=platform.platform())

    # ---- 独立重算 K5 − K1 ----
    rosters = sorted(set(data["roster"].tolist()))
    conf = np.empty((len(rosters), 2, 5, 2, 16, 5, 5))
    for ri, r in enumerate(rosters):
        for pi, s in enumerate(("add_people", "random")):
            for d in range(1, 6):
                for hi, k in enumerate((1, 5)):
                    p = a.c_results / f"{tag(r, d, s, k)}.npz"
                    if not p.exists():
                        fail(f"missing C result {p.name}")
                    conf[ri, pi, d - 1, hi] = np.load(p)["counts"]
    base = macro(conf.sum(-3))                       # r, policy, draw, head
    eff = (base[..., 1] - base[..., 0]).mean((1, 2))  # per roster
    point = float(eff.mean())
    rng = np.random.default_rng(20261129)
    flat = np.moveaxis(conf, -3, 0).reshape(16, -1)
    n, B = len(rosters), 10000
    boot = np.empty(B)
    for b in range(B):
        w = np.concatenate([np.bincount(rng.integers(8, size=8), minlength=8) for _ in range(2)]).astype(float)
        sc = macro((w @ flat).reshape(n, 2, 5, 2, 5, 5))
        rs = rng.integers(n, size=n)
        dr = rng.integers(5, size=(n, 5))
        d = sc[rs[:, None], :, dr, 1] - sc[rs[:, None], :, dr, 0]   # n,5,policy
        boot[b] = d.mean()
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    report["independent_recompute"] = dict(contrast="K5 - K1 (P/R equal weight)", mean_pp=point, ci95_pp=ci,
                                           head_means_pp=base.mean((0, 1, 2)).tolist(),
                                           policy_effects_pp=(base[..., 1] - base[..., 0]).mean((0, 2)).tolist(),
                                           repetitions=B, seed=20261129)
    print(f"INDEPENDENT K5-K1: {point:+.3f} pp [{ci[0]:+.3f}, {ci[1]:+.3f}]", flush=True)
    (a.out / "D_check_c_exp1.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
