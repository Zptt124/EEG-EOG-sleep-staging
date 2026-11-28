"""D：实验二出分。调用 C 的 analysis.py（evaluate）计算主比较和解释性比较。

主比较（预先登记）：BiGRU K1 的 coverage(V) − random(R)。
解释性：BiGRU V − add_people(P)、disagreement(U) − R、P − R、coverage_det − R；线性头同样四项。
coverage_det 每份名单只有 1 次拟合：在 10 个重复位置上放同一结果，名单内重抽不再引入额外波动（不当作 10 次独立重复）。
另报各臂均值、N1 F1、来源构成均值和实际有效标签数。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f1(conf):
    den = conf.sum(-2) + conf.sum(-1)
    return np.divide(200 * np.diagonal(conf, axis1=-2, axis2=-1), den, out=np.zeros(den.shape), where=den > 0)


def main():
    ap = argparse.ArgumentParser()
    for n in ("bigru-dir", "linear-dir", "anchor", "input", "composition", "c-src", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.c_src))
    from analysis import evaluate
    members = defaultdict(set)
    for r in rd(a.anchor):
        members[int(r["roster"])].add(r["subject_id"])
    rosters = sorted(members)
    anchor_members = [sorted(members[r]) for r in rosters]
    z = np.load(a.input, allow_pickle=False)
    eff = {(int(r), int(d), str(s)): int(e) for r, d, s, e in
           zip(z["roster"], z["draw"], z["strategy"], z["effective_additions"])}

    def load(model, strat, ro, d):
        if model == "bigru":
            p = a.bigru_dir / f"{ro:02d}_{d:02d}_{strat}_k1_seed0.npz"
        else:
            p = a.linear_dir / f"{ro:02d}_{d:02d}_{strat}_linear.npz"
        if not p.exists():
            fail(f"missing result {p}")
        return np.load(p)["counts"]

    def arm(model, strat):
        draws = [1] * 10 if strat == "coverage_det" else range(1, 11)
        return np.stack([np.stack([load(model, strat, ro, d) for d in draws]) for ro in rosters])

    cache, report = {}, dict(primary=None, explanatory={}, arms={})
    for model in ("bigru", "linear"):
        for s in ("random", "add_people", "coverage", "disagreement", "coverage_det"):
            cache[model, s] = arm(model, s)
            c = cache[model, s]
            pooled = c.sum(2)  # roster, draw, 5, 5
            report["arms"][f"{model}/{s}"] = dict(
                macro_f1_pp=float(f1(pooled).mean(-1).mean()),
                per_stage_f1_pp=dict(zip(("W", "N1", "N2", "N3", "REM"), f1(pooled).mean((0, 1)).round(3).tolist())),
                mean_effective_additions=float(np.mean([eff[(ro, d, s)] for ro in rosters
                                                        for d in ([1] if s == "coverage_det" else range(1, 11))])))
    pairs = [("random", "coverage"), ("add_people", "coverage"), ("random", "disagreement"),
             ("random", "add_people"), ("random", "coverage_det")]
    for model in ("bigru", "linear"):
        for base, new in pairs:
            conf = np.stack([cache[model, base], cache[model, new]], axis=2)[:, None]  # r,1,draw,arm,16,5,5
            res = evaluate(conf, anchor_members, paired=False)
            key = f"{model}: {new} - {base}"
            if model == "bigru" and (base, new) == ("random", "coverage"):
                report["primary"] = dict(contrast=key, **res)
            else:
                report["explanatory"][key] = {k: res[k] for k in ("mean_pp", "ci95_pp", "decision",
                                                                  "sensitivity_multi_membership_ci95_pp",
                                                                  "leave_one_range_pp", "subcohort_effects_pp")}
    comp = defaultdict(lambda: defaultdict(list))
    for r in rd(a.composition):
        for k in ("same_night_blocks", "other_night_blocks", "new_person_blocks", "subjects", "nights"):
            comp[r["strategy"]][k].append(float(r[k]))
    report["composition_means"] = {s: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for s, d in comp.items()}
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    p = report["primary"]
    print(f"PRIMARY {p['contrast']}: {p['mean_pp']:+.3f} pp, 95% CI [{p['ci95_pp'][0]:+.3f}, {p['ci95_pp'][1]:+.3f}] "
          f"-> {p['decision']}", flush=True)
    for k, v in report["explanatory"].items():
        print(f"  {k}: {v['mean_pp']:+.3f} [{v['ci95_pp'][0]:+.3f}, {v['ci95_pp'][1]:+.3f}] {v['decision']}")


if __name__ == "__main__":
    main()
