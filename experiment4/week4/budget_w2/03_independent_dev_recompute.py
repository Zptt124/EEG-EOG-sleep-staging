"""D 第 8 天：在开发集（pool36 / dev8）上独立复算三组 Macro-F1。

- 每个头（deepen / add_people / other_night）都由 D 自己在共享缓存上重新推理，不读别人的分数；
- Macro-F1 用 D 自己的混淆矩阵实现（不调用 sklearn）；
- 评分样本 = 该名单 evaluation_subjects.csv 里受试者在缓存中的全部行（与第一周 trainer 规则相同）；
- 与各成员自报分数逐项对比；
- 核对"新人新夜 − 原人原夜 = 换夜收益 + 新人较另一夜"，并与第一周 9 名单参考值对照（pool36 ≈ +1.69，dev8 ≈ +1.47）。
缺少的头（例如 A 还没交的名单）会跳过并在输出中标为 missing。只做开发评分，不碰预留 8 人和测试集。

用法：python 03_independent_dev_recompute.py --cache <共享缓存> --rosters <C week1 rosters 目录> \
        --old-c <C 旧头根目录> --old-d <D 第一周 budget_runs> --new <名单号=目录> ... \
        --c-week1-paired <C paired_macro_f1.csv> --out <输出目录>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROSTERS9 = ["01", "02", "03", "04", "06", "07", "08", "09", "10"]
ROSTERS10 = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
REF9 = {"pool36": 1.69, "dev8": 1.47}


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


def macro_f1(y, p, k=5):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def main():
    ap = argparse.ArgumentParser()
    for n in ("cache", "rosters", "old-c", "old-d", "c-week1-paired", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    ap.add_argument("--new", nargs="*", default=[], help="roster=dir with other_night head")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    new_dirs = dict(x.split("=", 1) for x in a.new)

    idx = rd(a.cache / "index.csv")
    subj = np.array([r["subject_id"] for r in idx])
    y_all = np.array([int(r["y_true"]) for r in idx])
    x_all = np.concatenate([np.load(a.cache / f"{m}_std.npy") for m in ("eeg", "eog")], axis=1).astype(np.float32)

    def head_dir(ro, arm):
        if arm == "other_night":
            d = new_dirs.get(ro)
            return Path(d) if d else None
        return (a.old_c if int(ro) <= 5 else a.old_d) / ro / "paired" / arm

    def run(d, ids):
        nz = np.load(d / "normalization.npz")
        m = torch.nn.Linear(x_all.shape[1], 5)
        m.load_state_dict(torch.load(d / "head.pt", map_location="cpu", weights_only=True))
        with torch.inference_mode():
            logits = m(torch.from_numpy((x_all[ids] - nz["mean"]) / nz["std"]))
        return logits.argmax(1).numpy()

    cw = {(r["roster"], r["cohort"]): r for r in rd(a.c_week1_paired)}
    rows, checks = [], []
    for ro in ROSTERS10:
        ev = rd(a.rosters / ro / "evaluation_subjects.csv")
        for cohort in ("pool36", "dev8"):
            members = {e["subject_id"] for e in ev if e["cohort"] == cohort}
            ids = np.flatnonzero(np.isin(subj, list(members)))
            if len(set(subj[ids])) != len(members):
                fail(f"{ro} {cohort}: some evaluation subjects missing from cache")
            rec = dict(roster=ro, cohort=cohort, n_samples=len(ids))
            for arm in ("deepen", "add_people", "other_night"):
                d = head_dir(ro, arm)
                if d is None or not (d / "head.pt").exists():
                    rec[arm] = ""
                    continue
                f1 = 100 * macro_f1(y_all[ids], run(d, ids))
                rec[arm] = f"{f1:.4f}"
                # 与自报分数对比
                if arm == "other_night":
                    rep = 100 * json.loads((d / "metrics_dev.json").read_text())[cohort]
                elif int(ro) <= 5:
                    rep = float(cw[(ro, cohort)][f"{arm}_macro_f1_pct"])
                else:
                    rep = 100 * json.loads((d.parent / "summary.json").read_text())[cohort]["macro_f1"][arm]
                checks.append(dict(roster=ro, cohort=cohort, arm=arm, d_recomputed=f"{f1:.6f}",
                                   reported=f"{rep:.6f}", abs_diff=f"{abs(f1 - rep):.2e}"))
                if abs(f1 - rep) > 1e-3:
                    fail(f"{ro} {cohort} {arm}: D {f1:.4f} vs reported {rep:.4f}")
            dp, ad, on = (float(rec[k]) if rec[k] else None for k in ("deepen", "add_people", "other_night"))
            rec["new_people_minus_same_night_pp"] = f"{ad - dp:+.4f}"
            if on is not None:
                g1, g2 = on - dp, ad - on
                rec["night_gain_pp"] = f"{g1:+.4f}"
                rec["new_people_minus_other_night_pp"] = f"{g2:+.4f}"
                if abs((g1 + g2) - (ad - dp)) > 1e-9:
                    fail(f"{ro} {cohort}: decomposition does not add up")
            else:
                rec["night_gain_pp"] = rec["new_people_minus_other_night_pp"] = "missing"
            rows.append(rec)
            print(rec, flush=True)
    wr(a.out / "dev_recompute_D.csv", rows)
    wr(a.out / "dev_recompute_vs_reported_D.csv", checks)

    summ = []
    for cohort in ("pool36", "dev8"):
        sub = {r["roster"]: r for r in rows if r["cohort"] == cohort}
        t1_10 = [float(sub[r]["new_people_minus_same_night_pp"]) for r in ROSTERS10]
        t1_9 = [float(sub[r]["new_people_minus_same_night_pp"]) for r in ROSTERS9]
        have = [r for r in ROSTERS9 if sub[r]["night_gain_pp"] != "missing"]
        g1 = [float(sub[r]["night_gain_pp"]) for r in have]
        g2 = [float(sub[r]["new_people_minus_other_night_pp"]) for r in have]
        summ.append(dict(cohort=cohort,
                         table1_10roster_mean=f"{np.mean(t1_10):+.3f}", table1_10_positive=f"{sum(v > 0 for v in t1_10)}/10",
                         table1_9roster_mean=f"{np.mean(t1_9):+.3f}", ref_9roster=f"{REF9[cohort]:+.2f}",
                         split_rosters_available=";".join(have) or "none",
                         night_gain_mean=f"{np.mean(g1):+.3f}" if g1 else "",
                         night_gain_positive=f"{sum(v > 0 for v in g1)}/{len(g1)}",
                         new_people_minus_other_night_mean=f"{np.mean(g2):+.3f}" if g2 else "",
                         new_people_minus_other_night_positive=f"{sum(v > 0 for v in g2)}/{len(g2)}"))
        if abs(np.mean(t1_9) - REF9[cohort]) > 0.01:
            print(f"WARNING: 9-roster mean {np.mean(t1_9):.3f} differs from reference {REF9[cohort]}", flush=True)
    wr(a.out / "dev_recompute_summary_D.csv", summ)
    for s in summ:
        print(s, flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
