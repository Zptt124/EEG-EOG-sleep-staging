"""D：(1) 用 dev8 重放 06–10 的 10 个第一周旧头（deepen / add_people），与存档预测逐行比对；
(2) 列出旧头清单（SHA、Release 内路径），交给 B；
(3) 汇总 04、06 三组的开发分数和两项拆分差（仅开发评分，非正式结论）。

用法：python 02_replay_and_dev_table.py --cache <D 全量共享缓存目录> --old-runs <D 第一周 budget_runs> \
        --new-runs <第二周 runs> --c-week1-paired <C paired_macro_f1.csv> --out <输出目录>
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

PROBS = ("p_W", "p_N1", "p_N2", "p_N3", "p_REM")
RELEASE = "d-budget-week1-results-v01-20261126"
ZIP = "d_budget_week1_results_v01.zip"


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


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for n in ("cache", "old-runs", "new-runs", "c-week1-paired", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    idx = rd(a.cache / "index.csv")
    pos = {r["sample_id"]: i for i, r in enumerate(idx)}
    x_all = np.concatenate([np.load(a.cache / f"{m}_std.npy", mmap_mode="r") for m in ("eeg", "eog")], axis=1)

    def replay(run_dir, cohort):
        saved = rd(run_dir / f"predictions_{cohort}.csv")
        ids = [pos[r["sample_id"]] for r in saved]
        x = np.asarray(x_all[ids], dtype=np.float32)
        nz = np.load(run_dir / "normalization.npz")
        model = torch.nn.Linear(x.shape[1], 5)
        model.load_state_dict(torch.load(run_dir / "head.pt", map_location="cpu", weights_only=True))
        with torch.inference_mode():
            p = model(torch.from_numpy((x - nz["mean"]) / nz["std"])).softmax(1).numpy()
        sp = np.array([[float(r[k]) for k in PROBS] for r in saved])
        y = np.array([int(r["y_true"]) for r in saved])
        yp = np.array([int(r["y_pred"]) for r in saved])
        if not all(int(idx[i]["y_true"]) == int(r["y_true"]) for i, r in zip(ids, saved)):
            fail(f"truth mismatch in {run_dir}")
        f1 = f1_score(y, p.argmax(1), labels=list(range(5)), average="macro", zero_division=0)
        return dict(n=len(saved), argmax_exact=bool((p.argmax(1) == yp).all()),
                    max_abs_prob_diff=float(np.abs(p - sp).max()), macro_f1=float(f1))

    heads = []
    for ro in ("06", "07", "08", "09", "10"):
        for arm in ("deepen", "add_people"):
            d = a.old_runs / ro / "paired" / arm
            r = replay(d, "dev8")
            if not r["argmax_exact"] or r["max_abs_prob_diff"] > 1e-5:
                fail(f"replay mismatch {ro} {arm}: {r}")
            heads.append(dict(roster=ro, arm=arm, owner="D", release=RELEASE, zip=ZIP,
                              zip_path=f"d_budget_week1_results_v01/{ro}/paired/{arm}/",
                              head_sha256=sha(d / "head.pt"), normalization_sha256=sha(d / "normalization.npz"),
                              dev8_samples=r["n"], dev8_argmax_exact=r["argmax_exact"],
                              dev8_max_abs_prob_diff=f"{r['max_abs_prob_diff']:.3e}"))
            print(f"replay {ro} {arm}: ok (max diff {r['max_abs_prob_diff']:.2e})", flush=True)
    wr(a.out / "old_heads_D_06_10.csv", heads)

    # 开发分数：04 的两组旧条件取 C 第一周表；06 取 D 第一周 summary.json；新组取本次拟合
    cw = {(r["roster"], r["cohort"]): r for r in rd(a.c_week1_paired)}
    table = []
    for ro in ("04", "06"):
        new = json.loads((a.new_runs / ro / "other_night" / "metrics_dev.json").read_text())
        for cohort in ("pool36", "dev8"):
            if ro == "06":
                s = json.loads((a.old_runs / ro / "paired" / "summary.json").read_text())[cohort]["macro_f1"]
                deep, add, src = 100 * s["deepen"], 100 * s["add_people"], "D week1 summary.json"
            else:
                c = cw[(ro, cohort)]
                deep, add, src = float(c["deepen_macro_f1_pct"]), float(c["add_people_macro_f1_pct"]), "C week1 paired_macro_f1.csv"
            oth = 100 * new[cohort]
            # 重算新组分数，防止 metrics 与预测不一致
            rr = replay(a.new_runs / ro / "other_night", cohort)
            if abs(100 * rr["macro_f1"] - oth) > 1e-6 or not rr["argmax_exact"]:
                fail(f"new run {ro} {cohort} metrics/prediction mismatch")
            table.append(dict(roster=ro, cohort=cohort, same_night_pct=f"{deep:.2f}", other_night_pct=f"{oth:.2f}",
                              new_people_pct=f"{add:.2f}", night_gain_pp=f"{oth - deep:+.2f}",
                              new_people_minus_other_night_pp=f"{add - oth:+.2f}",
                              new_people_minus_same_night_pp=f"{add - deep:+.2f}", old_arm_source=src))
    wr(a.out / "dev_three_arms_04_06_D.csv", table)
    for t in table:
        print(t, flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
