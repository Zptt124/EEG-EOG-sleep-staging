"""实验A 下一轮第3周：450 次主要拟合复跑核对（对照 C 的归档）。

从干净目录复跑 Sleep-EDF 的 450 次主要拟合后，逐拟合核对样本键、混淆计数和
Macro-F1，复算换夜比较与新人比较，输出 450 行核对表、最大差异与异常原因。

用法：
python scripts/51_a_w3_replication_verify.py \
  --mine D:/EEG-EOG-sleep-staging/runs/a_w3_replication_20261128 \
  --c-src experiment3/week3_budget_replication_20261128/results \
  --out reports/w3_replication_a
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

N_ROSTERS = 30
N_DRAWS = 5
N_ARMS = 3
ARMS = ("deepen", "other_night", "add_people")
ARM_CN = ("原人原夜", "原人另一夜", "新人新夜")


def rd(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
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


def load_mine(mine):
    """Assemble [3,30,5,3,16,25] counts from per-roster npz files."""
    counts = np.zeros((3, N_ROSTERS, N_DRAWS, N_ARMS, 16, 25), dtype=np.int64)
    scores = []
    for roster in range(1, N_ROSTERS + 1):
        path = mine / f"roster_{roster:02d}.npz"
        if not path.exists():
            raise RuntimeError(f"Missing roster archive: {path}")
        with np.load(path, allow_pickle=False) as z:
            counts[:, roster - 1] = z["counts"]
            scores.extend(json.loads(str(z["scores"])))
    return counts, scores


def sample_keys_check(mine, c_src, report):
    """Byte-level comparison of the frozen plan outputs."""
    checks = {}
    for name in ("blocks.csv", "candidate_blocks.csv"):
        mine_b = (mine / name).read_bytes()
        c_b = (c_src / name).read_bytes()
        checks[f"{name}_byte_identical"] = mine_b == c_b
        checks[f"{name}_rows"] = len(rd(mine / name))
    mine_plan = json.loads((mine / "plan.json").read_text(encoding="utf-8"))
    c_plan = json.loads((c_src / "plan.json").read_text(encoding="utf-8"))
    diff_fields, only_mine, only_c = [], [], []
    for key in sorted(set(mine_plan) | set(c_plan)):
        if key not in mine_plan:
            only_c.append(key)
        elif key not in c_plan:
            only_mine.append(key)
        elif mine_plan[key] != c_plan[key]:
            diff_fields.append(key)
    checks["plan_fields_differ"] = sorted(diff_fields)
    checks["plan_fields_only_mine"] = only_mine
    checks["plan_fields_only_c"] = only_c
    checks["mine_python"] = mine_plan.get("python")
    checks["c_python"] = c_plan.get("python")
    report["sample_keys"] = checks
    return checks


def per_fit_table(counts_m, counts_c, mine_scores, c_scores):
    """Build the per-fit verification rows."""
    c_by_key = {(int(r["roster"]), int(r["draw"]), r["arm"], int(r["seed"])): r
                for r in c_scores}
    m_by_key = {(int(r["roster"]), int(r["draw"]), r["arm"], int(r["seed"])): r
                for r in mine_scores}
    rows = []
    worst = []
    for roster in range(N_ROSTERS):
        seeds = (0, 1, 2) if roster < 3 else (0,)
        for seed in seeds:
            for draw in range(N_DRAWS):
                for arm_ix, arm in enumerate(ARMS):
                    cm = counts_m[seed, roster, draw, arm_ix]
                    cc = counts_c[seed, roster, draw, arm_ix]
                    diff = cm - cc
                    n_cells = int((diff != 0).sum())
                    max_cell = int(np.abs(diff).max()) if n_cells else 0
                    total_diff = int(np.abs(diff).sum())
                    key = (roster + 1, draw + 1, arm, seed)
                    m, c = m_by_key[key], c_by_key[key]
                    row = dict(
                        roster=roster + 1, draw=draw + 1, arm=arm, arm_cn=ARM_CN[arm_ix],
                        seed=seed, primary="yes" if seed == 0 else "no",
                        n_cells_diff=n_cells, max_cell_diff=max_cell,
                        total_samples_diff=total_diff,
                        macro_f1_mine=m["macro_f1"], macro_f1_c=c["macro_f1"],
                        dev8_mine=m["dev8"], dev8_c=c["dev8"],
                        pretrain_eval8_mine=m["pretrain_eval8"], pretrain_eval8_c=c["pretrain_eval8"],
                        updates_mine=m["updates"], updates_c=c["updates"],
                        added_stages_mine=m["added_stages"], added_stages_c=c["added_stages"],
                    )
                    rows.append(row)
                    if n_cells:
                        worst.append(row)
    worst.sort(key=lambda r: (-r["total_samples_diff"], -r["n_cells_diff"]))
    return rows, worst


def recompute_summary(counts_m, mine, c_src, tmp):
    """Import C's run.py, summarize A's counts with the identical code path."""
    spec = importlib.util.spec_from_file_location(
        "c_run", Path("experiment3/week3_budget_replication_20261128/run.py").resolve())
    c_run = importlib.util.module_from_spec(spec)
    sys.modules["c_run"] = c_run
    spec.loader.exec_module(c_run)
    lock = json.loads((mine / "plan.json").read_text(encoding="utf-8"))
    tmp.mkdir(parents=True, exist_ok=True)
    c_run.summarize(counts_m, lock, tmp)
    mine_summary = json.loads((tmp / "summary.json").read_text(encoding="utf-8"))
    c_summary = json.loads((c_src / "summary.json").read_text(encoding="utf-8"))
    return mine_summary, c_summary


def compare_summary(mine_s, c_s):
    out = {}
    for cohort in ("combined", "dev8", "pretrain_eval8"):
        out[cohort] = {}
        for name in ("other_minus_deepen", "new_minus_other", "new_minus_deepen"):
            m, c = mine_s[cohort][name], c_s[cohort][name]
            out[cohort][name] = dict(
                mean_pp_mine=round(m["mean_pp"], 6), mean_pp_c=round(c["mean_pp"], 6),
                mean_delta_pp=round(m["mean_pp"] - c["mean_pp"], 8),
                interval_mine=[round(v, 4) for v in m["interval_95_pp"]],
                interval_c=[round(v, 4) for v in c["interval_95_pp"]],
                positive_rosters=(m["positive_rosters"], c["positive_rosters"]),
                roster_deltas_max_abs_diff_pp=round(
                    float(np.max(np.abs(np.array(m["roster_deltas_pp"]) - np.array(c["roster_deltas_pp"])))), 8),
            )
    out["seed_check"] = dict(
        mine=mine_s["seed_check"], c=c_s["seed_check"])
    out["decision_mine"] = mine_s["decision"]
    out["decision_c"] = c_s["decision"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", type=Path, required=True)
    ap.add_argument("--c-src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}

    report["sample_keys"] = sample_keys_check(a.mine, a.c_src, report)
    counts_m, mine_scores = load_mine(a.mine)
    counts_c = np.load(a.c_src / "confusion_counts.npz")["counts"]
    if counts_c.shape != counts_m.shape:
        raise RuntimeError(f"Counts shape mismatch: mine {counts_m.shape} vs C {counts_c.shape}")

    c_scores = rd(a.c_src / "scores.csv")
    rows, worst = per_fit_table(counts_m, counts_c, mine_scores, c_scores)
    wr(a.out / "verification_540.csv", rows)
    wr(a.out / "verification_450.csv", [r for r in rows if r["primary"] == "yes"])
    exact = sum(1 for r in rows if r["n_cells_diff"] == 0)
    report["fits"] = dict(total=len(rows), primary=sum(1 for r in rows if r["primary"] == "yes"),
                          exact_count_matches=exact, differing=len(rows) - exact)
    report["worst_fits"] = worst[:20]

    mine_summary, c_summary = recompute_summary(counts_m, a.mine, a.c_src, a.out / "_tmp_summary")
    report["paired"] = compare_summary(mine_summary, c_summary)
    report["macro_selfcheck"] = dict(
        mine_max_abs_pp=float(np.max(np.abs(
            np.array([float(r["macro_f1_mine"]) for r in rows]) -
            np.array([float(r["macro_f1_c"]) for r in rows])))),
    )
    (a.out / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("fits", "sample_keys")}, ensure_ascii=False, indent=2))
    print("paired:", json.dumps(report["paired"], ensure_ascii=False, indent=2))
    print("top worst fits:", json.dumps(report["worst_fits"][:5], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
