"""B 第 3 周：Ying 外部复验的独立设计复核、480 次拟合与配对汇总。

步骤
1. design：按冻结协议（种子 20261128）独立重写 4 折划分和配对抽块，输出 blocks_b.csv，
   与 C 归档 blocks.csv 逐字节比较；再用 B 自己的标签逐条审计块的约束。
2. fit：每折 8 名单 × 5 抽块 × 3 臂共 480 次拟合，线性头训练器用组内共用的
   experiment3/week1_budget_20261126/train.py::fit（seed 0、30 轮）。按评估者记混淆计数。
3. summary：B 自己实现的配对差、折等权平均、三层 bootstrap（折内重抽评估人、名单、抽块），
   并与 C 的 summary.json 及各折混淆计数比较。

用法：python -I scripts/56_b_w3_ying_confirm.py runs/b_w3_ying [--features features]
"""
import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT / "experiment3/week4_budget_external_ying/results"
SEED = 20261128
FOLDS, ROSTERS, DRAWS, B = 4, 8, 5, 60
ARMS = ("deepen", "other_night", "add_people")
STAGES = ("W", "N1", "N2", "N3", "REM")
MISSING = {("SLP004", 2), ("SLP020", 2)}


# ---------- 数据 ----------
VARIANT = "main"
DESIGN_SEED_OFFSET = 1


def relabel_scored_majority(experts):
    """敏感性：只在已打分专家中投票；至少 2 位打分才保留，平票取 Expert 1（若其打分）否则取 Expert 2。"""
    e = experts.astype(np.int64)
    out = np.full(e.shape[1], -1)
    for t in range(e.shape[1]):
        v = [x for x in e[:, t] if x >= 0]
        if len(v) < 2:
            continue
        vals, cnt = np.unique(v, return_counts=True)
        if cnt.max() >= 2:
            out[t] = vals[cnt.argmax()]
        else:
            out[t] = e[0, t] if e[0, t] >= 0 else e[1, t]
    return out


def load(feat_dir):
    recs, cursor = {}, 0
    xs, ys = [], []
    for i in range(1, 21):
        for d in (1, 2):
            if (f"SLP{i:03d}", d) in MISSING:
                continue
            key = f"SLP{i:03d}_Day{d}"
            with np.load(feat_dir / f"{key}.npz", allow_pickle=False) as r:
                y = r["y"].astype(np.int64)
                if VARIANT == "scored_majority":
                    y = relabel_scored_majority(r["experts"])
                x = r["x"] if "x" in r.files else np.zeros((len(y), 0), np.float32)
            assert x.shape[0] == len(y) and x.shape[1] in (0, 384)
            # 夜内 60 段不重叠块，按块中点归入三分位；块内全部可评分才是候选
            cand = [[], [], []]
            for s in range(0, len(y) - B + 1, B):
                mid2 = 2 * s + B - 1                       # 2 × 中点（以 epoch 为单位）
                third = min(2, 3 * mid2 // (2 * len(y)))
                if (y[s:s + B] >= 0).all():
                    cand[third].append(s)
            recs[key] = dict(offset=cursor, n=len(y), cand=cand)
            xs.append(x)
            ys.append(y)
            cursor += len(y)
    return recs, np.concatenate(xs), np.concatenate(ys)


def folds():
    s = [f"SLP{i:03d}" for i in range(1, 21)]
    random.Random(SEED).shuffle(s)
    return [sorted(s[k::FOLDS]) for k in range(FOLDS)]


# ---------- 设计 ----------
def design(recs, fold_list):
    rng = random.Random(SEED + DESIGN_SEED_OFFSET)
    nights = {}
    for key, r in recs.items():
        subj, day = key.split("_Day")
        nights.setdefault(subj, {})[int(day)] = (key, r)

    def ok(subj):
        if len(nights[subj]) != 2:
            return False
        if VARIANT == "no_eligibility":
            return True
        return all(sum(len(c) for c in r["cand"]) >= 6 and sum(1 for c in r["cand"] if c) >= 2
                   for _, r in nights[subj].values())

    rows, plans, eligible_log = [], {}, {}
    for f, ev in enumerate(fold_list):
        pool = sorted(s for s in nights if s not in ev and ok(s))
        eligible_log[f + 1] = pool
        for ro in range(ROSTERS):
            chosen = rng.sample(pool, 10)
            days = [1, 1, 1, 2, 2] if (f + ro) % 2 == 0 else [1, 1, 2, 2, 2]
            rng.shuffle(days)
            base, added = [], {(d, a): [] for d in range(DRAWS) for a in ARMS}
            for old, new, day in zip(chosen[:5], chosen[5:], days):
                src_key, src = nights[old][day]
                oth_key, oth = nights[old][3 - day]
                new_key, nw = nights[new][3 - day]
                pairs = [(k, s) for k in range(3) for s in src["cand"][k]]
                for _ in range(500):
                    start = rng.sample(pairs, 4)
                    rest = [[s for s in src["cand"][k] if (k, s) not in start] for k in range(3)]
                    common = [k for k in range(3) if rest[k] and oth["cand"][k] and nw["cand"][k]]
                    if len(common) >= 2:
                        break
                else:
                    raise RuntimeError("no matched thirds")
                for k, s in start:
                    base.append(src["offset"] + s)
                    rows.append((f + 1, ro + 1, 0, "base", old, src_key, s, k + 1))
                for k in rng.sample(common, 2):
                    for arm, opts, subj, key, rec in zip(
                            ARMS, (rest[k], oth["cand"][k], nw["cand"][k]), (old, old, new),
                            (src_key, oth_key, new_key), (src, oth, nw)):
                        for d in range(DRAWS):
                            s = rng.choice(opts)
                            added[d, arm].append(rec["offset"] + s)
                            rows.append((f + 1, ro + 1, d + 1, arm, subj, key, s, k + 1))
            plans[f, ro] = (base, added)
    return rows, plans, eligible_log


def write_blocks(path, rows):
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["fold", "roster", "draw", "arm", "subject", "record", "start", "third"])
    w.writerows(rows)
    data = buf.getvalue().encode()
    path.write_bytes(data)
    return data


def audit_blocks(rows, recs, y, fold_list):
    """不依赖设计代码，直接从块表与标签检查约束。"""
    issues = []
    by = {}
    for f, ro, d, arm, subj, key, s, k in rows:
        by.setdefault((f, ro), []).append((d, arm, subj, key, s, k))
        r = recs[key]
        if subj in fold_list[f - 1]:
            issues.append(f"eval subject in training {f},{ro},{subj}")
        if not key.startswith(subj):
            issues.append(f"record/subject mismatch {key}")
        if s % B or not (y[r["offset"] + s: r["offset"] + s + B] >= 0).all():
            issues.append(f"block not fully scored {key}:{s}")
        mid_third = min(2, 3 * (2 * s + B - 1) // (2 * r["n"])) + 1
        if mid_third != k:
            issues.append(f"third mismatch {key}:{s}")
    for (f, ro), items in by.items():
        base = [(subj, key, s) for d, arm, subj, key, s, k in items if arm == "base"]
        olds = sorted({b[0] for b in base})
        if len(base) != 20 or len(olds) != 5 or any(sum(1 for b in base if b[0] == o) != 4 for o in olds):
            issues.append(f"base allocation {f},{ro}")
        for d in range(1, DRAWS + 1):
            arm_items = {a: [(subj, key, s, k) for dd, aa, subj, key, s, k in items if dd == d and aa == a]
                         for a in ARMS}
            all_blocks = [(b[1], b[2]) for b in base] + [(i[1], i[2]) for a in ARMS for i in arm_items[a]]
            if len(set(all_blocks)) != 20 + 30:
                issues.append(f"overlap {f},{ro},{d}")
            thirds = {a: sorted(i[3] for i in arm_items[a]) for a in ARMS}
            if not (thirds["deepen"] == thirds["other_night"] == thirds["add_people"]):
                issues.append(f"thirds unmatched {f},{ro},{d}")
            base_keys = {b[1] for b in base}
            if {i[1] for i in arm_items["deepen"]} - base_keys:
                issues.append(f"deepen outside base night {f},{ro},{d}")
            other_keys = {i[1] for i in arm_items["other_night"]}
            if other_keys & base_keys or {i[0] for i in arm_items["other_night"]} - set(olds):
                issues.append(f"other_night not other night of same people {f},{ro},{d}")
            new_subj = {i[0] for i in arm_items["add_people"]}
            if new_subj & set(olds) or len(new_subj) != 5:
                issues.append(f"newcomers {f},{ro},{d}")
            # 新人取的夜号与原人另一夜一致：按 (夜号, 时段) 的多重集比较
            other_dk = sorted((i[1].split("_Day")[1], i[3]) for i in arm_items["other_night"])
            new_dk = sorted((i[1].split("_Day")[1], i[3]) for i in arm_items["add_people"])
            if other_dk != new_dk:
                issues.append(f"newcomer day/third not matched to other_night {f},{ro},{d}")
            if any(len([i for i in arm_items[a]]) != 10 for a in ARMS):
                issues.append(f"arm size {f},{ro},{d}")
    return issues


# ---------- 拟合 ----------
def run_fits(plans, recs, x, y, fold_list, out, y_eval=None):
    y_eval = y if y_eval is None else y_eval
    sys.path.insert(0, str(ROOT / "experiment3/week1_budget_20261126"))
    import torch
    from train import fit
    torch.set_num_threads(4)
    counts = np.zeros((FOLDS, ROSTERS, DRAWS, 3, 5, 25), dtype=np.int64)  # 评估者 × (真值×5+预测)
    t0 = time.perf_counter()
    for f, ev in enumerate(fold_list):
        cache = out / f"b_fold{f+1}_confusions.npy"
        if cache.exists():
            counts[f] = np.load(cache)
            continue
        idx, who = [], []
        for j, s in enumerate(ev):
            for d in (1, 2):
                key = f"{s}_Day{d}"
                if key in recs:
                    r = recs[key]
                    ok = np.flatnonzero(y_eval[r["offset"]:r["offset"] + r["n"]] >= 0) + r["offset"]
                    idx.append(ok)
                    who.append(np.full(len(ok), j))
        idx, who = np.concatenate(idx), np.concatenate(who)
        xe, ye = x[idx], y_eval[idx]
        cell = who * 25 + ye * 5
        for ro in range(ROSTERS):
            base, added = plans[f, ro]
            base_ids = (np.array(base)[:, None] + np.arange(B)).ravel()
            for d in range(DRAWS):
                for a, arm in enumerate(ARMS):
                    add_ids = (np.array(added[d, arm])[:, None] + np.arange(B)).ravel()
                    ids = np.sort(np.concatenate([base_ids, add_ids]))
                    assert len(np.unique(ids)) == 1800 and (y[ids] >= 0).all()
                    model, mean, std, _ = fit(x[ids], y[ids], 0, 30)
                    with torch.inference_mode():
                        pred = model(torch.from_numpy((xe - mean) / std)).argmax(1).numpy()
                    counts[f, ro, d, a] = np.bincount(cell + pred, minlength=125).reshape(5, 25)
            print(f"fold {f+1} roster {ro+1} {time.perf_counter()-t0:.0f}s", flush=True)
        np.save(cache, counts[f], allow_pickle=False)
    return counts


# ---------- 汇总 ----------
def macro_f1(conf25):
    m = conf25.reshape(*conf25.shape[:-1], 5, 5).astype(float)
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    denom = m.sum(-1) + m.sum(-2)
    f1 = np.where(denom > 0, 200 * tp / np.where(denom > 0, denom, 1), 0.0)
    return f1.mean(-1)


def contrasts(f1):  # [..., 3 arms] -> [..., 3 contrasts]
    return np.stack([f1[..., 1] - f1[..., 0], f1[..., 2] - f1[..., 1], f1[..., 2] - f1[..., 0]], -1)


NAMES = ("other_night_minus_deepen", "add_people_minus_other_night", "add_people_minus_deepen")


def summarize(counts, n_boot=10000, seed=7):
    pooled = counts.sum(axis=4)                      # 合并评估者 -> [F,R,D,3,25]
    f1 = macro_f1(pooled)
    delta = contrasts(f1)                            # [F,R,D,3]
    rng = np.random.default_rng(seed)
    boot = np.zeros((n_boot, 3))
    for f in range(FOLDS):
        c = counts[f]                                # [R,D,3,5,25]
        for s in range(0, n_boot, 250):
            n = min(250, n_boot - s)
            w = rng.multinomial(5, [0.2] * 5, size=n)             # 评估人重抽
            pc = np.einsum("bp,rdapk->brdak", w, c)
            r_idx = rng.integers(ROSTERS, size=(n, ROSTERS))
            d_idx = rng.integers(DRAWS, size=(n, ROSTERS, DRAWS))
            dl = contrasts(macro_f1(pc))                          # [n,R,D,3]
            sel = dl[np.arange(n)[:, None, None], r_idx[:, :, None], d_idx]
            boot[s:s + n] += sel.mean(axis=(1, 2)) / FOLDS
    res = {}
    for k, name in enumerate(NAMES):
        d = delta[..., k]
        res[name] = dict(mean_pp=float(d.mean()), ci95_pp=np.percentile(boot[:, k], [2.5, 97.5]).tolist(),
                         fold_means_pp=d.mean(axis=(1, 2)).tolist(),
                         positive_rosters=int((d.mean(axis=2) > 0).sum()),
                         loo_fold=[float(np.delete(d.mean(axis=(1, 2)), f).mean()) for f in range(FOLDS)])
    p = res["add_people_minus_deepen"]
    if p["mean_pp"] >= 1 and p["ci95_pp"][0] > 0 and sum(v > 0 for v in p["fold_means_pp"]) >= 3:
        verdict = "external_support"
    elif p["ci95_pp"][1] < 1:
        verdict = "below_practical_target"
    else:
        verdict = "inconclusive"
    return dict(decision=verdict, arm_f1=dict(zip(ARMS, f1.mean(axis=(0, 1, 2)).tolist())),
                cohort_epochs=int(counts[:, 0, 0, 0].sum()), contrasts=res), f1, delta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--features", default="features")
    p.add_argument("--design-only", action="store_true")
    p.add_argument("--design-seed-offset", type=int, default=1)
    p.add_argument("--variant", choices=["main", "no_eligibility", "scored_majority", "scored_majority_eval_only", "design_seed"], default="main")
    a = p.parse_args()
    global VARIANT
    VARIANT = a.variant
    global DESIGN_SEED_OFFSET
    DESIGN_SEED_OFFSET = a.design_seed_offset
    if a.variant != "main":
        tag = a.variant + (f"_{a.design_seed_offset}" if a.variant == "design_seed" else "")
        a.out_main, a.out = a.out, a.out / f"sens_{tag}"
        a.out.mkdir(exist_ok=True)
        feat = a.out_main / a.features
    else:
        feat = a.out / a.features
    y_eval = None
    if VARIANT == "scored_majority_eval_only":
        VARIANT = "scored_majority"
        _, _, y_eval = load(feat)
        VARIANT = "main"
    recs, x, y = load(feat)
    fl = folds()
    c_proto = json.loads((C_DIR / "protocol.json").read_text(encoding="utf-8"))
    report = {"folds_match_c_protocol": fl == c_proto["evaluation_folds"], "folds": fl,
              "n_nights": len(recs), "n_epochs_all": int(len(y)), "n_epochs_scored": int((y >= 0).sum())}
    rows, plans, elig = design(recs, fl)
    data = write_blocks(a.out / "blocks_b.csv", rows)
    c_bytes = (C_DIR / "blocks.csv").read_bytes()
    report["blocks_sha256_b"] = hashlib.sha256(data).hexdigest()
    report["blocks_sha256_c"] = hashlib.sha256(c_bytes).hexdigest()
    report["blocks_identical"] = data.replace(b"\r\n", b"\n") == c_bytes.replace(b"\r\n", b"\n")
    report["eligible_training_pool_by_fold"] = elig
    c_rows = list(csv.reader(io.StringIO(c_bytes.decode())))
    report["c_blocks_header"] = c_rows[0]
    parsed = [(int(r[0]), int(r[1]), int(r[2]), r[3], r[4], r[5], int(r[6]), int(r[7])) for r in c_rows[1:]]
    issues = audit_blocks(parsed, recs, y, fl)
    report["c_blocks_audit_issues"] = issues[:50]
    report["c_blocks_audit_n_issues"] = len(issues)
    report["n_block_rows"] = len(parsed)
    # 人员隔离：评估折之间不重叠、并集为 20 人；训练名单不含本折评估人
    report["folds_disjoint_cover_20"] = sorted(sum(fl, [])) == [f"SLP{i:03d}" for i in range(1, 21)]
    report["training_subjects_in_eval_fold"] = sum(1 for r in parsed if r[4] in fl[r[0] - 1])
    (a.out / "design_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("eligible_training_pool_by_fold",)},
                     ensure_ascii=False)[:2000], flush=True)
    if a.design_only:
        return
    counts = run_fits(plans, recs, x, y, fl, a.out, y_eval)
    summary, f1, delta = summarize(counts)
    if a.variant != "main":
        summary["variant"] = a.variant
        summary["eligible_training_pool_by_fold"] = elig
        (a.out / "summary_b.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(json.dumps(summary, indent=1), flush=True)
        return
    # 与 C 比较
    c_sum = json.loads((C_DIR / "summary.json").read_text(encoding="utf-8"))
    c_counts = np.stack([np.load(C_DIR / f"fold{f+1}_confusions.npy") for f in range(FOLDS)])
    c_f1 = macro_f1(c_counts.sum(axis=4))
    cmp = {"confusion_cells_differ": int((c_counts != counts).sum()), "confusion_cells_total": int(counts.size),
           "fits_with_any_count_diff": int((c_counts != counts).any(axis=(4, 5)).sum()),
           "max_abs_fit_f1_diff_pp": float(np.abs(c_f1 - f1).max()),
           "mean_abs_fit_f1_diff_pp": float(np.abs(c_f1 - f1).mean()),
           "max_abs_delta_diff_pp": float(np.abs(contrasts(c_f1) - delta).max()),
           "contrast_mean_diff_pp": {n: summary["contrasts"][n]["mean_pp"] - c_sum["contrasts"][n]["mean_pp"]
                                     for n in NAMES},
           "fold_mean_diff_pp": {n: (np.array(summary["contrasts"][n]["fold_means_pp"])
                                     - np.array(c_sum["contrasts"][n]["fold_means_pp"])).tolist() for n in NAMES},
           "c_decision": c_sum["decision"], "b_decision": summary["decision"]}
    # 用 B 的 bootstrap 跑 C 的计数，分离蒙特卡洛与数据差异
    c_with_b_boot, _, _ = summarize(c_counts)
    cmp["c_counts_with_b_bootstrap"] = {n: c_with_b_boot["contrasts"][n]["ci95_pp"] for n in NAMES}
    summary["comparison_with_c"] = cmp
    (a.out / "summary_b.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (a.out / "per_fit_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "roster", "draw", "arm", "macro_f1_b", "macro_f1_c", "diff_pp"])
        for idx in np.ndindex(FOLDS, ROSTERS, DRAWS, 3):
            w.writerow([idx[0] + 1, idx[1] + 1, idx[2] + 1, ARMS[idx[3]],
                        f"{f1[idx]:.6f}", f"{c_f1[idx]:.6f}", f"{f1[idx]-c_f1[idx]:.6f}"])
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
