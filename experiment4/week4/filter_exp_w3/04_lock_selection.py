"""D：实验一锁定选块包（基于成员 C 的正式交接表 handoff.json）。

输入（成员 C，逐文件核对 handoff.json 的 SHA-256）：
  candidate_blocks.csv（块键 block_id = 记录号:起始 epoch；in_A / in_B）、cell_capacity.csv（n_C_to_keep）、
  mixed_quotas.csv（混合随机时段配额）、anchor_blocks.csv、existing_feature_index.csv、roster_feasibility.csv。
随机顺序（任务书第 2 页）：每份名单、每次重复，把该名单的唯一块按 block_id 稳定排序后生成两套优先级：
  k=0 只用于构造条件 C 的保留池；k=1 为 A、B、C 与各策略共用的选块顺序。
  numpy.random.default_rng([SEED, 名单, 重复, k]).random(n)，数值小者优先，平分按 block_id。
条件：A = in_A；B = in_B；C = 每个格（名单 × 受试者 × 记录 × 时段）从 A 中按 k=0 优先级保留 n_C_to_keep 个。
策略：add_people / deepen / other_night 每格按 k=1 取 1 块；random（混合随机）合并三种来源、按 block_id 去重，
  每个时段按 mixed_quotas 取 k=1 最前的块。主分析 = add_people + random（26 × 5 × 3 × 2 = 780 单元）；
  deepen / other_night 只在其可行子集内给出，供附表。
不读取任何新增片段的分期标签。

输出目录：priorities.csv、c_pools.csv、selections.csv、random_source_composition.csv、feasibility.csv、
  epochs_needed.csv（入选块全部 epoch 及其在既有缓存中的来源，标出待补提）、spotcheck_units.csv、lock.json
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 2026112903
N_DRAWS = 5
BLOCK = 60
CONDS = ("A", "B", "C")
ARMS = ("deepen", "other_night", "add_people")
STRATS = ("add_people", "random", "deepen", "other_night")
PRIMARY = ("add_people", "random")


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
    ap.add_argument("--handoff", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    H = a.handoff
    man = json.loads((H / "handoff.json").read_text())
    for name, digest in man["files"].items():
        if sha(H / name) != digest:
            fail(f"handoff file changed: {name}")
    rosters = man["rosters"]
    cand = rd(H / "candidate_blocks.csv")
    cap = {(int(r["roster"]), r["arm"], r["subject_id"], r["record_id"], int(r["night_third"])): r for r in rd(H / "cell_capacity.csv")}
    quota = defaultdict(dict)
    for r in rd(H / "mixed_quotas.csv"):
        quota[int(r["roster"])][int(r["night_third"])] = int(r["quota"])
    feas_c = {int(r["roster"]): r for r in rd(H / "roster_feasibility.csv")}
    anchors = defaultdict(set)
    for r in rd(H / "anchor_blocks.csv"):
        f0 = int(r["first_epoch"])
        anchors[int(r["roster"])].update((r["record_id"], e) for e in range(f0, f0 + BLOCK))
    feat_src = {(r["record_id"], int(r["epoch_index"])): r["cache_source"] for r in rd(H / "existing_feature_index.csv")}

    cells = defaultdict(lambda: defaultdict(list))
    info = {}
    for r in cand:
        ro = int(r["roster"])
        cell = (r["arm"], r["subject_id"], r["record_id"], int(r["night_third"]))
        bid = r["block_id"]
        if bid != f"{r['record_id']}:{int(r['first_epoch'])}" or int(r["last_epoch"]) != int(r["first_epoch"]) + BLOCK - 1:
            fail(f"bad block key {bid}")
        cells[ro][cell].append(bid)
        prev = info.get((ro, bid))
        cur = dict(record_id=r["record_id"], subject_id=r["subject_id"], first=int(r["first_epoch"]),
                   third=int(r["night_third"]), in_A=int(r["in_A"]), in_B=int(r["in_B"]))
        if prev and any(prev[k] != cur[k] for k in ("record_id", "first", "third", "in_A", "in_B")):
            fail(f"inconsistent block {bid} in roster {ro}")
        info[(ro, bid)] = cur
        info[(ro, bid)].setdefault("sources", set())
        info[(ro, bid)]["sources"] = (prev or {}).get("sources", set()) | {r["arm"]}

    pri_rows, pool_rows, sel_rows, comp_rows, feas_rows = [], [], [], [], []
    for ro in rosters:
        cs = cells[ro]
        if len(cs) != 30:
            fail(f"roster {ro}: expected 30 cells")
        new_q = defaultdict(int)
        for (arm, s, rec, th) in cs:
            if arm == "add_people":
                new_q[th] += 1
        if dict(new_q) != quota[ro]:
            fail(f"roster {ro}: mixed quota {quota[ro]} != new-arm cells {dict(new_q)}")
        keys = sorted({b for v in cs.values() for b in v})
        for b in keys:
            I = info[(ro, b)]
            if {(I["record_id"], e) for e in range(I["first"], I["first"] + BLOCK)} & anchors[ro]:
                fail(f"roster {ro}: {b} overlaps anchor")
        for d in range(1, N_DRAWS + 1):
            u0 = np.random.default_rng([SEED, ro, d, 0]).random(len(keys))
            u1 = np.random.default_rng([SEED, ro, d, 1]).random(len(keys))
            p0 = {b: (float(u0[i]), b) for i, b in enumerate(keys)}
            p1 = {b: (float(u1[i]), b) for i, b in enumerate(keys)}
            for b in keys:
                pri_rows.append(dict(roster=ro, draw=d, block_id=b, priority_cpool=repr(p0[b][0]), priority_select=repr(p1[b][0])))
            elig = {"A": {b for b in keys if info[(ro, b)]["in_A"]}, "B": {b for b in keys if info[(ro, b)]["in_B"]}}
            keepC = set()
            for cell, bl in cs.items():
                n = int(cap[(ro,) + cell]["n_C_to_keep"])
                if n != int(cap[(ro,) + cell]["n_B"]) or n != sum(info[(ro, b)]["in_B"] for b in bl):
                    fail(f"roster {ro} cell {cell}: C keep count != B capacity")
                kept = sorted((b for b in bl if info[(ro, b)]["in_A"]), key=lambda b: p0[b])[:n]
                keepC.update(kept)
                for b in kept:
                    pool_rows.append(dict(roster=ro, draw=d, arm=cell[0], subject_id=cell[1], record_id=cell[2],
                                          night_third=cell[3], block_id=b))
            elig["C"] = keepC
            for cond in CONDS:
                E = elig[cond]
                picked = {}
                for arm in ARMS:
                    ch, ok = [], True
                    for cell in sorted(k for k in cs if k[0] == arm):
                        opts = [b for b in cs[cell] if b in E]
                        if not opts:
                            ok = False
                            continue
                        ch.append(min(opts, key=lambda b: p1[b]))
                    picked[arm] = ch if ok else None
                ch, ok = [], True
                for th, q in sorted(quota[ro].items()):
                    pool = sorted((b for b in keys if info[(ro, b)]["third"] == th and b in E), key=lambda b: p1[b])
                    ok &= len(pool) >= q
                    ch += pool[:q]
                picked["random"] = ch if ok else None
                for st in STRATS:
                    feas_rows.append(dict(condition=cond, roster=ro, draw=d, strategy=st, feasible=int(picked[st] is not None)))
                    if picked[st] is None:
                        continue
                    if len(set(picked[st])) != 10:
                        fail(f"{cond} {ro} {d} {st}: not 10 blocks")
                    for b in picked[st]:
                        I = info[(ro, b)]
                        sel_rows.append(dict(condition=cond, roster=ro, draw=d, strategy=st, primary=int(st in PRIMARY),
                                             block_id=b, subject_id=I["subject_id"], record_id=I["record_id"],
                                             night_third=I["third"], first_epoch=I["first"], last_epoch=I["first"] + BLOCK - 1,
                                             sources="|".join(sorted(I["sources"]))))
                    if st == "random":
                        src = defaultdict(int)
                        for b in picked[st]:
                            for s in info[(ro, b)]["sources"]:
                                src[s] += 1
                        comp_rows.append(dict(condition=cond, roster=ro, draw=d, from_deepen=src["deepen"],
                                              from_other_night=src["other_night"], from_add_people=src["add_people"],
                                              people=len({info[(ro, b)]["subject_id"] for b in picked[st]}),
                                              nights=len({info[(ro, b)]["record_id"] for b in picked[st]})))
    # 可行性与 C 的名单表对照
    okr = lambda st, ro: all(r["feasible"] for r in feas_rows if r["strategy"] == st and r["roster"] == ro)
    for ro in rosters:
        for st, col in (("add_people", "add_people"), ("random", "mixed_random"), ("deepen", "deepen"), ("other_night", "other_night")):
            if int(okr(st, ro)) != int(feas_c[ro][col]):
                fail(f"roster {ro} {st}: feasibility differs from C's roster_feasibility.csv")
    # 需要的 epoch 与缓存来源
    need = {}
    for r in sel_rows:
        for e in range(int(r["first_epoch"]), int(r["last_epoch"]) + 1):
            k = (r["record_id"], e)
            if k not in feat_src:
                fail(f"epoch {k} not in C's feature index")
            need[k] = feat_src[k]
    ep_rows = [dict(record_id=k[0], epoch_index=k[1], cache_source=v) for k, v in sorted(need.items())]
    # 跨机抽查单元：最小编号可行名单、第 1 次重复，三条件 × 两策略
    r0 = min(rosters)
    spot = []
    for cond in CONDS:
        for st in PRIMARY:
            bl = sorted(r["block_id"] for r in sel_rows if r["condition"] == cond and r["roster"] == r0 and r["draw"] == 1 and r["strategy"] == st)
            spot.append(dict(unit=f"{cond}_{st}_r{r0:02d}_d1", condition=cond, strategy=st, roster=r0, draw=1,
                             blocks=";".join(bl), blocks_sha256=hashlib.sha256(";".join(bl).encode()).hexdigest()))
    a.out.mkdir(parents=True, exist_ok=True)
    files = {"priorities.csv": pri_rows, "c_pools.csv": pool_rows, "selections.csv": sel_rows,
             "random_source_composition.csv": comp_rows, "feasibility.csv": feas_rows,
             "epochs_needed.csv": ep_rows, "spotcheck_units.csv": spot}
    for n, rows in files.items():
        wr(a.out / n, rows)
    n_primary = len({(r["condition"], r["roster"], r["draw"], r["strategy"]) for r in sel_rows if r["primary"] == 1})
    cnt = defaultdict(int)
    for v in need.values():
        cnt[v] += 1
    lock = dict(
        seed=SEED,
        rng="numpy.random.default_rng([seed, roster, draw, k]).random(n_unique_blocks_in_roster) over block_id-sorted keys; "
            "k=0 condition-C retention priority, k=1 shared selection priority; smaller first, ties by block_id",
        draws=N_DRAWS, rosters=rosters, conditions=list(CONDS), primary_strategies=list(PRIMARY),
        primary_fit_units=n_primary, handoff_json_sha256=sha(H / "handoff.json"), handoff_files=man["files"],
        script_sha256=sha(__file__), numpy=np.__version__, labels_of_new_epochs_read=False,
        epochs_needed=len(need), epochs_by_cache_source=dict(cnt),
        files={n: sha(a.out / n) for n in files})
    if n_primary != 780:
        fail(f"primary units {n_primary} != 780")
    (a.out / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({k: lock[k] for k in ("primary_fit_units", "epochs_needed", "epochs_by_cache_source")}))


if __name__ == "__main__":
    main()
