"""D：实验一（固定关灯网格，只改候选规则）的共享随机排序与选块。

条件（同一张关灯网格，C 的 window_budget_c_20261128 锁定候选块，只用 paired_roster=1）：
  A  关灯网格全部候选块；
  B  只保留完整落入分期辅助睡眠边界 [b_lo, b_hi) 的块（边界见 01_sleep_boundaries.py，与旧窗口裁剪规则相同）；
  C  每份名单、每次重复、每条记录 × 每个时段，从 A 的块中按"保留优先级"取与 B 相同数量的块。
策略：
  deepen / other_night / add_people：沿用原名单的格子（受试者 × 记录 × 时段），每格取 1 块；
  random：合并三种来源的合格块、按（记录, 起始段）去重，每个时段取与新人组同样多的块。
随机顺序（任务书第 2 页）：每份名单、每次重复，对全部唯一块（先按块键稳定排序）生成两套优先级：
  k=0 只用于构造条件 C 的保留池；k=1 为选块顺序，A、B、C 和四种策略共用。
  rng = numpy.random.default_rng([SEED, 名单, 重复, k])，优先级小者先选，平分按块键。
不读任何新增片段的分期标签（B 只用到睡眠边界，这正是条件 B 的定义）。

输出：<out>/selections.csv、feasibility.csv、random_source_composition.csv、sampler_lock.json
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
STRATS = ARMS + ("random",)


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
    ap.add_argument("--window-dir", type=Path, required=True, help="C's window_budget_c_20261128/results")
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lock = json.loads((a.window_dir / "protocol.json").read_text())
    for name, digest in lock["files"].items():
        if sha(a.window_dir / name) != digest:
            fail(f"C lock mismatch: {name}")
    rosters = sorted(lock["paired_rosters"])
    bnd = {r["record_id"]: (int(r["b_lo"]), int(r["b_hi_excl"])) for r in rd(a.boundaries)}
    anchors = defaultdict(set)
    for r in rd(a.window_dir / "anchor_blocks.csv"):
        f0 = int(r["first_epoch"])
        anchors[int(r["roster"])].update((r["record_id"], e) for e in range(f0, f0 + BLOCK))
    cells = defaultdict(lambda: defaultdict(list))   # roster -> (arm, subj, rec, third) -> [first...]
    for r in rd(a.window_dir / "candidate_blocks.csv"):
        if r["paired_roster"] != "1":
            continue
        cells[int(r["roster"])][(r["arm"], r["subject_id"], r["record_id"], int(r["night_third"]))].append(int(r["first_epoch"]))

    sel_rows, feas_rows, comp_rows = [], [], []
    cap = defaultdict(set)   # (cond) -> unique blocks across rosters (for capacity report)
    for ro in rosters:
        cs = cells[ro]
        if len(cs) != 30:
            fail(f"roster {ro}: expected 30 cells, got {len(cs)}")
        for k in cs:
            for f0 in cs[k]:
                eps = {(k[2], e) for e in range(f0, f0 + BLOCK)}
                if eps & anchors[ro]:
                    fail(f"roster {ro}: candidate overlaps anchor")
        quota = defaultdict(int)
        for (arm, s, rec, th) in cs:
            if arm == "add_people":
                quota[th] += 1
        if sum(quota.values()) != 10:
            fail(f"roster {ro}: new-arm cells != 10")
        # 唯一块与所属格
        block_cells = defaultdict(set)
        for key, firsts in cs.items():
            for f0 in firsts:
                block_cells[(key[2], f0)].add(key)
        keys = sorted(block_cells)
        th_of = {}
        for b, ks in block_cells.items():
            ths = {k[3] for k in ks}
            if len(ths) != 1:
                fail(f"roster {ro}: block {b} in two thirds")
            th_of[b] = ths.pop()
        in_b = {b: (bnd[b[0]][0] <= b[1] and b[1] + BLOCK <= bnd[b[0]][1]) for b in keys}
        for b in keys:
            cap["A"].add(b)
            if in_b[b]:
                cap["B"].add(b)
        # 记录 × 时段的块集合（用于 C）
        rec_th = defaultdict(list)
        for b in keys:
            rec_th[(b[0], th_of[b])].append(b)
        for d in range(1, N_DRAWS + 1):
            pri = {}
            for kk in (0, 1):
                u = np.random.default_rng([SEED, ro, d, kk]).random(len(keys))
                pri[kk] = {b: (float(u[i]), b) for i, b in enumerate(keys)}
            # 各条件的合格集合
            elig = {"A": set(keys), "B": {b for b in keys if in_b[b]}}
            keepC = set()
            for rt, bl in rec_th.items():
                nB = sum(in_b[b] for b in bl)
                keepC.update(sorted(bl, key=lambda b: pri[0][b])[:nB])
            elig["C"] = keepC
            for cond in CONDS:
                E = elig[cond]
                picked = {}
                ok = {}
                for arm in ARMS:
                    chosen, good = [], True
                    for key in sorted(k for k in cs if k[0] == arm):
                        opts = [(key[2], f0) for f0 in cs[key] if (key[2], f0) in E]
                        if not opts:
                            good = False
                            continue
                        chosen.append((min(opts, key=lambda b: pri[1][b]), key[3]))
                    picked[arm], ok[arm] = chosen, good
                # random：三来源合并去重，按时段配额取优先级最前的块
                chosen, good = [], True
                for th, q in sorted(quota.items()):
                    pool = sorted((b for b in keys if th_of[b] == th and b in E), key=lambda b: pri[1][b])
                    if len(pool) < q:
                        good = False
                    chosen += [(b, th) for b in pool[:q]]
                picked["random"], ok["random"] = chosen, good
                for st in STRATS:
                    feas_rows.append(dict(condition=cond, roster=ro, draw=d, strategy=st, feasible=int(ok[st]),
                                          blocks=len(picked[st])))
                    if not ok[st]:
                        continue
                    if len(picked[st]) != 10 or len({b for b, _ in picked[st]}) != 10:
                        fail(f"{cond} roster {ro} draw {d} {st}: not 10 unique blocks")
                    for b, th in picked[st]:
                        srcs = sorted({k[0] for k in block_cells[b]})
                        subj = next(iter(block_cells[b]))[1]
                        sel_rows.append(dict(condition=cond, roster=ro, draw=d, strategy=st, subject_id=subj,
                                             record_id=b[0], night_third=th, first_epoch=b[1],
                                             last_epoch=b[1] + BLOCK - 1, sources="|".join(srcs),
                                             select_priority=round(pri[1][b][0], 8)))
                    if st == "random":
                        src = defaultdict(int)
                        for b, _ in picked[st]:
                            for s in {k[0] for k in block_cells[b]}:
                                src[s] += 1
                        comp_rows.append(dict(condition=cond, roster=ro, draw=d,
                                              from_deepen=src["deepen"], from_other_night=src["other_night"],
                                              from_add_people=src["add_people"],
                                              people=len({next(iter(block_cells[b]))[1] for b, _ in picked[st]}),
                                              nights=len({b[0] for b, _ in picked[st]})))
    a.out.mkdir(parents=True, exist_ok=True)
    wr(a.out / "selections.csv", sel_rows)
    wr(a.out / "feasibility.csv", feas_rows)
    wr(a.out / "random_source_composition.csv", comp_rows)
    # 名单级可行性汇总：某策略在三个条件、五次重复都可行才算该名单可用
    ok_r = defaultdict(lambda: defaultdict(bool))
    for st in STRATS:
        for ro in rosters:
            ok_r[st][ro] = all(r["feasible"] for r in feas_rows if r["strategy"] == st and r["roster"] == ro)
    summary = {
        "new_and_random": sum(ok_r["add_people"][r] and ok_r["random"][r] for r in rosters),
        "deepen_and_random": sum(ok_r["deepen"][r] and ok_r["random"][r] for r in rosters),
        "other_and_random": sum(ok_r["other_night"][r] and ok_r["random"][r] for r in rosters),
        "all_four": sum(all(ok_r[s][r] for s in STRATS) for r in rosters),
    }
    lk = dict(seed=SEED, rng="numpy default_rng([seed, roster, draw, k]); k=0 C-pool priority, k=1 shared selection priority",
              draws=N_DRAWS, rosters=rosters, unique_blocks_A=len(cap["A"]), unique_blocks_B=len(cap["B"]),
              roster_feasibility=summary, window_protocol_sha256=sha(a.window_dir / "protocol.json"),
              boundaries_sha256=sha(a.boundaries), sampler_sha256=sha(__file__),
              selections_sha256=sha(a.out / "selections.csv"), labels_of_new_epochs_read=False,
              status="DEBUG build on C's existing lights-off candidate table; relock on C's formal candidate table")
    (a.out / "sampler_lock.json").write_text(json.dumps(lk, indent=2) + "\n")
    print(json.dumps({k: lk[k] for k in ("unique_blocks_A", "unique_blocks_B", "roster_feasibility")}))
    # 共享顺序效果：A 与 B、A 与 C 的选块重合率（新人组、随机组）
    S = defaultdict(set)
    for r in sel_rows:
        S[r["condition"], r["roster"], r["draw"], r["strategy"]].add((r["record_id"], r["first_epoch"]))
    for st in ("add_people", "random"):
        for c2 in ("B", "C"):
            ov = [len(S["A", ro, d, st] & S[c2, ro, d, st]) / 10 for ro in rosters for d in range(1, 6)
                  if S["A", ro, d, st] and S[c2, ro, d, st]]
            print(f"overlap A vs {c2} ({st}): mean {np.mean(ov):.2f} of blocks shared over {len(ov)} draws")


if __name__ == "__main__":
    main()
