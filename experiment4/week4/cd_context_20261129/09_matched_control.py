"""D：来源与时段匹配的随机对照（V − R 达到跟进门槛后触发，方案第 3 页）。

对每份名单、每次重复，读取 V（coverage）的入选表，按“原人原夜 / 原人另一夜 / 新人” × “早 / 中 / 晚”统计块数，
在同一格的合法候选块（条件 A、与起点无交集）中不放回等概率抽取同样块数。不按分期匹配；可与 V 重合。
来源类别与 composition.csv 相同：块若属于原人原夜记为原人原夜，否则若属于原人另一夜记为另一夜，否则记为新人。
种子 SEED_M = 2026112925。**披露**：方案要求在看到主比较前锁定本脚本和种子，D 未做到；本脚本在主比较出分后写成，
只用一个预先写死的种子运行一次，抽样过程不依赖任何得分。
输出：matched_blocks.csv（make_d_input.py 格式，strategy = matched）、matched_audit.csv（重合率、无选择余地格数）、
matched_lock.json。
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED_M = 2026112925
BLOCK = 60


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def wr(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def category(sources):
    return "deepen" if "deepen" in sources else "other_night" if "other_night" in sources else "add_people"


def main():
    ap = argparse.ArgumentParser()
    for n in ("candidates", "anchor", "d-blocks", "lock", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    lock = json.loads(a.lock.read_text())
    if sha(a.d_blocks) != lock["outputs"]["D_blocks.csv"]:
        fail("D_blocks.csv differs from the selection lock")
    a.out.mkdir(parents=True, exist_ok=True)
    anchors = defaultdict(set)
    for r in rd(a.anchor):
        f0 = int(r["first_epoch"])
        anchors[int(r["roster"])].update((r["record_id"], e) for e in range(f0, f0 + BLOCK))
    info = {}
    for r in rd(a.candidates):
        ro = int(r["roster"])
        if ro not in anchors or not int(r["in_A"]):
            continue
        cur = info.setdefault((ro, r["block_id"]), dict(record_id=r["record_id"], subject_id=r["subject_id"],
                                                        first=int(r["first_epoch"]), third=int(r["night_third"]),
                                                        sources=set()))
        cur["sources"].add(r["arm"])
    cells = defaultdict(list)
    for (ro, b), I in info.items():
        if {(I["record_id"], e) for e in range(I["first"], I["first"] + BLOCK)} & anchors[ro]:
            continue
        cells[(ro, category(I["sources"]), I["third"])].append(b)
    for k in cells:
        cells[k].sort()
    v = defaultdict(list)
    for r in rd(a.d_blocks):
        if r["strategy"] == "coverage":
            v[(int(r["roster"]), int(r["draw"]))].append(r["block_id"])
    rows, audit = [], []
    for (ro, d) in sorted(v):
        chosen = v[(ro, d)]
        need = defaultdict(int)
        for b in chosen:
            I = info[(ro, b)]
            need[(category(I["sources"]), I["third"])] += 1
        rng = np.random.default_rng([SEED_M, ro, d])
        picked, no_choice = [], 0
        for (cat, th) in sorted(need):
            pool = cells[(ro, cat, th)]
            n = need[(cat, th)]
            if len(pool) < n:
                fail(f"roster {ro} draw {d}: cell {cat}/{th} has {len(pool)} < {n}")
            no_choice += len(pool) == n
            picked += [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))]
        for b in picked:
            I = info[(ro, b)]
            rows.append(dict(roster=ro, draw=d, strategy="matched", record_id=I["record_id"], first_epoch=I["first"],
                             last_epoch=I["first"] + BLOCK - 1, block_id=b, subject_id=I["subject_id"],
                             night_third=I["third"], sources="+".join(sorted(I["sources"]))))
        audit.append(dict(roster=ro, draw=d, overlap_with_V=len(set(picked) & set(chosen)), cells=len(need),
                          cells_without_choice=no_choice,
                          subjects_V=len({info[(ro, b)]["subject_id"] for b in chosen}),
                          subjects_matched=len({info[(ro, b)]["subject_id"] for b in picked}),
                          nights_V=len({info[(ro, b)]["record_id"] for b in chosen}),
                          nights_matched=len({info[(ro, b)]["record_id"] for b in picked})))
    wr(a.out / "matched_blocks.csv", rows)
    wr(a.out / "matched_audit.csv", audit)
    out = dict(seed=SEED_M, disclosure="script and seed written after the primary V-R result; single run",
               d_blocks_sha256=sha(a.d_blocks), matched_blocks_sha256=sha(a.out / "matched_blocks.csv"),
               code_sha256=sha(Path(__file__)), units=len(audit),
               mean_overlap_with_V=float(np.mean([r["overlap_with_V"] for r in audit])),
               mean_cells_without_choice=float(np.mean([r["cells_without_choice"] for r in audit])))
    (a.out / "matched_lock.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
