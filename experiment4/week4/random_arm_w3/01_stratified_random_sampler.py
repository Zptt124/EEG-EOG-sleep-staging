"""D：分层随机抽样组（random arm）的抽块器。

规则（任务书）：
  - 每份名单合并三组（原人原夜 deepen、原人另一夜 other_night、新人新夜 add_people）的合格候选块，
    按 (record_id, first_epoch) 去重，并剔除与 1,200 段起点重叠的块；
  - 固定时段配额：每个夜间三分位抽的块数 = 其他三组在该时段各自追加的块数（三组相同，脚本核对）；
  - 每个时段内按块等概率、不放回抽取；每名单 5 次独立抽块（与原设计的抽块次数一致）。
  - 只用记录号、epoch 号和时段信息，不读分期标签。

窗口：
  --window old        C 开发复验（week3_budget_replication_20261128）的旧窗口候选池，用于调通；
  --window lights_off C 关灯窗口锁定的候选池（window_budget_c_20261128，只用 paired_roster=1），正式比较。

输出：<out>/random_blocks.csv、<out>/random_coverage.csv、<out>/sampler_lock.json
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

BLOCK = 60
N_DRAWS = 5
SEED = {"old": 2026112901, "lights_off": 2026112902}
ARMS = ("deepen", "other_night", "add_people")


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
    ap.add_argument("--window", choices=("old", "lights_off"), required=True)
    ap.add_argument("--c-src", type=Path, required=True, help="root containing experiment3/")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    old_dir = a.c_src / "experiment3/week3_budget_replication_20261128/results"
    if a.window == "old":
        cand_path = old_dir / "candidate_blocks.csv"
        quota_path = old_dir / "blocks.csv"
        anchor_path = old_dir / "blocks.csv"
        cands = rd(cand_path)
        rosters = sorted({int(r["roster"]) for r in cands})
    else:
        wdir = a.c_src / "experiment3/window_budget_c_20261128/results"
        lock = json.loads((wdir / "protocol.json").read_text())
        for name, digest in lock["files"].items():
            if sha(wdir / name) != digest:
                fail(f"C lock mismatch: {name}")
        cand_path = wdir / "candidate_blocks.csv"
        quota_path = wdir / "selected_blocks.csv"
        anchor_path = wdir / "anchor_blocks.csv"
        cands = [r for r in rd(cand_path) if r["paired_roster"] == "1"]
        rosters = sorted(lock["paired_rosters"])
        if sorted({int(r["roster"]) for r in cands}) != rosters:
            fail("paired rosters in candidates differ from protocol")

    # 起点（已标）epoch
    anchor = defaultdict(set)
    for r in rd(anchor_path):
        if r["arm"] == "base":
            f0 = int(r["first_epoch"])
            anchor[int(r["roster"])].update((r["record_id"], e) for e in range(f0, f0 + BLOCK))
    # 时段配额：各组在每个 draw 的各时段块数，要求三组、五次都相同
    per = defaultdict(lambda: defaultdict(int))
    for r in rd(quota_path):
        if r["arm"] in ARMS and int(r["draw"]) >= 1 and int(r["roster"]) in rosters:
            per[int(r["roster"]), int(r["draw"]), r["arm"]][int(r["night_third"])] += 1
    quota = {}
    for ro in rosters:
        qs = {tuple(sorted(per[ro, d, arm].items())) for d in range(1, N_DRAWS + 1) for arm in ARMS}
        if len(qs) != 1:
            fail(f"roster {ro}: arms/draws do not share the same third quota: {qs}")
        quota[ro] = dict(next(iter(qs)))
        if sum(quota[ro].values()) != 10:
            fail(f"roster {ro}: quota does not sum to 10 blocks")

    # 合并候选池
    pool = defaultdict(dict)  # (roster, third) -> {(record, first): info}
    for r in cands:
        ro, th = int(r["roster"]), int(r["night_third"])
        key = (r["record_id"], int(r["first_epoch"]))
        info = pool[ro, th].setdefault(key, dict(subject_id=r["subject_id"], record_id=key[0],
                                                 first_epoch=key[1], sources=set()))
        if info["subject_id"] != r["subject_id"]:
            fail(f"block {key} listed under two subjects")
        info["sources"].add(r["arm"])
    rows, cover = [], []
    stats = dict(pool_blocks=0, dropped_anchor_overlap=0)
    for ro in rosters:
        rng = random.Random(SEED[a.window] * 100 + ro)
        clean = {}
        for th in (1, 2, 3):
            items = []
            for key, info in sorted(pool[ro, th].items()):
                eps = {(key[0], e) for e in range(key[1], key[1] + BLOCK)}
                if eps & anchor[ro]:
                    stats["dropped_anchor_overlap"] += 1
                    continue
                items.append(info)
            # 同一时段内块之间不能重叠（候选网格本应不重叠，这里核对）
            spans = defaultdict(list)
            for it in items:
                spans[it["record_id"]].append(it["first_epoch"])
            for rec, fs in spans.items():
                fs.sort()
                if any(b - a_ < BLOCK for a_, b in zip(fs, fs[1:])):
                    fail(f"roster {ro} third {th}: overlapping candidate blocks in {rec}")
            k = quota[ro].get(th, 0)
            if len(items) < k:
                fail(f"roster {ro} third {th}: pool {len(items)} < quota {k}")
            clean[th] = items
            stats["pool_blocks"] += len(items)
        for d in range(1, N_DRAWS + 1):
            picked = []
            for th in (1, 2, 3):
                k = quota[ro].get(th, 0)
                for it in rng.sample(clean[th], k):
                    picked.append(dict(roster=ro, draw=d, arm="random", subject_id=it["subject_id"],
                                       record_id=it["record_id"], night_third=th, first_epoch=it["first_epoch"],
                                       last_epoch=it["first_epoch"] + BLOCK - 1,
                                       sources="|".join(sorted(it["sources"])),
                                       pool_size_in_third=len(clean[th])))
            eps = [(p["record_id"], e) for p in picked for e in range(p["first_epoch"], p["first_epoch"] + BLOCK)]
            if len(eps) != 600 or len(set(eps)) != 600 or set(eps) & anchor[ro]:
                fail(f"roster {ro} draw {d}: malformed random allocation")
            rows.extend(picked)
            src = defaultdict(int)
            for p in picked:
                for s in p["sources"].split("|"):
                    src[s] += 1
            cover.append(dict(roster=ro, draw=d, blocks=len(picked),
                              people=len({p["subject_id"] for p in picked}),
                              nights=len({p["record_id"] for p in picked}),
                              from_deepen=src["deepen"], from_other_night=src["other_night"],
                              from_add_people=src["add_people"],
                              quota="/".join(str(quota[ro].get(t, 0)) for t in (1, 2, 3))))
    a.out.mkdir(parents=True, exist_ok=True)
    wr(a.out / "random_blocks.csv", rows)
    wr(a.out / "random_coverage.csv", cover)
    lock_out = dict(window=a.window, seed_rule=f"random.Random({SEED[a.window]} * 100 + roster)", draws=N_DRAWS,
                    rosters=rosters, candidate_file=str(cand_path.name), candidate_sha256=sha(cand_path),
                    quota_file=str(quota_path.name), quota_sha256=sha(quota_path),
                    anchor_file=str(anchor_path.name), anchor_sha256=sha(anchor_path),
                    sampler_sha256=sha(__file__), labels_read=False, **stats,
                    random_blocks_sha256=sha(a.out / "random_blocks.csv"))
    (a.out / "sampler_lock.json").write_text(json.dumps(lock_out, indent=2) + "\n")
    n = len(cover)
    mean = lambda k: sum(c[k] for c in cover) / n
    print(f"{a.window}: {len(rosters)} rosters x {N_DRAWS} draws, {len(rows)} blocks; "
          f"mean people {mean('people'):.2f}, nights {mean('nights'):.2f}, "
          f"from deepen/other/new {mean('from_deepen'):.2f}/{mean('from_other_night'):.2f}/{mean('from_add_people'):.2f}; "
          f"anchor-overlap dropped {stats['dropped_anchor_overlap']}")


if __name__ == "__main__":
    main()
