"""D 独立核对 C 第二周"原人另一夜"选块（不调用 C 的 plan.py / match_blocks.py）。

用法：python 01_check_matched_blocks.py --pkg <C 包解压根目录> --out <输出目录>
检查项（每份名单）：
  1. manifest 里的文件锁 SHA 一致；
  2. 起点 20 块（base）与第一周名单逐块相同；
  3. 原人原夜追加块（deepen）= matched_blocks 的 original 列；
  4. 另一夜块：每个起点受试者 2 块、同一条另一夜记录、记录 ≠ 起点记录；
  5. 每块 60 个在缓存索引中连续存在的 epoch，块间不重叠；
  6. 另一夜块的夜间三分位（D 自己按块中点计算）与对应 deepen 块相同，也与文件记录相同；
  7. 另一夜块起点落在该记录连续段的 60 步候选网格上（沿用第一周候选规则）；
  8. training_epochs = base ∪ other_night，共 1,800，按缓存行序；
  9. 训练受试者与 held-out 8 人、dev8、pool36 评分受试者都不相交；
  10. direction 字段与记录编号先后一致。
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

BLOCK = 60
ROSTERS = ["01", "02", "03", "04", "06", "07", "08", "09", "10"]


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    w1 = a.pkg / "experiment3/week1_budget_20261126/rosters"
    w2 = a.pkg / "experiment3/week2_budget_20261127/rosters"
    idx_path = a.pkg / "data/experiment4/week4/manifests/pretrain_epoch_index.csv"
    held_path = a.pkg / "data/manifests/pretrain_split_w4_v01.csv"

    man = json.loads((w2 / "manifest.json").read_text())
    for name, digest in man["locks"].items():
        if sha(w2 / name) != digest:
            fail(f"lock mismatch: {name}")
    if man["first_week_plan_sha256"] != sha(w1 / "manifest.json"):
        fail("week-one plan sha mismatch")

    idx = rd(idx_path)
    row_of = {}
    by_rec = defaultdict(list)
    for r in idx:
        k = (r["subject_id"], r["record_id"], int(r["epoch_index"]))
        row_of[k] = int(r["row"])
        by_rec[r["record_id"]].append(int(r["epoch_index"]))
    span = {}
    grid = {}
    for rec, ep in by_rec.items():
        ep.sort()
        span[rec] = (ep[0], ep[-1])
        starts = set()
        run = [ep[0]]
        for e in ep[1:] + [None]:
            if e is not None and e == run[-1] + 1:
                run.append(e)
                continue
            for i in range(0, len(run) - BLOCK + 1, BLOCK):
                starts.add(run[i])
            if e is not None:
                run = [e]
        grid[rec] = starts

    held = {r["subject_id"] for r in rd(held_path) if r["pretrain_role"] == "heldout_val"}
    if len(held) != 8:
        cols = rd(held_path)[0].keys()
        fail(f"could not read 8 held-out subjects from {held_path} (columns {list(cols)}, got {len(held)})")

    def third(rec, start):
        lo, hi = span[rec]
        mid = start + (BLOCK - 1) / 2 - lo
        return min(3, int(3 * mid / (hi - lo + 1)) + 1)

    def block_keys(subj, rec, first, last):
        if last - first + 1 != BLOCK:
            fail(f"block length != 60: {rec} {first}-{last}")
        ks = [(subj, rec, e) for e in range(first, last + 1)]
        for k in ks:
            if k not in row_of:
                fail(f"epoch not in cache index: {k}")
        return ks

    report = []
    for ro in ROSTERS:
        old = rd(w1 / ro / "blocks.csv")
        new = rd(w2 / ro / "training_blocks.csv")
        mb = rd(w2 / ro / "matched_blocks.csv")
        te = rd(w2 / ro / "training_epochs.csv")
        ev = rd(w2 / ro / "evaluation_subjects.csv")
        f = lambda rows, arm: sorted((b["subject_id"], b["record_id"], int(b["first_epoch"]), int(b["last_epoch"]), int(b["night_third"])) for b in rows if b["arm"] == arm)
        base_old, base_new = f(old, "base"), f(new, "base")
        if base_old != base_new or len(base_new) != 20:
            fail(f"{ro}: base blocks differ from week one")
        deepen = f(old, "deepen")
        arms_new = {b["arm"] for b in new}
        other_arm = [x for x in arms_new if x != "base"]
        if len(other_arm) != 1:
            fail(f"{ro}: unexpected arms {arms_new}")
        other = f(new, other_arm[0])
        if len(other) != 10 or len(mb) != 10:
            fail(f"{ro}: expected 10 other-night blocks")
        start_rec = {s: r for s, r, *_ in base_new}
        if len(start_rec) != 5:
            fail(f"{ro}: base should have 5 subjects on one record each")
        orig_mb = sorted((m["subject_id"], m["starting_record_id"], int(m["original_first_epoch"]), int(m["original_last_epoch"])) for m in mb)
        if orig_mb != sorted(d[:4] for d in deepen):
            fail(f"{ro}: matched original blocks != week-one deepen blocks")
        oth_mb = sorted((m["subject_id"], m["other_record_id"], int(m["other_first_epoch"]), int(m["other_last_epoch"])) for m in mb)
        if oth_mb != sorted(o[:4] for o in other):
            fail(f"{ro}: matched other blocks != training_blocks other arm")
        per_subj = defaultdict(set)
        cnt = defaultdict(int)
        for m in mb:
            s, sr, orr = m["subject_id"], m["starting_record_id"], m["other_record_id"]
            if start_rec.get(s) != sr:
                fail(f"{ro}: {s} starting record mismatch")
            if orr == sr or not orr.startswith(s):
                fail(f"{ro}: {s} other record invalid {orr}")
            per_subj[s].add(orr)
            cnt[s] += 1
            of, ol = int(m["other_first_epoch"]), int(m["other_last_epoch"])
            block_keys(s, orr, of, ol)
            if of not in grid[orr]:
                fail(f"{ro}: {orr} start {of} not on 60-epoch candidate grid")
            t_orig = third(sr, int(m["original_first_epoch"]))
            t_oth = third(orr, of)
            if not (t_orig == t_oth == int(m["night_third"])):
                fail(f"{ro}: night third mismatch {s} orig {t_orig} other {t_oth} file {m['night_third']}")
            want = "later" if orr > sr else "earlier"
            if m["direction"] != want:
                fail(f"{ro}: direction field wrong for {s}")
        if any(len(v) != 1 for v in per_subj.values()) or any(c != 2 for c in cnt.values()) or set(cnt) != set(start_rec):
            fail(f"{ro}: each start subject must get exactly 2 blocks from one other record")
        keys = []
        for s, r, a1, a2, _ in base_new + other:
            keys += block_keys(s, r, a1, a2)
        if len(set(keys)) != 1800 or len(keys) != 1800:
            fail(f"{ro}: blocks overlap or count != 1800")
        want_rows = sorted(row_of[k] for k in keys)
        got = [(r["subject_id"], r["record_id"], int(r["epoch_index"])) for r in te]
        got_rows = [row_of[k] for k in got]
        if got_rows != want_rows:
            fail(f"{ro}: training_epochs != base + other blocks in cache row order")
        train_subj = set(start_rec)
        ev_subj = {e["subject_id"] for e in ev}
        if train_subj & held or train_subj & ev_subj:
            fail(f"{ro}: training subjects overlap held-out or evaluation subjects")
        report.append(dict(roster=ro, base_blocks=20, other_blocks=10, train_epochs=1800,
                           other_records=";".join(sorted(next(iter(v)) for v in per_subj.values())),
                           thirds_matched=10, eval_subjects=len(ev_subj), status="pass"))
        print(f"{ro}: pass", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "matched_blocks_check_D.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    print("ALL 9 ROSTERS PASS")


if __name__ == "__main__":
    main()
