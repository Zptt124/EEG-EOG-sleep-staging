"""D：选块锁定之后，导出 make_d_input.py 需要的中心标签表 D_locked_center_labels.csv。

只在 lock.json 存在、且 D_blocks.csv 的 SHA-256 与 lock.json 一致时运行。
标签来源：共享缓存 index.csv 的 y_true；缓存外的 epoch 用 PhysioNet hypnogram + 团队 04_preprocess.build_epoch_grid
（无法评分 = -1）。缓存内外重叠处做一致性核对。
"""
import argparse
import csv
import hashlib
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for n in ("lock-dir", "anchor", "cache", "manifest", "preprocess", "raw", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    lock = json.loads((a.lock_dir / "lock.json").read_text())
    blocks = a.lock_dir / "D_blocks.csv"
    if sha(blocks) != lock["outputs"]["D_blocks.csv"]:
        fail("D_blocks.csv differs from lock.json")
    need = set()
    for r in rd(a.anchor) + rd(blocks):
        f0 = int(r["first_epoch"])
        need.update((r["record_id"], e) for e in range(f0, f0 + 60))
    cache = {}
    for r in rd(a.cache / "index.csv"):
        k = (r["record_id"], int(r["epoch_index"]))
        if k in need:
            cache[k] = int(r["y_true"])
    import mne
    sys.path.insert(0, str(a.preprocess))
    pre = importlib.import_module("04_preprocess")
    man = {r["record_id"]: r for r in rd(a.manifest)}
    by_rec = defaultdict(list)
    for k in need:
        by_rec[k[0]].append(k[1])
    labels, checked = {}, 0
    for rec in sorted(by_rec):
        eps = by_rec[rec]
        if all((rec, e) in cache for e in eps):
            labels.update({(rec, e): cache[(rec, e)] for e in eps})
            continue
        hyp = a.raw / man[rec]["hyp_file"]
        if not hyp.exists():
            fail(f"{rec}: hypnogram not downloaded ({hyp.name})")
        grid = pre.build_epoch_grid(mne.read_annotations(hyp))
        for e in eps:
            y = pre.LABEL_MAP.get(grid[e], -1) if e < len(grid) else -1
            if (rec, e) in cache:
                checked += 1
                if cache[(rec, e)] != y:
                    fail(f"{rec} epoch {e}: hypnogram label {y} != cache {cache[(rec, e)]}")
            labels[(rec, e)] = y
    rows = [dict(record_id=r, epoch_index=e, y_true=labels[(r, e)]) for r, e in sorted(labels)]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["record_id", "epoch_index", "y_true"])
        w.writeheader()
        w.writerows(rows)
    n_un = sum(r["y_true"] < 0 for r in rows)
    print(f"{len(rows)} center labels, {n_un} unscorable, {checked} cache-overlap labels verified | "
          f"sha256 {sha(a.out)}", flush=True)


if __name__ == "__main__":
    main()
