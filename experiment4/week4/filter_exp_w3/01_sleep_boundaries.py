"""D：为条件 B 计算每条记录的分期辅助睡眠边界（与团队 04_preprocess.py 的裁剪规则完全相同）。

规则：在整条记录的分期网格上，找第一个和最后一个睡眠段（N1、N2、N3/4、REM，即 '1','2','3','4','R'），
lo = max(0, 首睡眠段 − 60)，hi = min(网格长度, 末睡眠段 + 60 + 1)（左闭右开，单位为 30 秒段）。
运动（M）和未评分（?）不参与定位；区间内的清醒段保留。
直接调用团队 scripts/04_preprocess.py 的 build_epoch_grid 和 SLEEP_STAGES，避免另写一套。
另与共享缓存核对：缓存中该记录的最小 / 最大 epoch 应落在 [lo, hi) 内（缓存另删了 M/? 段，所以只核对包含关系）。

用法：python 01_sleep_boundaries.py --records <记录列表 txt> --raw <Sleep-EDF 原始目录> --manifest <dataset_manifest.csv> \
        --preprocess <scripts 目录> --cache <共享缓存> --out <csv>
"""
import argparse
import csv
import hashlib
import importlib
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

BASES = ("https://physionet.org/files/sleep-edfx/1.0.0/",
         "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/")
MARGIN = 60


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for n in ("records", "raw", "manifest", "preprocess", "cache", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    records = [l.strip() for l in a.records.read_text().splitlines() if l.strip()]
    man = {r["record_id"]: r for r in csv.DictReader(open(a.manifest, encoding="utf-8-sig"))}
    sums = a.raw / "SHA256SUMS.txt"
    hashes = {n: d for d, n in (l.split() for l in sums.read_text().splitlines() if l.strip())}
    for rec in records:
        hyp = a.raw / man[rec]["hyp_file"]
        key = "sleep-cassette/" + hyp.name
        if hyp.exists() and sha(hyp) == hashes[key]:
            continue
        for base in BASES:
            try:
                with urlopen(base + key, timeout=60) as r, open(hyp, "wb") as f:
                    shutil.copyfileobj(r, f)
                if sha(hyp) == hashes[key]:
                    break
            except OSError as e:
                print(f"retry {key}: {e}", flush=True)
        if sha(hyp) != hashes[key]:
            fail(f"cannot verify {key}")
    import mne
    sys.path.insert(0, str(a.preprocess))
    pre = importlib.import_module("04_preprocess")
    cache_ep = {}
    for r in csv.DictReader(open(a.cache / "index.csv")):
        if r["record_id"] in records:
            cache_ep.setdefault(r["record_id"], []).append(int(r["epoch_index"]))
    rows = []
    for rec in records:
        grid = pre.build_epoch_grid(mne.read_annotations(a.raw / man[rec]["hyp_file"]))
        sleep = [i for i, s in enumerate(grid) if s in pre.SLEEP_STAGES]
        if not sleep:
            fail(f"{rec}: no sleep epochs")
        lo = max(0, sleep[0] - MARGIN)
        hi = min(len(grid), sleep[-1] + MARGIN + 1)
        ce = cache_ep.get(rec, [])
        if ce and not (lo <= min(ce) and max(ce) < hi):
            fail(f"{rec}: cache epochs [{min(ce)}, {max(ce)}] outside boundary [{lo}, {hi})")
        rows.append(dict(record_id=rec, grid_len=len(grid), first_sleep=sleep[0], last_sleep=sleep[-1],
                         b_lo=lo, b_hi_excl=hi, cache_min=min(ce) if ce else "", cache_max=max(ce) if ce else "",
                         cache_matches_boundary=int(bool(ce) and min(ce) == lo and max(ce) == hi - 1)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} records; cache min/max equal boundary in {sum(r['cache_matches_boundary'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
