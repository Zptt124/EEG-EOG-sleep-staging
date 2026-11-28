"""D：为分层随机组拟合线性头，并在 C 的 16 位开发评估者上记录混淆计数。

- 训练 = 名单的 1,200 段起点 + 随机组 600 段；头 = 第一周 trainer 的 fit（30 轮、batch 128），与 C 完全相同；
- 评估 = C 开发复验的 16 人（dev8 + pretrain_eval8），样本顺序与 C 的 run.py 一致；
- old 窗口：特征和标签都来自共享缓存；head seed 0，名单 1–3 另跑 seed 1、2（与 C 一致）；
- lights_off 窗口：缓存里没有的 epoch 从 --supplement 目录（03 号提取结果）读取特征和标签，无法评分（标签 -1）的段屏蔽；
- 开跑前先重拟 C 的一个头（old：名单 1 第 1 次 add_people seed 0），混淆计数必须与 C 归档逐格相同；
- 每份名单完成后写盘，中断后重跑同一命令自动续上。

用法：python 02_fit_random.py --window old|lights_off --c-src <dir> --cache <共享缓存> --sampler-out <01 输出> \
        [--supplement <03 输出>] --out <输出目录>
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BLOCK = 60
FEATURE_HASH = "392122c602030d8dcfa903a8d5a5f55fd582f0e09dc6d086248c86bac4267bdf"


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
    ap.add_argument("--window", choices=("old", "lights_off"), required=True)
    for n in ("c-src", "cache", "sampler-out", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    ap.add_argument("--supplement", type=Path)
    a = ap.parse_args()
    sys.path.insert(0, str(a.c_src / "experiment3/week1_budget_20261126"))
    from train import fit  # 第一周 trainer，C 的 run.py 也用它

    old_dir = a.c_src / "experiment3/week3_budget_replication_20261128/results"
    plan = json.loads((old_dir / "plan.json").read_text())
    for name, digest in plan["locked_files"].items():
        if sha(old_dir / name) != digest:
            fail(f"C plan file changed: {name}")
    if sha(a.cache / "features_info.json") != FEATURE_HASH:
        fail("shared cache receipt mismatch")
    slock = json.loads((a.sampler_out / "sampler_lock.json").read_text())
    if slock["window"] != a.window or sha(a.sampler_out / "random_blocks.csv") != slock["random_blocks_sha256"]:
        fail("sampler output does not match its lock")

    rows = rd(a.cache / "index.csv")
    arrays = [np.load(a.cache / f"{m}_std.npy", mmap_mode="r") for m in ("eeg", "eog")]
    feat = lambda ids: np.concatenate([arr[ids] for arr in arrays], axis=1).astype(np.float32)
    eval_people = plan["dev8"] + plan["pretrain_eval8"]
    order = {s: i for i, s in enumerate(eval_people)}
    eval_ids = sorted((i for i, r in enumerate(rows) if r["subject_id"] in order),
                      key=lambda i: (order[rows[i]["subject_id"]], i))
    eval_x = feat(eval_ids)
    eval_y = np.array([int(rows[i]["y_true"]) for i in eval_ids])
    eval_s = np.array([order[rows[i]["subject_id"]] for i in eval_ids])
    pos = {(r["record_id"], int(r["epoch_index"])): i for i, r in enumerate(rows)
           if r["role"] == "pretrain_train" and r["subject_id"] not in order}

    supp_x, supp_y = {}, {}
    if a.window == "lights_off":
        if not a.supplement:
            fail("--supplement is required for lights_off")
        for f in sorted(a.supplement.glob("*.npz")):
            z = np.load(f)
            rec = f.stem
            for e, y in zip(z["epoch"], z["y"]):
                supp_y[rec, int(e)] = int(y)
            for e, x in zip(z["missing_epoch"], z["x"]):
                supp_x[rec, int(e)] = x

    def keys_of(blocks):
        return [(b["record_id"], e) for b in blocks for e in range(int(b["first_epoch"]), int(b["first_epoch"]) + BLOCK)]

    def xy(keys):
        xs, ys = [], []
        for k in keys:
            if k in pos:
                i = pos[k]
                y = int(rows[i]["y_true"])
                if a.window == "lights_off" and k in supp_y and supp_y[k] != y:
                    fail(f"hypnogram label disagrees with cache at {k}")
                xs.append(feat([i])[0])
            else:
                if k not in supp_y:
                    fail(f"no label/feature source for {k}")
                y = supp_y[k]
                if y < 0:
                    continue  # 无法评分的段屏蔽（与 C 的固定尝试口径一致）
                xs.append(supp_x[k])
            ys.append(y)
        return np.stack(xs), np.array(ys)

    def run(x, y, seed):
        model, mean, std, curve = fit(x, y, seed, 30)
        with torch.inference_mode():
            pred = model(torch.from_numpy((eval_x - mean) / std)).argmax(dim=1).numpy()
        return np.bincount(eval_s * 25 + eval_y * 5 + pred, minlength=400).reshape(16, 25), curve[-1]["cumulative_updates"]

    # 起点块
    if a.window == "old":
        anchors_src = rd(old_dir / "blocks.csv")
    else:
        anchors_src = rd(a.c_src / "experiment3/window_budget_c_20261128/results/anchor_blocks.csv")
    base_blocks = defaultdict(list)
    for b in anchors_src:
        if b["arm"] == "base":
            base_blocks[int(b["roster"])].append(b)
    torch.set_num_threads(4)

    # 先重拟 C 的一个头，确认流程和 C 逐格一致
    a.out.mkdir(parents=True, exist_ok=True)
    if a.window == "old":
        c_counts = np.load(old_dir / "confusion_counts.npz")["counts"]
        added = [b for b in rd(old_dir / "blocks.csv") if b["roster"] == "1" and b["draw"] == "1" and b["arm"] == "add_people"]
        x, y = xy(sorted(set(keys_of(base_blocks[1]) + keys_of(added)), key=lambda k: pos[k]))
        conf, _ = run(x, y, 0)
        if not np.array_equal(conf, c_counts[0, 0, 0, 2]):
            fail("re-fit of C roster 1 draw 1 add_people does not reproduce C's confusion counts")
        print("sanity: C roster 1 draw 1 add_people re-fit reproduces confusion counts exactly", flush=True)
    else:
        wdir = a.c_src / "experiment3/window_budget_c_20261128/results"
        cz = np.load(wdir / "paired_confusions.npz")
        r0 = int(cz["rosters"][0])
        added = [b for b in rd(wdir / "selected_blocks.csv") if b["roster"] == str(r0) and b["draw"] == "1" and b["arm"] == "add_people"]
        base_keys = keys_of(base_blocks[r0])
        x, y = xy(sorted(set(base_keys + keys_of(added))))
        conf, _ = run(x, y, 0)
        diff = int(np.abs(conf - cz["counts"][0, 0, 1, 2]).sum())
        print(f"sanity: C lights-off roster {r0} draw 1 add_people re-fit, total abs count diff {diff}", flush=True)
        if diff != 0:
            print("WARNING: not bitwise identical (training row order may differ from C); see README", flush=True)

    rblocks = defaultdict(list)
    for b in rd(a.sampler_out / "random_blocks.csv"):
        rblocks[int(b["roster"]), int(b["draw"])].append(b)
    rosters = slock["rosters"]
    for ro in rosters:
        done = a.out / f"roster_{ro:02d}.npz"
        if done.exists():
            continue
        seeds = [0, 1, 2] if (a.window == "old" and ro <= 3) else [0]
        counts = np.zeros((3, 5, 16, 25), dtype=np.int64)
        scorable = np.zeros(5, dtype=np.int64)
        base_keys = keys_of(base_blocks[ro])
        if len(set(base_keys)) != 1200:
            fail(f"roster {ro}: anchor is not 1200 epochs")
        for d in range(1, 6):
            add_keys = keys_of(rblocks[ro, d])
            if len(set(add_keys)) != 600 or set(add_keys) & set(base_keys):
                fail(f"roster {ro} draw {d}: malformed random blocks")
            # 训练样本顺序与 C 一致：old 按缓存行号排序（run.py），lights_off 按 (record, epoch) 排序（fit_compare.py）
            keys = sorted(set(base_keys + add_keys), key=lambda k: pos[k]) \
                if a.window == "old" else sorted(set(base_keys + add_keys))
            x, y = xy(keys)
            scorable[d - 1] = len(y) - 1200
            for s in seeds:
                conf, upd = run(x, y, s)
                counts[s, d - 1] = conf
        tmp = done.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, counts=counts, scorable_added=scorable, seeds=np.array(seeds))
        tmp.replace(done)
        print(f"roster {ro:02d} done | scorable added per draw {scorable.tolist()}", flush=True)
    print("ALL FITS DONE", flush=True)


if __name__ == "__main__":
    main()
