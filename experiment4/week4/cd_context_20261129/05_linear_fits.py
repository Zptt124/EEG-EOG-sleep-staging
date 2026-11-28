"""D：实验二辅助模型——线性头（只用中心段特征），与 BiGRU 用同一份 D_k1_input.npz、同一批单元。

沿用第一轮 780 拟合的线性头：C 的 historical_head/train.py 的 fit（head seed 0、30 轮、CPU 4 线程）。
只在双方签出后运行：--joint-lock 中 C、D 均已批准，且 cache_D 等于输入的 SHA-256。
每个单元写 <out>/<roster>_<draw>_<strategy>_linear.npz（pred、counts[16,5,5]），已存在则跳过（可续跑）。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--joint-lock", type=Path, required=True)
    ap.add_argument("--linear-src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data_sha = sha(a.input)
    lock = json.loads(a.joint_lock.read_text())
    if set(lock["approved_by"]) != {"C", "D"} or not all(lock["approved_by"].values()) \
            or lock["sha256"].get("cache_D") != data_sha:
        fail("joint lock missing approvals or cache_D does not match the input")
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(a.linear_src))
    from train import fit
    z = np.load(a.input, allow_pickle=False)
    x, ty, mask = z["x"], z["train_y"], z["unit_mask"]
    tc = z["train_windows"][:, z["train_windows"].shape[1] // 2]
    ec = z["eval_windows"][:, z["eval_windows"].shape[1] // 2]
    ex, truth, subj = x[ec], z["truth"], z["subject_index"]
    a.out.mkdir(parents=True, exist_ok=True)
    for i in range(len(mask)):
        ro, dr, st = int(z["roster"][i]), int(z["draw"][i]), str(z["strategy"][i])
        path = a.out / f"{ro:02d}_{dr:02d}_{st}_linear.npz"
        if path.exists():
            continue
        ids = mask[i][mask[i] >= 0]
        model, m, s, _ = fit(x[tc[ids]], ty[ids], 0, 30)
        with torch.inference_mode():
            pred = model(torch.from_numpy((ex - m) / s)).argmax(1).numpy().astype(np.uint8)
        counts = np.bincount(subj * 25 + truth * 5 + pred, minlength=400).reshape(16, 5, 5)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, pred=pred, counts=counts, roster=ro, draw=dr, strategy=st,
                            input_sha256=data_sha, joint_sha256=sha(a.joint_lock))
        tmp.replace(path)
        if i % 50 == 0:
            print(f"{i + 1}/{len(mask)} {path.name}", flush=True)
    print(f"LINEAR DONE | {len(mask)} units", flush=True)


if __name__ == "__main__":
    main()
