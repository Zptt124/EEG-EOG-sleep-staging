"""D：来源匹配随机对照的 BiGRU K1 拟合（主模型，head seed 0，与正式运行相同的 C 版 fit/infer）。

匹配对照不在共同签出的缓存清单内（触发后才生成），所以不走 train_bigru.py 的签出检查，而是直接调用其中的
validate / fit / infer，并在每个结果里记录输入 SHA-256 与共同签出 SHA-256，输出格式与正式结果相同。
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for n in ("c-src", "input", "joint-lock", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.c_src))
    import train_bigru as tb
    dev = tb.configure("cuda")
    data_sha, run_sha = sha(a.input), sha(a.joint_lock)
    z = np.load(a.input, allow_pickle=False)
    data = {k: z[k] for k in z.files if k != "old_eval_windows"}
    radius = tb.validate(data)
    span = slice(radius - 1, radius + 2)
    a.out.mkdir(parents=True, exist_ok=True)
    for i in range(len(data["unit_mask"])):
        ro, dr, st = int(data["roster"][i]), int(data["draw"][i]), str(data["strategy"][i])
        path = a.out / f"{ro:02d}_{dr:02d}_{st}_k1_seed0.npz"
        if path.exists():
            continue
        sup = data["unit_mask"][i][data["unit_mask"][i] >= 0]
        model, mean, std = tb.fit(data["x"][data["train_windows"][sup, span]], data["train_y"][sup], 0, False, dev)
        pred, pr = tb.infer(model, mean, std, data["x"], data["eval_windows"][:, span], dev)
        counts = np.bincount(data["subject_index"] * 25 + data["truth"] * 5 + pred, minlength=400).reshape(16, 5, 5)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, pred=pred, probabilities=pr, counts=counts, roster=ro, draw=dr, strategy=st, k=1,
                            seed=0, balanced=False, joint_sha256=run_sha, input_sha256=data_sha)
        tmp.replace(path)
        print(f"{path.stem} done", flush=True)
    print("MATCHED DONE", flush=True)


if __name__ == "__main__":
    main()
