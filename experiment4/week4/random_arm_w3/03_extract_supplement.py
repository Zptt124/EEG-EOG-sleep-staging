"""D：为关灯窗口随机组补提缓存里没有的 epoch 特征（只在抽块锁定后运行）。

做法与 C 的 window_budget_c_20261128/extract.py 相同：
  - 从 PhysioNet 下载 PSG + Hypnogram，按官方 SHA256SUMS 校验；
  - PSG 与 Hypnogram 的起始时钟一致；标签用团队 04_preprocess.build_epoch_grid 读出（无法评分 = -1）；
  - 与缓存重叠的 epoch：标签必须与缓存相同；抽 3 个重叠 epoch 重新提特征，与缓存比对（容差同 C：atol 2e-4, rtol 5e-3）；
  - 特征 = W4 30k EMA、t=1、gate、第 10–12 层、1 秒 std 池化（与共享缓存 09 号脚本完全相同，复用 D 的 04_extract_features）。
每条记录写一个 <out>/<record>.npz（epoch, y, missing_epoch, x, checked_overlap, max_abs_overlap_error），已存在则跳过。

用法：python 03_extract_supplement.py --blocks <random_blocks.csv> --cache <共享缓存> --manifest <dataset_manifest.csv> \
        --w4-src <D W4 src> --preprocess <scripts 目录> --ckpt <W4 30k ckpt> --raw <下载目录> --out <输出目录>
"""
import argparse
import csv
import hashlib
import importlib
import json
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np

BASES = ("https://physionet.org/files/sleep-edfx/1.0.0/",
         "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/")
CHANNELS = ("EEG Fpz-Cz", "EOG horizontal")
BLOCK = 60
LAYERS = (10, 11, 12)
BATCH = 24


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_file(rel, target, digest):
    if target.exists() and sha(target) == digest:
        return
    part = target.with_suffix(".partial")
    for base in BASES * 2:
        try:
            with urlopen(base + rel, timeout=120) as r, part.open("wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            if sha(part) != digest:
                raise ValueError("digest mismatch")
            part.replace(target)
            return
        except (OSError, ValueError) as e:
            print(f"retry {rel}: {e}", flush=True)
    fail(f"cannot download {rel}")


def main():
    ap = argparse.ArgumentParser()
    for n in ("blocks", "cache", "manifest", "w4-src", "preprocess", "ckpt", "raw", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    a.raw.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)

    info = json.loads((a.cache / "features_info.json").read_text())
    if sha(a.ckpt) != info["checkpoint_sha256"]:
        fail("checkpoint does not match the shared cache")
    rows = rd(a.cache / "index.csv")
    shared = defaultdict(dict)
    for i, r in enumerate(rows):
        shared[r["record_id"]][int(r["epoch_index"])] = (i, int(r["y_true"]))
    need = defaultdict(set)
    for b in rd(a.blocks):
        f0 = int(b["first_epoch"])
        need[b["record_id"]].update(range(f0, f0 + BLOCK))
    todo = sorted(r for r, eps in need.items() if any(e not in shared[r] for e in eps))
    print(f"{len(need)} records in random blocks, {len(todo)} need supplemental features", flush=True)

    sums = a.raw / "SHA256SUMS.txt"
    if not sums.exists():
        with urlopen(BASES[0] + "SHA256SUMS.txt", timeout=60) as r:
            sums.write_bytes(r.read())
    hashes = {name: d for d, name in (l.split() for l in sums.read_text().splitlines() if l.strip())}
    man = {r["record_id"]: r for r in rd(a.manifest)}
    files = [(man[r][k]) for r in todo for k in ("psg_file", "hyp_file")]
    pending = [f for f in files if not (a.out / f"{f[:6]}.npz").exists()]
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(lambda f: get_file("sleep-cassette/" + f, a.raw / f, hashes["sleep-cassette/" + f]), pending))
    print("downloads verified", flush=True)

    import mne
    import torch
    sys.path[:0] = [str(a.w4_src), str(a.preprocess)]
    X = importlib.import_module("04_extract_features")
    C = X.C
    pre = importlib.import_module("04_preprocess")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, ck, opts = X.load_model(a.ckpt, "ema_model")
    model = model.cuda()
    d = model.d_model
    cols = slice((LAYERS[0] - 1) * d, LAYERS[-1] * d)
    arrays = [np.load(a.cache / f"{m}_std.npy", mmap_mode="r") for m in ("eeg", "eog")]

    def extract(sig):  # sig [n, 2, 3000] uV
        out = []
        for mi in range(2):
            xa = C.transform_uv(sig[:, mi])
            parts = []
            for s in range(0, len(xa), BATCH):
                with torch.no_grad():
                    act, _ = X.gate_activations(model, torch.from_numpy(xa[s:s + BATCH, None, :]).cuda(), mi, X.DIFFUSION_T)
                    sd, _ = X.pooled(act, 1)
                parts.append(sd[:, cols].cpu().numpy())
            out.append(np.concatenate(parts))
        res = np.concatenate(out, axis=1).astype(np.float32)
        if res.shape[1] != 384 or not np.isfinite(res).all():
            fail("bad extracted features")
        return res

    worst = 0.0
    for n_done, rec in enumerate(todo, 1):
        dest = a.out / f"{rec}.npz"
        if dest.exists():
            continue
        psg, hyp = a.raw / man[rec]["psg_file"], a.raw / man[rec]["hyp_file"]
        with psg.open("rb") as f:
            header = f.read(256)
        with hyp.open("rb") as f:
            hclock = f.read(184)[168:184]
        if hclock != header[168:184]:
            fail(f"{rec}: PSG and hypnogram clocks differ")
        epochs = np.array(sorted(need[rec]), dtype=np.int64)
        grid = pre.build_epoch_grid(mne.read_annotations(hyp))
        y = np.array([pre.LABEL_MAP.get(grid[e], -1) if e < len(grid) else -1 for e in epochs], dtype=np.int64)
        raw = mne.io.read_raw_edf(psg, include=list(CHANNELS), preload=False, verbose="ERROR")
        if raw.info["sfreq"] != 100 or (epochs[-1] + 1) * 3000 > raw.n_times:
            fail(f"{rec}: unexpected sampling rate or blocks beyond signal end")
        overlap = [i for i, e in enumerate(epochs) if int(e) in shared[rec]]
        for i in overlap:
            if y[i] != shared[rec][int(epochs[i])][1]:
                fail(f"{rec}: hypnogram label disagrees with cache at epoch {epochs[i]}")
        check = [overlap[j] for j in sorted({0, len(overlap) // 2, len(overlap) - 1})] if overlap else []
        missing = [i for i, e in enumerate(epochs) if int(e) not in shared[rec]]
        idx = sorted(set(check + missing))
        ch = [raw.ch_names.index(c) for c in CHANNELS]
        sig = np.stack([raw.get_data(picks=ch, start=int(epochs[i]) * 3000, stop=(int(epochs[i]) + 1) * 3000) * 1e6
                        for i in idx]).astype(np.float32)
        raw.close()
        feats = dict(zip(idx, extract(sig)))
        delta = 0.0
        for i in check:
            ci = shared[rec][int(epochs[i])][0]
            ref = np.concatenate([arr[ci] for arr in arrays])
            delta = max(delta, float(np.abs(ref - feats[i]).max()))
            if not np.allclose(ref, feats[i], atol=2e-4, rtol=5e-3):
                fail(f"{rec}: feature replay mismatch at epoch {epochs[i]} (max abs {delta:.3g})")
        worst = max(worst, delta)
        x = np.stack([feats[i] for i in missing]) if missing else np.empty((0, 384), np.float32)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, epoch=epochs, y=y, missing_epoch=epochs[missing], x=x,
                 checked_overlap=len(check), max_abs_overlap_error=delta)
        tmp.replace(dest)
        print(f"{n_done}/{len(todo)} {rec}: {len(epochs)} epochs, {len(missing)} supplemental, "
              f"{int((y < 0).sum())} unscorable, replay max abs {delta:.3g}", flush=True)
    print(f"EXTRACTION DONE | worst replay error {worst:.3g}", flush=True)


if __name__ == "__main__":
    main()
