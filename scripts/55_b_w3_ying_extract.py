"""B 第 3 周：从 Ying 原始 EDF 与专家标注独立重提 W4 30k 特征。

独立于 C 的 extract.py 实现：
- 专家标注：自己解析 TXT 区间与 CSV 逐段标签，两者逐段比对；三专家多数投票，
  三人都不同时取 Expert 1；任一专家 NotScored 记 -1。
- 对齐：直接读 EDF 头（字节级）拿导联、采样率、记录时长、开始时间与设备字段，
  逐夜比较信号长度与标注长度，不预设哪一夜例外，全部写进审计表。
- 只纳入 Day1、Day2（前两夜）；目录标 “no data” 的夜记为缺失。
信号路径与组内冻结表示一致（这部分必须相同）：F4-A1 与 EOG-R − EOG-L，
500 Hz 整夜 resample_poly(1,5) 到 100 Hz、×1e6 换成 µV，W4 的 transform_uv，
W4 30k EMA 后三层 gate、1 秒 std 池化，每模态 192 维。

用法：
python -I scripts/55_b_w3_ying_extract.py cache/ying_osf --out runs/b_w3_ying \
    --w4-source cache/w4_src_6590ec9/experiment4/week4/src --eegdm-vendor cache/vendor/EEGDM \
    --ckpt <ckpt_step30000.pt> --device mps
加 --labels-only 只做标注与对齐审计。
"""
import argparse
import csv
import hashlib
import importlib
import io
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

CKPT_SHA = "21c8514ea7761ef383d70b3e26ac19ea33ad0e7ccacfe690812ad359b0118e5b"
STAGE = {"Wake": 0, "NREM1": 1, "NREM2": 2, "NREM3": 3, "REM": 4, "NotScored": -1}
EPOCH_S = 30
FS = 500


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def edf_header(raw_bytes_head):
    """解析 EDF 固定头和信号头（只用标准字段）。"""
    h = raw_bytes_head
    ns = int(h[252:256].decode().strip())
    n_records = int(h[236:244].decode().strip())
    rec_dur = float(h[244:252].decode().strip())
    off = 256

    def field(width):
        nonlocal off
        vals = [h[off + i * width: off + (i + 1) * width].decode("latin-1").strip() for i in range(ns)]
        off += ns * width
        return vals

    labels = field(16)
    field(80)  # transducer
    dims = field(8)
    pmin = [float(v) for v in field(8)]
    pmax = [float(v) for v in field(8)]
    dmin = [int(v) for v in field(8)]
    dmax = [int(v) for v in field(8)]
    field(80)  # prefilter
    spr = [int(v) for v in field(8)]
    return {
        "patient": h[8:88].decode("latin-1").strip(),
        "recording": h[88:168].decode("latin-1").strip(),
        "startdate": h[168:176].decode("latin-1").strip(),
        "starttime": h[176:184].decode("latin-1").strip(),
        "header_bytes": int(h[184:192].decode().strip()),
        "n_records": n_records,
        "record_duration_s": rec_dur,
        "labels": labels,
        "units": dims,
        "samples_per_record": spr,
        "physical_min": pmin, "physical_max": pmax, "digital_min": dmin, "digital_max": dmax,
    }


def read_edf_channels(fh, hdr, names, n_samples):
    """按 EDF 标准读取指定导联的前 n_samples 个点，返回伏特（与常用读取器同一换算）。"""
    spr = hdr["samples_per_record"]
    rec_len = sum(spr)
    idx = [hdr["labels"].index(n) for n in names]
    first = [sum(spr[:j]) for j in range(len(spr))]
    need_rec = -(-n_samples // spr[idx[0]])
    assert need_rec <= hdr["n_records"], "signal shorter than requested"
    fh.seek(hdr["header_bytes"])
    raw = np.frombuffer(fh.read(need_rec * rec_len * 2), dtype="<i2").reshape(need_rec, rec_len)
    out = []
    for j in idx:
        assert spr[j] == spr[idx[0]]
        dig = raw[:, first[j]:first[j] + spr[j]].reshape(-1)[:n_samples].astype(np.float64)
        gain = (hdr["physical_max"][j] - hdr["physical_min"][j]) / (hdr["digital_max"][j] - hdr["digital_min"][j])
        phys = (dig - hdr["digital_min"][j]) * gain + hdr["physical_min"][j]
        scale = {"uV": 1e-6, "mV": 1e-3, "V": 1.0}[hdr["units"][j]]
        out.append(phys * scale)
    return np.stack(out)


def parse_expert(archive, stem):
    csv_bytes = archive.read(stem + ".csv")
    txt = archive.read(stem + ".txt").decode("utf-8-sig")
    per_epoch = [int(v) for v in re.split(r"[,\s]+", csv_bytes.decode("utf-8-sig").strip()) if v != ""]
    n = len(per_epoch)
    from_txt = [None] * n
    tail_note = None
    for line in txt.splitlines():
        parts = line.split()
        if not parts:
            continue
        name, lo, hi = parts[0], int(parts[1]), int(parts[2])
        if lo == hi:
            tail_note = line.strip()  # 零长度区间（仅出现在末尾）
            continue
        assert lo % EPOCH_S == 0 and hi % EPOCH_S == 0 and 0 <= lo < hi <= n * EPOCH_S, (stem, line)
        for e in range(lo // EPOCH_S, hi // EPOCH_S):
            assert from_txt[e] is None, (stem, "overlap", e)
            from_txt[e] = STAGE[name]
    assert all(v is not None for v in from_txt), (stem, "gap in TXT intervals")
    assert from_txt == per_epoch, (stem, "CSV/TXT disagree")
    assert set(per_epoch) <= set(STAGE.values()), (stem, set(per_epoch))
    return np.array(per_epoch, dtype=np.int8), tail_note, sha256_bytes(csv_bytes), sha256_bytes(txt.encode())


def consensus(experts):
    n = min(len(e) for e in experts)
    e1, e2, e3 = (e[:n].astype(np.int16) for e in experts)
    out = np.where(e2 == e3, e2, e1)                     # 2、3 一致则取之（含与 1 一致的情况）
    out = np.where((e1 == e2) | (e1 == e3), e1, out)      # 1 与任一一致取 1
    out[(e1 < 0) | (e2 < 0) | (e3 < 0)] = -1
    return out.astype(np.int8), np.stack([e1, e2, e3]).astype(np.int8)


def nights_in(archive):
    names = archive.namelist()
    found = {}
    for m in names:
        mm = re.fullmatch(r"(SLP\d{3})/(SLP\d{3}_Day(\d+))/\2\.edf", m)
        if mm:
            found[int(mm.group(3))] = (mm.group(2), m)
    missing = sorted(int(re.search(r"Day(\d+)", m).group(1)) for m in names
                     if m.endswith("/") and "no data" in m)
    return found, missing


def load_encoder(args):
    import torch
    os.environ["EEGDM_VENDOR"] = str(Path(args.eegdm_vendor).resolve())
    sys.path.insert(0, str(Path(args.w4_source).resolve()))
    w4 = importlib.import_module("04_extract_features")
    digest = hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()
    assert digest == CKPT_SHA, "checkpoint hash mismatch"
    model, _, opts = w4.load_model(Path(args.ckpt), "ema_model")
    model = model.to(args.device)
    for layer in model.layers:
        layer.layer.kernel.forward = lru_cache(maxsize=2)(layer.layer.kernel.forward)
    torch.set_num_threads(args.threads)
    return w4, model, opts


def encode(w4, model, planar_uv, n_epochs, device, batch):
    import torch
    out = []
    for modality in (0, 1):
        x = w4.C.transform_uv(planar_uv[modality].reshape(n_epochs, 3000))
        pieces = []
        for s in range(0, n_epochs, batch):
            t = torch.from_numpy(np.ascontiguousarray(x[s:s + batch, None])).to(device)
            a = w4.gate_activations(model, t, modality)
            std, _ = w4.pooled(a[:, -3:], 1)
            pieces.append(std.cpu().numpy())
        out.append(np.concatenate(pieces))
    f = np.concatenate(out, axis=1).astype(np.float32)
    assert f.shape == (n_epochs, 384) and np.isfinite(f).all()
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--w4-source")
    p.add_argument("--eegdm-vendor")
    p.add_argument("--ckpt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--labels-only", action="store_true")
    p.add_argument("--only", nargs="*", help="只处理这些夜（如 SLP001_Day1）")
    p.add_argument("--feature-dir", default="features")
    args = p.parse_args()

    raw_dir = args.data / "raw"
    feat_dir = args.out / args.feature_dir
    feat_dir.mkdir(parents=True, exist_ok=True)
    w4 = model = None
    if not args.labels_only:

        from scipy.signal import resample_poly
        w4, model, opts = load_encoder(args)
        print("encoder", opts, args.device, flush=True)

    audit, t0 = [], time.monotonic()
    for i in range(1, 21):
        subj = f"SLP{i:03d}"
        with zipfile.ZipFile(raw_dir / f"{subj}.zip") as z:
            found, missing = nights_in(z)
            for day in (1, 2):
                key = f"{subj}_Day{day}"
                if day not in found:
                    audit.append({"night": key, "status": "missing",
                                  "missing_marked_no_data": day in missing,
                                  "available_days": sorted(found)})
                    continue
                if args.only and key not in args.only:
                    continue
                stem_dir = found[day][1].rsplit("/", 1)[0] + "/" + key
                parsed = [parse_expert(z, f"{stem_dir}_Expert{k}") for k in (1, 2, 3)]
                lens = [len(p_[0]) for p_ in parsed]
                y, experts = consensus([p_[0] for p_ in parsed])
                info = z.getinfo(found[day][1])
                with z.open(info) as f:
                    head = f.read(256)
                    ns = int(head[252:256].decode().strip())
                    head += f.read(256 * ns)
                hdr = edf_header(head)
                fs = [s / hdr["record_duration_s"] for s in hdr["samples_per_record"]]
                total_s = hdr["n_records"] * hdr["record_duration_s"]
                row = {
                    "night": key, "status": "used", "available_days": sorted(found),
                    "expert_epochs": lens, "consensus_epochs": len(y),
                    "edf_seconds": total_s, "edf_full_epochs": int(total_s // EPOCH_S),
                    "edf_minus_label_epochs": int(total_s // EPOCH_S) - len(y),
                    "expert_len_spread": max(lens) - min(lens),
                    "tail_zero_length_intervals": [p_[1] for p_ in parsed],
                    "n_notscored": int((y < 0).sum()), "n_scored": int((y >= 0).sum()),
                    "stage_counts": {s: int((y == v).sum()) for s, v in STAGE.items() if v >= 0},
                    "ties_resolved_by_e1": int(((experts[0] != experts[1]) & (experts[0] != experts[2])
                                                & (experts[1] != experts[2]) & (y >= 0)).sum()),
                    "edf_startdate": hdr["startdate"], "edf_starttime": hdr["starttime"],
                    "edf_recording_field": hdr["recording"],
                    "edf_labels": hdr["labels"], "edf_units": hdr["units"], "edf_fs": sorted(set(fs)),
                    "label_sha256": {f"E{k+1}": {"csv": parsed[k][2], "txt": parsed[k][3]} for k in range(3)},
                }
                for ch in ("F4-A1", "EOG-R", "EOG-L"):
                    j = hdr["labels"].index(ch)
                    assert fs[j] == FS, (key, ch, fs[j])
                    row[f"unit_{ch}"] = hdr["units"][j]
                audit.append(row)
                if args.labels_only:
                    (args.out / "labels").mkdir(exist_ok=True)
                    np.savez(args.out / "labels" / f"{key}.npz", y=y, experts=experts)
                    continue
                if (feat_dir / f"{key}.npz").exists():
                    continue
                with z.open(info) as r:
                    buf = io.BytesIO(r.read())
                sig = read_edf_channels(buf, hdr, ["F4-A1", "EOG-R", "EOG-L"], len(y) * EPOCH_S * FS)
                del buf
                sig = sig.astype(np.float32)
                planar = np.stack([sig[0], sig[1] - sig[2]])
                planar = resample_poly(planar, 1, 5, axis=1) * 1e6
                ts = time.monotonic()
                feats = encode(w4, model, planar, len(y), args.device, args.batch)
                tmp_out = feat_dir / f"{key}.partial.npz"
                np.savez(tmp_out, y=y, experts=experts, x=feats)
                tmp_out.replace(feat_dir / f"{key}.npz")
                print(f"{key} epochs={len(y)} {time.monotonic()-ts:.0f}s total={time.monotonic()-t0:.0f}s",
                      flush=True)
    name = "label_audit.json" if not args.only else f"label_audit_{args.feature_dir}.json"
    (args.out / name).write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")
    with (args.out / name.replace(".json", ".csv")).open("w", newline="", encoding="utf-8") as f:
        cols = ["night", "status", "available_days", "expert_epochs", "consensus_epochs", "edf_full_epochs",
                "edf_minus_label_epochs", "n_scored", "n_notscored", "ties_resolved_by_e1",
                "edf_startdate", "edf_starttime", "edf_recording_field", "unit_F4-A1", "unit_EOG-R", "unit_EOG-L"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(audit)
    print("done", flush=True)


if __name__ == "__main__":
    main()
