"""W4 D 06：对任意 W4 checkpoint 重算 held-out val MSE（与 01_pretrain.py 训练时的 val 完全同一口径）。

用途：组员在别的电脑上训练时，val 噪声由 CPU 上的 torch.randn 生成，不同 CPU 在浮点末位可能不同
（噪声 SHA256 不同，但配方和 seed 相同）。这里在 D 的电脑上用同一份噪声（SHA256 前缀 3465eadc）重算，
让三组的 val MSE 逐位同口径。

- 模型开关（self-cond、cross-modal）从 checkpoint 的 run_config 读出（与 04_extract_features.py 相同）。
- 主口径 eeg_cond=0（不给 EEG）；跨模态模型另算 eeg_cond=1（给 EEG，仅 EOG 目标窗），同窗/错时由 run_config 决定。
- 自检：--check-csv 给出训练时的 val_mse.csv 时，比较同一 step、weights、eeg_cond 的各列差值。

输出：一个 CSV，列与 01_pretrain.py 的 val_mse.csv 相同，另加 source 列。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import w4d_common as C
from w4d_train import ValSet
import importlib.util

_spec = importlib.util.spec_from_file_location('w4_extract', Path(__file__).resolve().parent / '04_extract_features.py')
_ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ex)

REF_NOISE_SHA = '3465eadc841288d1cebe5e4d5e046d90c19313088bbc7c454cec3720fde3a120'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='W4 checkpoint (ckpt_step<N>.pt)')
    p.add_argument('--heldout-csv', required=True)
    p.add_argument('--weights', choices=['ema_model', 'model'], default='ema_model')
    p.add_argument('--out', required=True, help='output CSV')
    p.add_argument('--shift-pairs', default=str(C.PAIRS_NPY), help='needed only for shifted cross-modal models')
    p.add_argument('--check-csv', help='training-time val_mse.csv to compare against (self-check)')
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        C.fail(f'Checkpoint not found: {ckpt}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, ck, opts = _ex.load_model(ckpt, args.weights)
    model = model.cuda()
    step = int(ck.get('step'))
    run_name = (ck.get('run_config') or {}).get('run_name') or ckpt.parent.name
    print(f'checkpoint {ckpt.name} run {run_name} step {step} weights {args.weights} switches {opts}', flush=True)

    heldout, _ = C.load_heldout(args.heldout_csv)
    pool = C.TrainPool(load_to_ram=True)
    pairs = None
    if opts['cross_modal'] == 'shifted':
        pairs, _ = C.load_shift_pairs(pool, Path(args.shift_pairs))
    val = ValSet(pool, heldout, cross_modal=opts['cross_modal'], pairs=pairs)
    fp = val.fingerprint
    if fp['noise_sha256'] != REF_NOISE_SHA:
        print(f'WARNING: val noise SHA256 {fp["noise_sha256"][:8]} differs from the team reference '
              f'{REF_NOISE_SHA[:8]} (this machine generates slightly different CPU noise)', flush=True)
    print(f'val set: units {fp["units_sha256"][:8]} t {fp["t_sha256"][:8]} noise {fp["noise_sha256"][:8]}', flush=True)

    sch = C.make_scheduler()
    wname = 'ema' if args.weights == 'ema_model' else 'online'
    conds = [0] + ([1] if opts['cross_modal'] != 'none' else [])
    rows = []
    for cond in conds:
        t0 = time.perf_counter()
        summ = val.summarize(val.evaluate(model, sch, with_eeg_cond=bool(cond)))
        rows.append(dict(step=step, weights=wname, eeg_cond=cond, **{k: round(v, 6) for k, v in summ.items()},
                         wall_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                         source=f'recomputed by 06_val_mse_ckpt.py, noise {fp["noise_sha256"][:8]}'))
        print(f'VAL {run_name} step {step} eeg_cond {cond}: stop_metric {summ["stop_metric"]:.6f} '
              f'(eeg t5-25 {summ["eeg_t05_25"]:.6f}, eog t5-25 {summ["eog_t05_25"]:.6f}) '
              f'[{time.perf_counter() - t0:.0f}s]', flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}', flush=True)

    if args.check_csv:
        ref = [r for r in csv.DictReader(open(args.check_csv, encoding='utf-8')) if r['step'] == str(step)
               and r['weights'] == wname]
        if not ref:
            C.fail(f'No row for step {step} weights {wname} in {args.check_csv}')
        for r in rows:
            m = [x for x in ref if x['eeg_cond'] == str(r['eeg_cond'])]
            if not m:
                continue
            keys = [k for k in r if k.startswith(('eeg_', 'eog_', 'stop'))]
            diff = max(abs(float(m[0][k]) - r[k]) for k in keys)
            print(f'CHECK vs training csv, eeg_cond {r["eeg_cond"]}: max abs diff {diff:.2e} '
                  f'(stop_metric {float(m[0]["stop_metric"]):.6f} -> {r["stop_metric"]:.6f})', flush=True)
    print('VAL RECOMPUTE DONE', flush=True)


if __name__ == '__main__':
    main()
