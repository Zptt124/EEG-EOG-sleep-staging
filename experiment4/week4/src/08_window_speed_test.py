"""W5 D 08：5 秒 / 连续 30 秒预训练的显存与步时实测（同一台机器、同一模型 m64x12、fp32 + TF32）。

对每个 (window_sec, batch) 组合：建 Trainer（与 01_pretrain 相同的训练步），先跑 --warmup 步，再计时 --steps 步，
记录峰值显存、稳定 s/step、每步信号量（秒）、每小时处理的信号小时数。显存上限 92%，超过就记为 OOM（不会悄悄溢出）。
输出：analysis/window_speed/speed_<tag>.csv
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import w4d_common as C
from w4d_train import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--heldout-csv', required=True)
    p.add_argument('--combos', required=True, help='comma list of window_sec:batch, e.g. 5:108,30:12,30:9')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--tag', default='v01')
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    free, total = torch.cuda.mem_get_info()
    if (total - free) / 2**30 > 2.0:
        C.fail('GPU already in use by another process')
    torch.cuda.set_per_process_memory_fraction(0.92)
    heldout, _ = C.load_heldout(args.heldout_csv)
    base = C.TrainPool(load_to_ram=True)
    rows = []
    for item in args.combos.split(','):
        ws, bs = (int(v) for v in item.split(':'))
        pool = base.with_window(ws)
        units = pool.train_units(heldout)
        row = dict(window_sec=ws, batch=bs, signal_sec_per_step=ws * bs, status='ok')
        tr = None
        try:
            tr = Trainer('m64x12', pool, units, bs, 0)
            for _ in range(args.warmup):
                tr.train_step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(args.steps):
                tr.train_step()
            torch.cuda.synchronize()
            sec = (time.perf_counter() - t0) / args.steps
            row.update(sec_per_step=round(sec, 4), peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                       signal_hours_per_gpu_hour=round(ws * bs / sec, 1),
                       hours_for_30k_steps=round(30000 * sec / 3600, 2),
                       hours_for_same_signal_as_5s_30k=round(30000 * 5 * 108 / (ws * bs) * sec / 3600, 2))
        except torch.OutOfMemoryError:
            row['status'] = 'OOM'
        tr = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(row, flush=True)
        rows.append(row)
    out = C.EXP_D / 'analysis' / 'window_speed'
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    C.write_csv(out / f'speed_{args.tag}.csv', [{k: r.get(k, '') for k in keys} for r in rows])
    print('SPEED TEST DONE', flush=True)


if __name__ == '__main__':
    main()
