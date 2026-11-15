"""第 3 步：测速（每个候选模型几百步，总共约 15–40 分钟，GPU）。

对每个候选配置：
  1) 先跑 warmup 步（不计时），再跑计时步，取稳态每步耗时的中位数；
  2) 检查 loss/梯度有限、参数确实在更新（相对变化）、固定诊断集 loss 的前后变化；
  3) 单独测数据读取（gather + 拷到 GPU）的耗时；
  4) 估算完整 training epoch（全部 / 50% 窗口）以及 50/100/200 epochs 的时间。
输出：D:/.../week3-1/exp_d_pretrain/analysis/speed_test/speed_test_summary.csv 和 speed_<config>.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import w3d_common as C
from w3d_train import Trainer, timed_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', default='m64x12,m64x16,m96x16,official_ref')
    p.add_argument('--batch-size', type=int, default=108)    # 与官方相同
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--max-seconds', type=float, default=600)  # 每个配置计时部分的上限
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = C.ANALYSIS / 'speed_test'
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pool = C.TrainPool(load_to_ram=True)
    print(f'pool loaded: epochs={pool.cfg["n_epochs"]} units={pool.n_units} '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)
    full_units = pool.n_units

    # 数据读取耗时（与模型无关）
    tr_io = np.random.default_rng(0)
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(200):
        x, m = pool.gather(tr_io.choice(pool.n_units, args.batch_size, replace=False))
        torch.from_numpy(x).cuda()
        torch.from_numpy(m).cuda()
    torch.cuda.synchronize()
    io_ms = (time.perf_counter() - s) / 200 * 1000
    print(f'data gather+copy per batch: {io_ms:.1f} ms', flush=True)

    rows = []
    for name in args.configs.split(','):
        print(f'=== {name} {C.MODEL_CONFIGS[name]} batch={args.batch_size} ===', flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            tr = Trainer(name, pool, np.arange(pool.n_units), args.batch_size, args.seed)
        except torch.cuda.OutOfMemoryError:
            print(f'  OOM when building {name}; skipped', flush=True)
            continue
        n_params = C.count_params(tr.model)
        d0 = tr.diag_loss()
        p0 = tr.param_vector()
        try:
            timed_steps(tr, args.warmup, log_every=args.warmup)
            times, losses, gnorms = timed_steps(tr, args.steps, log_every=50, max_seconds=args.max_seconds)
        except torch.cuda.OutOfMemoryError:
            print(f'  OOM during training {name}; skipped', flush=True)
            del tr
            continue
        d1 = tr.diag_loss()
        p1 = tr.param_vector()
        rel_change = float((p1 - p0).norm() / p0.norm())
        step_s = float(np.median(times))
        wps = args.batch_size / step_s
        peak = torch.cuda.max_memory_allocated() / 2**20
        est = {}
        for frac in (1.0, 0.5):
            steps_per_epoch = int(full_units * frac) // args.batch_size
            h = steps_per_epoch * step_s / 3600
            est[frac] = dict(steps_per_epoch=steps_per_epoch, hours_per_epoch=h)
        row = dict(config=name, **C.MODEL_CONFIGS[name], n_params=n_params, batch_size=args.batch_size,
                   timed_steps=len(times), step_ms_median=step_s * 1000,
                   step_ms_p90=float(np.percentile(times, 90)) * 1000, windows_per_s=wps,
                   io_ms_per_batch=io_ms, peak_gpu_mib=peak,
                   loss_first50=float(np.mean(losses[:50])), loss_last50=float(np.mean(losses[-50:])),
                   grad_norm_median=float(np.median(gnorms)), grad_finite=True, param_rel_change=rel_change,
                   diag_before=d0['all'], diag_after=d1['all'],
                   diag_after_t_lt10=d1['t_lt10'], diag_after_t_ge30=d1['t_ge30'],
                   steps_per_epoch_full=est[1.0]['steps_per_epoch'], hours_per_epoch_full=est[1.0]['hours_per_epoch'],
                   steps_per_epoch_half=est[0.5]['steps_per_epoch'], hours_per_epoch_half=est[0.5]['hours_per_epoch'])
        for n_ep in (50, 100, 200):
            row[f'hours_{n_ep}ep_full'] = n_ep * est[1.0]['hours_per_epoch']
            row[f'hours_{n_ep}ep_half'] = n_ep * est[0.5]['hours_per_epoch']
        rows.append(row)
        C.write_json(out_dir / f'speed_{name}.json', dict(row=row, diag_before=d0, diag_after=d1,
                                                          losses=losses, grad_norms=gnorms, step_times=times))
        print(f'  params={n_params:,} step={step_s * 1000:.0f} ms  windows/s={wps:.0f}  peak={peak:.0f} MiB  '
              f'param_rel_change={rel_change:.2e}', flush=True)
        print(f'  diag loss {d0["all"]:.4f} -> {d1["all"]:.4f}; '
              f'one full epoch = {est[1.0]["steps_per_epoch"]} steps = {est[1.0]["hours_per_epoch"]:.2f} h '
              f'(50%: {est[0.5]["hours_per_epoch"]:.2f} h)', flush=True)
        if rel_change <= 0:
            C.fail(f'Parameters did not change for {name}')
        del tr
    if not rows:
        C.fail('No config finished')
    C.write_csv(out_dir / 'speed_test_summary.csv', rows)
    print(f'saved {out_dir / "speed_test_summary.csv"}', flush=True)
    print('SPEED TEST DONE', flush=True)


if __name__ == '__main__':
    main()
