"""第 4 步：固定预算短训（GPU，后台长时间运行，支持断点续训）。

- 开训前先保存 step 0 权重 = random-init（与预训练同一初始化、同一 backbone seed）。
- 只在事先固定的 milestone 步数保存权重（--milestones），最后一个 milestone = 总步数 = 主比较用的 checkpoint；
  中间 milestone 只作探索性参考。
- 每 --save-every 步写一次断点（latest.pt），意外中断后用同一条命令重新运行即可从断点继续。
- 每 --log-every 步把 loss、梯度、速度写到 D: 上的 history.csv；每个 milestone 记录固定诊断集 loss。
输出：
  WSL：~/week3-1/runs/<run_name>/ckpt_step<N>.pt（权重）、latest.pt（断点）
  D:  week3-1/exp_d_pretrain/runs/<run_name>/ config.json、history.csv、milestones.csv、units_summary.json
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import w3d_common as C
from w3d_train import Trainer


def append_csv(path, row):
    new = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, choices=[k for k in C.MODEL_CONFIGS if k != 'official_ref'])
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--milestones', required=True, help='comma separated steps, must include 0 and max-steps')
    p.add_argument('--fraction', type=float, default=1.0, help='per-subject fixed window fraction')
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--seed', type=int, default=0, help='backbone seed')
    p.add_argument('--batch-size', type=int, default=108)
    p.add_argument('--lr', type=float, default=None, help='override general lr (default official 1e-4)')
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--run-name', required=True)
    p.add_argument('--init-only', action='store_true', help='only save step-0 random-init checkpoint, then exit')
    args = p.parse_args()

    milestones = sorted({int(s) for s in args.milestones.split(',')})
    if milestones[0] != 0 or milestones[-1] != args.max_steps:
        C.fail('milestones must start with 0 and end with max-steps')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    big = C.RUNS_BIG / args.run_name
    small = C.EXP_D / 'runs' / args.run_name
    big.mkdir(parents=True, exist_ok=True)
    small.mkdir(parents=True, exist_ok=True)

    cfg = dict(run_name=args.run_name, config=args.config, model_kwargs=C.model_kwargs(args.config),
               backbone_seed=args.seed, batch_size=args.batch_size, max_steps=args.max_steps,
               milestones=milestones, primary_checkpoint=f'step{args.max_steps} EMA weights (fixed before training)',
               window_fraction=args.fraction, sample_seed=args.sample_seed,
               noise_scheduler=C.NOISE_SCH_KWARGS, timestep_sampling='uniform integer in [0, 50) per sample',
               denoising_target='v_prediction, MSE', ema=C.EMA_KWARGS,
               optimizer=dict(C.OPT_KWARGS, lr_override=args.lr, official_s4_param_groups=True),
               grad_clip_norm=C.GRAD_CLIP, precision='float32 with TF32 matmul',
               conditioning='modality id: 0=EEG Fpz-Cz, 1=EOG horizontal (n_class=2)',
               data='train split only (54 subjects), signal only, labels never read',
               sampling='each training epoch = random permutation of selected units; unit=(30s epoch, modality, 5s window)',
               team_main_commit=C.TEAM_MAIN_COMMIT, upstream_commit=C.UPSTREAM_COMMIT)
    cfg_path = small / 'config.json'
    if cfg_path.exists():
        import json
        old = json.loads(cfg_path.read_text(encoding='utf-8'))
        if old != cfg:
            C.fail(f'{cfg_path} exists with different settings; use a new --run-name')
    else:
        C.write_json(cfg_path, cfg)

    pool = C.TrainPool(load_to_ram=True)
    units = pool.units_for_fraction(args.fraction, args.sample_seed)
    unit_subject = np.repeat(pool.subject, 2 * C.N_WIN)[units]
    summ = dict(n_units_selected=int(len(units)), n_units_total=int(pool.n_units),
                fraction=args.fraction, sample_seed=args.sample_seed,
                n_eeg=int(((units // C.N_WIN) % 2 == 0).sum()), n_eog=int(((units // C.N_WIN) % 2 == 1).sum()),
                n_subjects=int(len(set(unit_subject))),
                steps_per_training_epoch=int(len(units) // args.batch_size),
                training_epochs_covered_at_max_steps=args.max_steps * args.batch_size / len(units),
                cache_sha256=pool.cfg['sha256_signals'])
    C.write_json(small / 'units_summary.json', summ)
    if args.fraction < 1.0:
        np.save(big / 'selected_units.npy', units)
    print(f'units: {summ}', flush=True)

    tr = Trainer(args.config, pool, units, args.batch_size, args.seed, lr=args.lr)
    print(f'params: {C.count_params(tr.model):,}', flush=True)
    latest = big / 'latest.pt'
    if latest.exists():
        tr.load_state(torch.load(latest, map_location='cpu', weights_only=False))
        print(f'RESUMED from step {tr.step} (training epoch {tr.epoch})', flush=True)

    def save_milestone():
        path = big / f'ckpt_step{tr.step}.pt'
        if path.exists():
            return
        d_on, d_ema = tr.diag_loss(False), tr.diag_loss(True)
        torch.save(tr.weights(), path)
        row = dict(step=tr.step, training_epoch=round(tr.step * args.batch_size / len(units), 4),
                   wall_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                   **{f'diag_online_{k}': round(v, 6) for k, v in d_on.items()},
                   **{f'diag_ema_{k}': round(v, 6) for k, v in d_ema.items()},
                   sha256=C.sha256_file(path))
        append_csv(small / 'milestones.csv', row)
        print(f'MILESTONE step {tr.step}: diag online {d_on["all"]:.4f} ema {d_ema["all"]:.4f} -> {path}', flush=True)

    if tr.step == 0:
        save_milestone()   # random-init
        torch.save(tr.state(), latest)
    if args.init_only:
        print('INIT ONLY DONE (rerun the same command without --init-only to start training)', flush=True)
        return

    hist = small / 'history.csv'
    losses, gnorms = [], []
    t_last, steps_last = time.perf_counter(), tr.step
    torch.cuda.reset_peak_memory_stats()
    while tr.step < args.max_steps:
        loss, gn = tr.train_step()
        losses.append(loss)
        gnorms.append(gn)
        if tr.step % args.log_every == 0:
            l = torch.stack(losses).cpu().numpy()
            g = torch.stack(gnorms).cpu().numpy()
            if not (np.isfinite(l).all() and np.isfinite(g).all()):
                C.fail(f'Non-finite loss/grad before step {tr.step}')
            el = time.perf_counter() - t_last
            sps = (tr.step - steps_last) / el
            row = dict(step=tr.step, training_epoch=round(tr.step * args.batch_size / len(units), 4),
                       loss=round(float(l.mean()), 6), grad_norm=round(float(g.mean()), 4),
                       sec_per_step=round(1 / sps, 4),
                       peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                       eta_hours=round((args.max_steps - tr.step) / sps / 3600, 2),
                       wall_time=time.strftime('%Y-%m-%d %H:%M:%S'))
            append_csv(hist, row)
            print(f'step {tr.step}/{args.max_steps} loss {row["loss"]:.4f} grad {row["grad_norm"]:.3f} '
                  f'{row["sec_per_step"]:.3f}s/step ETA {row["eta_hours"]:.1f}h', flush=True)
            losses, gnorms = [], []
            t_last, steps_last = time.perf_counter(), tr.step
        if tr.step in milestones:
            save_milestone()
        if tr.step % args.save_every == 0 or tr.step == args.max_steps:
            tmp = big / 'latest.tmp.pt'
            torch.save(tr.state(), tmp)
            shutil.move(tmp, latest)
    print('PRETRAIN DONE', flush=True)


if __name__ == '__main__':
    main()
