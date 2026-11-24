"""W4 公共预训练脚本（全组共用；所有 pretraining 变体都在这个脚本上用开关实现）。

功能
- 训练池 54 人中剔除 held-out 8 人（--heldout-csv，B 给出），只用其余 46 人的无标签信号训练。
- 每 --val-every 步（默认 5k，step 0 也算一次）在 held-out 上算 val MSE：t 每 5 步一段、EEG/EOG 分开，
  窗口/噪声/t 全部固定（见 w4d_train.ValSet）；同时保存该步的权重 ckpt_step<N>.pt（online + EMA）。
- 训练 loss 按模态分开记录（history.csv 的 loss_eeg / loss_eog）。
- 停止规则（D 于 2026-11-14 定稿，之后不改）：
    指标 = EMA 权重、空条件（不给 EEG）下，EEG 与 EOG 各自在 t ∈ [5, 25) 的 val MSE 平均，再等权平均；
    每 5k 步评一次；连续 3 次验证（15k 步）都没有严格低于此前最好值就停；保留最好的 checkpoint（best.pt）；
    上限 240k；30k 之前不触发停止（保证全组共用的 30k 对照一定存在）。
  新方法筛查（固定 30k）用 --no-early-stop --max-steps 30000。
- 开关：--self-cond-layer/--self-cond-pool（self-conditioning），--cross-modal/--cond-dropout/--shift-pairs（跨模态，
  按 C 的规则 experiment3/week4/cross_modal_protocol.md：错时 60–300 秒配对表、dropout 默认 0.2）。
- W5：--window-sec 30 让 backbone 直接吃连续 30 秒段（不切 5 秒窗；模型尺寸、标准化、noise schedule 不变）。
  停止规则仍看 5 秒标准 val 集（全组可比），另把连续 30 秒 val 集写到 val_mse_30s.csv。暂不支持与 --cross-modal 同用。
- 登记：config.json 记录训练窗清单 SHA、名单 SHA、配对表 SHA；checkpoints.csv 记录每个 checkpoint 的 SHA256。
- 断点续训：每 --save-every 步写 latest.pt；中断后用同一条命令重跑即可续上。

输出
  大文件（SLEEP_W4_BIG/runs/<run>/）：ckpt_step<N>.pt、best.pt、latest.pt
  小文件（exp_d_pretrain/runs/<run>/）：config.json、history.csv、val_mse.csv、checkpoints.csv、stop_decision.json
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import w4d_common as C
from w4d_train import Trainer, ValSet

STOP_RULE = dict(
    metric='EMA weights, no EEG condition; mean over {EEG, EOG} of mean val MSE over t in [5, 25)',
    val_every=5000, patience_evals=3, min_steps_before_stop=30000, max_steps=240000,
    improvement='strictly lower than best so far', keep='best checkpoint (best.pt) + every val checkpoint',
    finalized='2026-11-14 by D')


def read_val_rows(path):
    if not path.exists():
        return []
    return C.read_csv(path)


def stop_status(rows, patience, min_steps, step_now):
    """只用 val_mse.csv 里 EMA + 空条件 的行计算停止状态（重跑/续训时可完全复现）。"""
    seen = {}
    for r in rows:
        if r['weights'] == 'ema' and r['eeg_cond'] == '0':
            seen[int(r['step'])] = float(r['stop_metric'])
    best_step, best, since = None, np.inf, 0
    for s in sorted(seen):
        if seen[s] < best:
            best_step, best, since = s, seen[s], 0
        else:
            since += 1
    stop = step_now >= min_steps and since >= patience
    return dict(best_step=best_step, best_metric=best, evals_since_best=since, stop=stop)


def gpu_check(allow_shared):
    free, total = torch.cuda.mem_get_info()
    used_gib = (total - free) / 2**30
    # 本进程自己的 CUDA context 在 WSL 上约占 1.1 GiB，所以阈值取 2 GiB
    print(f'GPU: {torch.cuda.get_device_name(0)}, in use before training (incl. own CUDA context) '
          f'{used_gib:.2f} GiB / {total / 2**30:.1f} GiB', flush=True)
    if used_gib > 2.0 and not allow_shared:
        C.fail(f'GPU already has {used_gib:.2f} GiB in use by another process; '
               'stop it first (or pass --allow-shared-gpu if you are sure)')
    torch.cuda.set_per_process_memory_fraction(0.92)   # 超过就真正报 OOM，而不是悄悄溢出到共享内存


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='m64x12', choices=[k for k in C.MODEL_CONFIGS if k != 'official_ref'])
    p.add_argument('--run-name', required=True)
    p.add_argument('--heldout-csv', required=True, help='CSV with subject_id column (8 held-out subjects)')
    p.add_argument('--max-steps', type=int, default=240000)
    p.add_argument('--seed', type=int, default=0, help='backbone seed')
    p.add_argument('--batch-size', type=int, default=108)
    p.add_argument('--lr', type=float, default=None, help='override general lr (default official 1e-4)')
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--val-every', type=int, default=5000)
    p.add_argument('--patience', type=int, default=3, help='evals without improvement before stopping')
    p.add_argument('--min-steps', type=int, default=30000, help='no early stop before this step')
    p.add_argument('--no-early-stop', action='store_true', help='fixed-budget run (e.g. 30k screening)')
    p.add_argument('--self-cond-layer', type=int, default=0, help='1-based block index m; 0 = off')
    p.add_argument('--self-cond-pool', default='mean', choices=['mean', 'std', 'meanstd'])
    p.add_argument('--cross-modal', default='none', choices=['none', 'same_window', 'shifted'])
    p.add_argument('--cond-dropout', type=float, default=None,
                   help='per-window prob. of dropping the EEG condition (default 0.2 with --cross-modal)')
    p.add_argument('--shift-pairs', default=str(C.PAIRS_NPY), help='shift-pair table for --cross-modal shifted')
    p.add_argument('--window-sec', type=int, default=5, choices=sorted(C.WINDOW_SECS),
                   help='pretraining window: 5 = six 5 s windows per epoch (W4 default); 30 = whole 30 s epoch (W5)')
    p.add_argument('--smoke-small-val', action='store_true',
                   help='SMOKE TEST ONLY: 16 windows per subject per modality instead of 128')
    p.add_argument('--allow-shared-gpu', action='store_true')
    args = p.parse_args()

    if args.max_steps % args.val_every != 0:
        C.fail('--max-steps must be a multiple of --val-every')
    if args.cross_modal == 'none':
        if args.cond_dropout is not None:
            C.fail('--cond-dropout only makes sense with --cross-modal')
        args.cond_dropout = 0.0
    else:
        if args.cond_dropout is None:
            args.cond_dropout = 0.2   # C 的规则默认值
        if not (0 < args.cond_dropout < 1):
            C.fail('--cross-modal requires 0 < --cond-dropout < 1 (team rule 5)')
    if args.window_sec != 5 and args.cross_modal != 'none':
        C.fail('--window-sec 30 is only implemented without --cross-modal')
    if args.smoke_small_val:
        print('WARNING: --smoke-small-val is for smoke tests only; val MSE is NOT comparable with real runs',
              flush=True)

    heldout, ho_info = C.load_heldout(args.heldout_csv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gpu_check(args.allow_shared_gpu)

    big = C.RUNS_BIG / args.run_name
    small = C.RUNS_SMALL / args.run_name
    big.mkdir(parents=True, exist_ok=True)
    small.mkdir(parents=True, exist_ok=True)
    if (small / 'stop_decision.json').exists():
        print(f'Run already finished: {(small / "stop_decision.json").read_text()}', flush=True)
        return

    pool = C.TrainPool(load_to_ram=True, window_sec=args.window_sec)
    units = pool.train_units(heldout)
    n_train_subj = len(set(pool.unit_subject()[units]))
    import hashlib
    units_sha = hashlib.sha256(np.ascontiguousarray(units, dtype='<i8').tobytes()).hexdigest()
    if not (big / 'train_units.npy').exists():
        np.save(big / 'train_units.npy', units)   # 训练窗清单本身（约 11 MB），SHA 见 config.json
    pairs, pairs_info = None, None
    if args.cross_modal == 'shifted':
        pairs, pairs_info = C.load_shift_pairs(pool, args.shift_pairs)
    # 停止规则 / 全组可比的 val MSE 一律用 5 秒标准 val 集（30 秒模型也在同一批 5 秒窗上评）
    val = ValSet(pool.with_window(5), heldout, cross_modal=args.cross_modal, pairs=pairs,
                 per_subject=(16 if args.smoke_small_val else 128))
    # 30 秒模型另评一份连续 30 秒的 val 集（每人每模态 32 段，只看 EMA），写到 val_mse_30s.csv，不参与停止规则
    val30 = None
    if args.window_sec == 30:
        val30 = ValSet(pool, heldout, per_subject=(4 if args.smoke_small_val else 32))

    cfg = dict(run_name=args.run_name, config=args.config, model_kwargs=C.model_kwargs(args.config),
               backbone_seed=args.seed, batch_size=args.batch_size, max_steps=args.max_steps,
               lr_override=args.lr, val_every=args.val_every,
               early_stop=(None if args.no_early_stop else
                           dict(STOP_RULE, val_every=args.val_every, patience_evals=args.patience,
                                min_steps_before_stop=args.min_steps, max_steps=args.max_steps)),
               self_cond=dict(layer=args.self_cond_layer, pool=args.self_cond_pool,
                              method='arXiv 2505.10999 single-shot conditioning residual: '
                                     'cond_l += Linear0(pool(h_m)) * phi(t) for l > m'),
               cross_modal=dict(mode=args.cross_modal, cond_dropout=args.cond_dropout,
                                condition_targets='EOG target windows only; EEG targets always null',
                                shift_pairs=pairs_info,
                                protocol='experiment3/week4/cross_modal_protocol.md (C, 2026-11-14)',
                                eval_condition='val main columns use NO EEG condition'),
               heldout=ho_info, n_train_subjects=n_train_subj, n_train_units=int(len(units)),
               train_units_sha256=units_sha,
               train_units_definition='int64 little-endian array of unit ids u=(row*2+m)*6+k, sorted',
               smoke_small_val=args.smoke_small_val,
               val_set=val.fingerprint,
               noise_scheduler=C.NOISE_SCH_KWARGS, timestep_sampling='uniform integer in [0, 50) per sample',
               denoising_target='v_prediction, MSE', ema=C.EMA_KWARGS,
               optimizer=dict(C.OPT_KWARGS, official_s4_param_groups=True), grad_clip_norm=C.GRAD_CLIP,
               precision='float32 with TF32 matmul',
               conditioning='modality id: 0=EEG Fpz-Cz, 1=EOG horizontal (n_class=2)',
               data=f'train split minus held-out ({n_train_subj} subjects), signal only, labels never read',
               cache_sha256=pool.cfg['sha256_signals'], upstream_commit=C.UPSTREAM_COMMIT)
    if args.window_sec != 5:   # 5 秒 run 的 config.json 保持与 W4 逐字相同
        cfg.update(window_sec=args.window_sec,
                   train_units_definition='int64 little-endian array of unit ids u=row*2+m (whole 30 s epoch), sorted',
                   val_set_note='val_mse.csv = standard 5 s val set (team-comparable, used by the stop rule); '
                                'val_mse_30s.csv = continuous 30 s val set, EMA only',
                   val_set_30s=val30.fingerprint)
    cfg_path = small / 'config.json'
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding='utf-8'))
        if old != json.loads(json.dumps(cfg)):
            C.fail(f'{cfg_path} exists with different settings; use a new --run-name')
    else:
        C.write_json(cfg_path, cfg)
    print(f'train subjects {n_train_subj}, train units {len(units):,}, held-out {heldout}', flush=True)

    tr = Trainer(args.config, pool, units, args.batch_size, args.seed, lr=args.lr,
                 self_cond_layer=args.self_cond_layer, self_cond_pool=args.self_cond_pool,
                 cross_modal=args.cross_modal, cond_dropout=args.cond_dropout, pairs=pairs)
    print(f'params: {C.count_params(tr.model):,}', flush=True)
    latest = big / 'latest.pt'
    if latest.exists():
        tr.load_state(torch.load(latest, map_location='cpu', weights_only=False))
        print(f'RESUMED from step {tr.step} (training epoch {tr.epoch})', flush=True)

    val_csv = small / 'val_mse.csv'

    def run_val():
        """在当前 step 评 val MSE、存 ckpt；已评过的 step（续训重跑时）跳过。返回停止状态。"""
        rows = read_val_rows(val_csv)
        done = {(r['step'], r['weights'], r['eeg_cond']) for r in rows}
        variants = [('ema', 0), ('online', 0)] + ([('ema', 1)] if args.cross_modal != 'none' else [])
        t0 = time.perf_counter()
        for wname, cond in variants:
            if (str(tr.step), wname, str(cond)) in done:
                continue
            net = tr.ema.ema_model if wname == 'ema' else tr.model
            summ = val.summarize(val.evaluate(net, tr.sch, with_eeg_cond=bool(cond)))
            row = dict(step=tr.step, weights=wname, eeg_cond=cond,
                       **{k: round(v, 6) for k, v in summ.items()},
                       wall_time=time.strftime('%Y-%m-%d %H:%M:%S'))
            C.append_csv(val_csv, row)
            if wname == 'ema' and cond == 0:
                print(f'VAL step {tr.step}: stop_metric {summ["stop_metric"]:.5f} '
                      f'(eeg t5-25 {summ["eeg_t05_25"]:.5f}, eog t5-25 {summ["eog_t05_25"]:.5f}; '
                      f'eeg all {summ["eeg_all"]:.5f}, eog all {summ["eog_all"]:.5f})', flush=True)
        if val30 is not None:
            v30_csv = small / 'val_mse_30s.csv'
            if str(tr.step) not in {r['step'] for r in read_val_rows(v30_csv)}:
                summ = val30.summarize(val30.evaluate(tr.ema.ema_model, tr.sch))
                C.append_csv(v30_csv, dict(step=tr.step, weights='ema', eeg_cond=0,
                                           **{k: round(v, 6) for k, v in summ.items()},
                                           wall_time=time.strftime('%Y-%m-%d %H:%M:%S')))
                print(f'VAL30 step {tr.step}: stop_metric {summ["stop_metric"]:.5f} '
                      f'(eeg t5-25 {summ["eeg_t05_25"]:.5f}, eog t5-25 {summ["eog_t05_25"]:.5f})', flush=True)
        ck = big / f'ckpt_step{tr.step}.pt'
        if not ck.exists():
            tmp = big / f'ckpt_step{tr.step}.tmp.pt'
            torch.save(tr.weights(dict(run_config=cfg)), tmp)
            shutil.move(tmp, ck)
            C.append_csv(small / 'checkpoints.csv', dict(step=tr.step, file=ck.name, sha256=C.sha256_file(ck),
                                                         wall_time=time.strftime('%Y-%m-%d %H:%M:%S')))
        st = stop_status(read_val_rows(val_csv), args.patience, args.min_steps, tr.step)
        print(f'  val took {time.perf_counter() - t0:.0f}s; best step {st["best_step"]} '
              f'({st["best_metric"]:.5f}), evals since best {st["evals_since_best"]}', flush=True)
        return st

    def finish(reason, st):
        best_ck = big / f'ckpt_step{st["best_step"]}.pt'
        if not args.no_early_stop:
            shutil.copy2(best_ck, big / 'best.pt')
        dec = dict(reason=reason, stopped_at_step=tr.step, best_step=st['best_step'],
                   best_stop_metric=st['best_metric'], evals_since_best=st['evals_since_best'],
                   best_checkpoint_sha256=C.sha256_file(best_ck),
                   early_stop=not args.no_early_stop, wall_time=time.strftime('%Y-%m-%d %H:%M:%S'))
        C.write_json(small / 'stop_decision.json', dec)
        print(f'PRETRAIN DONE: {dec}', flush=True)

    def save_latest():
        tmp = big / 'latest.tmp.pt'
        torch.save(tr.state(), tmp)
        shutil.move(tmp, latest)

    if tr.step % args.val_every == 0:   # step 0（或恰好在 val 步中断后）补评
        st = run_val()
        save_latest()
        if not args.no_early_stop and st['stop'] and tr.step > 0:
            finish(f'early stop: {st["evals_since_best"]} evals without improvement', st)
            return

    hist = small / 'history.csv'
    buf = []
    t_last, steps_last = time.perf_counter(), tr.step
    baseline = []
    torch.cuda.reset_peak_memory_stats()
    while tr.step < args.max_steps:
        buf.append(tr.train_step())
        if tr.step % args.log_every == 0:
            a = torch.stack(buf).cpu().numpy()
            if not (np.isfinite(a[:, 0]).all() and np.isfinite(a[:, 3]).all()):
                C.fail(f'Non-finite loss/grad before step {tr.step}')
            el = time.perf_counter() - t_last
            sec = el / (tr.step - steps_last)
            row = dict(step=tr.step, training_epoch=round(tr.step * args.batch_size / len(units), 4),
                       loss=round(float(a[:, 0].mean()), 6), loss_eeg=round(float(np.nanmean(a[:, 1])), 6),
                       loss_eog=round(float(np.nanmean(a[:, 2])), 6), grad_norm=round(float(a[:, 3].mean()), 4),
                       sec_per_step=round(sec, 4), peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                       eta_hours=round((args.max_steps - tr.step) * sec / 3600, 2),
                       wall_time=time.strftime('%Y-%m-%d %H:%M:%S'))
            C.append_csv(hist, row)
            print(f'step {tr.step}/{args.max_steps} loss {row["loss"]:.4f} (eeg {row["loss_eeg"]:.4f} '
                  f'eog {row["loss_eog"]:.4f}) grad {row["grad_norm"]:.3f} {sec:.3f}s/step '
                  f'ETA(max) {row["eta_hours"]:.1f}h', flush=True)
            if len(baseline) < 5:
                baseline.append(sec)
            elif sec > 3 * float(np.median(baseline)):
                print(f'WARNING SLOWDOWN: {sec:.3f}s/step vs baseline {np.median(baseline):.3f}s/step '
                      '(GPU memory may be spilling to shared memory; check nvidia-smi)', flush=True)
            buf = []
            t_last, steps_last = time.perf_counter(), tr.step
        if tr.step % args.val_every == 0:
            st = run_val()
            save_latest()
            t_last, steps_last = time.perf_counter(), tr.step   # val 耗时不计入 s/step
            if not args.no_early_stop and st['stop']:
                finish(f'early stop: {st["evals_since_best"]} evals without improvement', st)
                return
        elif tr.step % args.save_every == 0:
            save_latest()
    st = stop_status(read_val_rows(val_csv), args.patience, args.min_steps, tr.step)
    finish('reached max steps', st)


if __name__ == '__main__':
    main()
