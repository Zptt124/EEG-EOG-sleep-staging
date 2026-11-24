"""W4 提特征：用冻结的 W4 backbone 提取下游 probe 特征（GPU，每个 checkpoint 约 10–30 分钟）。

适用于公共预训练脚本训出的所有模型（无 EEG / 同窗 / 错时 / self-cond），也能读第三周的 checkpoint。
读出配方沿用 W3（C 的跨模态规则：三组统一）：
  30 秒段 -> 200 Hz、uV/100、staged mu-law -> 折成 6 个 5 秒窗
  diffusion_t=1、不加噪、query='gate'、条件 = 模态编号（EEG 0 / EOG 1）、全部层
  每 --pool-sec 秒求 std（torch correction=1）和 mean -> 在 30/pool_sec 个位置上取均值 -> 每个模态 n_layer*d_model 维
跨模态模型一律用**空条件**提特征（与训练时 condition dropout 的空分支相同，含"无条件"flag 嵌入）：
  提 EOG 特征时只读 NPZ 的 EOG 通道，从不读 EEG。--verify-eog-only 会把 EEG 通道换成 0 再提一遍，逐位比对。
为什么不用官方 LatentActivityExtractor：它自己走一遍各层，不经过 W4 新增的 flag 嵌入和 self-cond 反馈，
  对跨模态 / self-cond 模型会得到和训练时空条件不一致的特征。不开开关的模型两者结果相同（已核对）。

只提取 probe 需要的人：label25 的 14 人（训练侧，含 label10 的 5 人）+ 8 个 dev；测试集不碰。
输出：SLEEP_W4_BIG/features/<run>/step<N>_<weights>_pool<k>s/{eeg,eog}_{std,mean}.npy、index.csv、features_info.json、DONE
     格式与第三周 05_extract_features.py 相同（B 的 27 号 probe 脚本可直接读）。
     self-cond 模型另存 {eeg,eog}_feedback.npy：第 m 层反馈特征 s_m（6 个 5 秒窗取平均），即论文里的表征。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from einops import rearrange

import w4d_common as C
from w4d_model import build_model, pool_time

DIFFUSION_T = 1


def load_model(ckpt_path, weights):
    """按 checkpoint 里记录的开关重建模型（第三周 checkpoint 没有 run_config，视为不开开关）。"""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    rc = ck.get('run_config') or {}
    sc = rc.get('self_cond') or {}
    xm = rc.get('cross_modal') or {}
    opts = dict(self_cond_layer=int(sc.get('layer') or 0), self_cond_pool=sc.get('pool') or 'mean',
                cross_modal=xm.get('mode') or 'none')
    cfg = ck.get('cfg_name') or 'm64x12'
    if ck['model_kwargs'] != C.model_kwargs(cfg):
        C.fail(f'model_kwargs in {ckpt_path.name} do not match config {cfg}')
    model = build_model(cfg, 0, **opts)
    model.load_state_dict(ck[weights], strict=True)
    model.requires_grad_(False).eval()
    model.window_sec = int(rc.get('window_sec') or 5)   # W5：按训练时的窗口长度读出（30 秒模型整段送入）
    return model, ck, opts


@torch.no_grad()
def gate_activations(model, x, modality, diffusion_t=DIFFUSION_T):
    """x [B,1,6000]（单一模态的 30 秒段）-> (gate [B, n_layer, 6000, d_model], feedback [B, D] 或 None)。空 EEG 条件。

    feedback：self-cond 模型第 m 层的池化反馈特征 s_m，在 6 个 5 秒窗上取平均（下游 probe 用）。

    与官方 LatentActivityExtractor(diffusion_t=1, query=['gate']) 的计算顺序相同，另外加上
    W4 的空条件 flag 嵌入（跨模态模型）和 self-cond 反馈（self-cond 模型）。
    """
    B = x.shape[0]
    nw = C.WINDOW_SECS[getattr(model, 'window_sec', 5)][1]                 # 5 秒模型 = 6；连续 30 秒模型 = 1
    xw = rearrange(x, 'B C (f L) -> (B f) C L', f=nw)                      # 折成 6 个 5 秒窗（30 秒模型不折）
    n = xw.shape[0]
    t = torch.full((n, 1), diffusion_t, dtype=torch.long, device=x.device)
    cls = torch.full((n, 1), modality, dtype=torch.long, device=x.device)
    cond = model.calc_cond(t, cls)
    if model.cross_modal != 'none':                                         # 空条件分支
        cond = cond + model.xc_flag(torch.zeros(n, dtype=torch.long, device=x.device))[:, None, :]
    h = model.in_layer(rearrange(xw, 'B H L -> B L H'))
    acts, feedback = [], None
    for i, layer in enumerate(model.layers, start=1):
        h, _, q = layer(h, cond, None, query=['gate'], skip_skip=True)
        acts.append(q[0])
        if model.self_cond_layer and i == model.self_cond_layer:
            fb = pool_time(h, model.self_cond_pool)
            phi = model.sc_t_embed(model._t_embed_fn(t, model.d_embed))
            cond = cond + (model.sc_proj(fb) * phi)[:, None, :]
            feedback = rearrange(fb, '(B f) D -> B f D', f=nw).mean(dim=1)
    a = torch.stack(acts, dim=1)                                            # (B f) n L H
    return rearrange(a, '(B f) n L H -> B n (f L) H', f=nw), feedback


def pooled(a, pool_sec):
    """a [B, n_layer, 6000, d] -> (std, mean) 各 [B, n_layer*d]"""
    w = 200 * pool_sec
    B, nl, L, d = a.shape
    a = a.reshape(B, nl, L // w, w, d)
    s = a.std(dim=3).mean(dim=2)
    mu = a.mean(dim=3).mean(dim=2)
    return s.reshape(B, -1), mu.reshape(B, -1)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run-name', help='W4 run under SLEEP_W4_BIG/runs/ (use with --step)')
    g.add_argument('--ckpt', help='explicit checkpoint path (e.g. a teammate run or a W3 checkpoint)')
    p.add_argument('--step', type=int, help='checkpoint step for --run-name')
    p.add_argument('--weights', choices=['ema_model', 'model'], default='ema_model')
    p.add_argument('--diffusion-t', type=int, default=DIFFUSION_T,
                   help='timestep fed to the model without noise (team readout = 1; other values = diagnostics only)')
    p.add_argument('--pool-sec', type=int, default=1, choices=[1, 2, 3, 5],
                   help='pooling window in seconds (W3 = 1); must be agreed by all three groups')
    p.add_argument('--modalities', default='eeg,eog', help='comma list; EOG-only downstream needs eog')
    p.add_argument('--out-name', default=None, help='output folder name under features/ (default: run name)')
    p.add_argument('--batch-epochs', type=int, default=32)
    p.add_argument('--limit-records', type=int, default=0, help='TEST ONLY: first N records')
    p.add_argument('--verify-eog-only', type=int, default=2,
                   help='re-extract EOG with EEG channel zeroed on the first N records and compare bitwise')
    args = p.parse_args()

    mods = [m.strip() for m in args.modalities.split(',')]
    if not mods or any(m not in C.MODALITIES for m in mods):
        C.fail(f'--modalities must be a subset of {list(C.MODALITIES)}')
    if args.run_name:
        if args.step is None:
            C.fail('--run-name needs --step')
        ckpt_path = C.RUNS_BIG / args.run_name / f'ckpt_step{args.step}.pt'
        name = args.out_name or args.run_name
    else:
        ckpt_path = Path(args.ckpt)
        name = args.out_name or ckpt_path.parent.name
    if not ckpt_path.exists():
        C.fail(f'Checkpoint not found: {ckpt_path}')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, ck, opts = load_model(ckpt_path, args.weights)
    model = model.cuda()
    n_layer, d_model = len(model.layers), model.d_model
    step = ck.get('step')
    print(f'checkpoint {ckpt_path.name} step {step} weights {args.weights} switches {opts} '
          f'window_sec {model.window_sec}', flush=True)

    tag = f'step{step}_{args.weights}_pool{args.pool_sec}s' + (f'_t{args.diffusion_t}' if args.diffusion_t != DIFFUSION_T else '')
    if args.limit_records:
        tag += f'_TEST{args.limit_records}rec'
    out = C.BIG / 'features' / name / tag
    if (out / 'DONE').exists():
        C.fail(f'{out} already complete')
    out.mkdir(parents=True, exist_ok=True)

    recs = C.split_records()
    labeled = set(C.label25_subjects())
    todo = sorted(r for r, (s, sp) in recs.items() if (sp == 'train' and s in labeled) or sp == 'val')
    if args.limit_records:
        todo = todo[:args.limit_records]

    def extract(sig_mod, mi):
        x_all = C.transform_uv(sig_mod)
        S, M, F = [], [], []
        for b in range(0, len(x_all), args.batch_epochs):
            x = torch.from_numpy(x_all[b:b + args.batch_epochs, None, :]).cuda()
            a, fb = gate_activations(model, x, mi, args.diffusion_t)
            s, mu = pooled(a, args.pool_sec)
            if not torch.isfinite(s).all():
                C.fail('Non-finite features')
            S.append(s.cpu().numpy())
            M.append(mu.cpu().numpy())
            if fb is not None:
                F.append(fb.cpu().numpy())
        return np.concatenate(S), np.concatenate(M), (np.concatenate(F) if F else None)

    index, verify = [], []
    kinds = ('std', 'mean') + (('feedback',) if model.self_cond_layer else ())
    feats = {f'{m}_{r}': [] for m in mods for r in kinds}
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for i, rec in enumerate(todo):
        subj, sp = recs[rec]
        with np.load(C.DATA / f'{rec}.npz', allow_pickle=False) as z:
            sig, lab, ep = z['signal'], z['labels'], z['epoch_index']
        for m in mods:
            mi = C.MODALITIES[m]
            s, mu, fb = extract(sig[:, mi, :], mi)             # 只把这一模态的通道送进模型
            feats[f'{m}_std'].append(s)
            feats[f'{m}_mean'].append(mu)
            if fb is not None:
                feats[f'{m}_feedback'].append(fb)
            if m == 'eog' and i < args.verify_eog_only:
                sig0 = sig.copy()
                sig0[:, C.MODALITIES['eeg'], :] = 0.0           # 没有 EEG 的输入环境
                s0, mu0, _ = extract(sig0[:, mi, :], mi)
                ok = bool(np.array_equal(s, s0) and np.array_equal(mu, mu0))
                verify.append(dict(record_id=rec, eog_identical_without_eeg=ok))
                if not ok:
                    C.fail(f'EOG features changed when EEG was removed ({rec})')
        role = 'labeled_train' if sp == 'train' else 'val'
        for e, y in zip(ep, lab):
            index.append(dict(sample_id=f'{rec}_{int(e):04d}', subject_id=subj, record_id=rec,
                              epoch_index=int(e), split=sp, role=role, y_true=int(y)))
        print(f'[{i + 1:2d}/{len(todo)}] {rec} {role} n={len(ep)} elapsed={time.perf_counter() - start:.0f}s',
              flush=True)

    elapsed = time.perf_counter() - start
    info = dict(checkpoint=ckpt_path.name, checkpoint_sha256=C.sha256_file(ckpt_path), step=step,
                weights=args.weights, switches=opts, run_config_name=(ck.get('run_config') or {}).get('run_name'),
                modalities=mods, n_samples=len(index),
                n_labeled_train=sum(r['role'] == 'labeled_train' for r in index),
                n_val=sum(r['role'] == 'val' for r in index), limit_records=args.limit_records,
                readout=f'diffusion_t={args.diffusion_t}, no noise, query=gate, all {n_layer} layers, '
                        f'per-{args.pool_sec}s std/mean, mean over {30 // args.pool_sec} positions; '
                        'cross-modal models: null EEG condition (flag 0)',
                pool_sec=args.pool_sec, dim_per_modality=n_layer * d_model,
                eog_only_verification=verify, seconds=elapsed,
                **({'window_sec': model.window_sec} if model.window_sec != 5 else {}), epochs_per_second=len(index) / elapsed,
                peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20, files={})
    for k, v in feats.items():
        arr = np.concatenate(v).astype(np.float32)
        if k.endswith('_feedback'):
            if arr.shape[0] != len(index):
                C.fail(f'Bad shape for {k}: {arr.shape}')
        elif arr.shape != (len(index), n_layer * d_model):
            C.fail(f'Bad shape for {k}: {arr.shape}')
        np.save(out / f'{k}.npy', arr)
        info['files'][k] = dict(shape=list(arr.shape), sha256=C.sha256_file(out / f'{k}.npy'))
    C.write_csv(out / 'index.csv', index)
    info['index_sha256'] = C.sha256_file(out / 'index.csv')
    C.write_json(out / 'features_info.json', info)
    C.write_json(C.RUNS_SMALL / name / f'features_{tag}.json', info)
    (out / 'DONE').write_text('ok')
    print(f'samples={len(index)} seconds={elapsed:.0f} peak_gpu={info["peak_gpu_mib"]:.0f} MiB '
          f'eog_only_check={verify}', flush=True)
    print('FEATURES DONE', flush=True)


if __name__ == '__main__':
    main()
