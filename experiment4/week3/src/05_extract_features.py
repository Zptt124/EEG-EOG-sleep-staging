"""第 5 步：用冻结的 backbone 提取下游探针特征（GPU，每个 checkpoint 约 10–30 分钟）。

只提取探针需要的人：14 人标签预算（label25，训练侧）+ 8 个开发验证者；测试集不碰。
读出方式与 R1 / C 的 EEGDM 特征一致：
  30 秒段 -> 200 Hz、uV/100、staged mu-law -> 官方 LatentActivityExtractor（内部折成 6 个 5 秒窗）
  diffusion_t=1、无随机加噪、query='gate'、条件 = 模态编号（EEG 0 / EOG 1）、全部层
  每 1 秒（200 点）求 std（torch correction=1，与官方 LatentActivityReducer 相同）和 mean
  -> 30 个时间位置取均值 -> 每个模态 n_layer*d_model 维
权重默认使用 EMA（与官方提取一致）；step 0 时 EMA = online = random-init。
输出：~/week3-1/features/<run_name>/step<N>_<weights>/{eeg_std,eog_std,eeg_mean,eog_mean}.npy、index.csv
     以及 D: 上 runs/<run_name>/features_step<N>_<weights>.json（形状、耗时、SHA）
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import w3d_common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--weights', choices=['ema_model', 'model'], default='ema_model')
    p.add_argument('--batch-epochs', type=int, default=32)
    p.add_argument('--save-full', action='store_true', help='also keep per-second std [n_layer,30,d_model] float16')
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    C.add_vendor_to_path()
    from model.classifier import LatentActivityExtractor

    ckpt_path = C.RUNS_BIG / args.run_name / f'ckpt_step{args.step}.pt'
    if not ckpt_path.exists():
        C.fail(f'Checkpoint not found: {ckpt_path}')
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    from model.diffusion_model import Wavenet
    model = Wavenet(**ck['model_kwargs'])
    model.load_state_dict(ck[args.weights], strict=True)
    model.requires_grad_(False).eval().cuda()
    n_layer, d_model = ck['model_kwargs']['n_layer'], ck['model_kwargs']['d_model']
    extractors = {m: LatentActivityExtractor(model, diffusion_t=1, query=['gate'], use_cond=[i]).eval().cuda()
                  for m, i in C.MODALITIES.items()}

    tag = f'step{args.step}_{args.weights}'
    out = C.BIG / 'features' / args.run_name / tag
    if (out / 'DONE').exists():
        C.fail(f'{out} already complete')
    out.mkdir(parents=True, exist_ok=True)

    recs = C.split_records()
    labeled = set(C.label25_subjects())
    todo = sorted(r for r, (s, sp) in recs.items() if (sp == 'train' and s in labeled) or sp == 'val')
    index = []
    feats = {f'{m}_{r}': [] for m in C.MODALITIES for r in ('std', 'mean')}
    full = {m: [] for m in C.MODALITIES}
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for i, rec in enumerate(todo):
        subj, sp = recs[rec]
        with np.load(C.DATA / f'{rec}.npz', allow_pickle=False) as z:
            sig, lab, ep = z['signal'], z['labels'], z['epoch_index']
        for m, mi in C.MODALITIES.items():
            x_all = C.transform_uv(sig[:, mi, :])
            for b in range(0, len(x_all), args.batch_epochs):
                x = torch.from_numpy(x_all[b:b + args.batch_epochs, None, :]).cuda()
                with torch.inference_mode():
                    a = extractors[m]((x, None))          # (B, 1, n_layer, 1, 6000, d_model)
                    a = a[:, 0, :, 0]                      # (B, n_layer, 6000, d_model)
                    a = a.reshape(a.shape[0], n_layer, 30, 200, d_model)
                    s = a.std(dim=3)                       # (B, n_layer, 30, d_model)
                    mu = a.mean(dim=3)
                if not torch.isfinite(s).all():
                    C.fail(f'Non-finite features in {rec}')
                feats[f'{m}_std'].append(s.mean(dim=2).reshape(len(x), -1).cpu().numpy())
                feats[f'{m}_mean'].append(mu.mean(dim=2).reshape(len(x), -1).cpu().numpy())
                if args.save_full:
                    full[m].append(s.cpu().numpy().astype(np.float16))
        role = 'labeled_train' if sp == 'train' else 'val'
        for e, y in zip(ep, lab):
            index.append(dict(sample_id=f'{rec}_{int(e):04d}', subject_id=subj, record_id=rec,
                              epoch_index=int(e), split=sp, role=role, y_true=int(y)))
        print(f'[{i + 1:2d}/{len(todo)}] {rec} {role} n={len(ep)} elapsed={time.perf_counter() - start:.0f}s', flush=True)

    elapsed = time.perf_counter() - start
    info = dict(run_name=args.run_name, step=args.step, weights=args.weights, checkpoint=str(ckpt_path),
                checkpoint_sha256=C.sha256_file(ckpt_path), n_samples=len(index),
                n_labeled_train=sum(r['role'] == 'labeled_train' for r in index),
                n_val=sum(r['role'] == 'val' for r in index),
                readout='diffusion_t=1, no noise, query=gate, all layers, per-second std/mean (200 pts), mean over 30 positions',
                dim_per_modality=n_layer * d_model, seconds=elapsed,
                epochs_per_second=len(index) / elapsed,
                peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20, files={})
    for k, v in feats.items():
        arr = np.concatenate(v).astype(np.float32)
        if arr.shape != (len(index), n_layer * d_model):
            C.fail(f'Bad shape for {k}: {arr.shape}')
        np.save(out / f'{k}.npy', arr)
        info['files'][k] = dict(shape=list(arr.shape), sha256=C.sha256_file(out / f'{k}.npy'))
    if args.save_full:
        for m, v in full.items():
            np.save(out / f'{m}_std_full_f16.npy', np.concatenate(v))
    C.write_csv(out / 'index.csv', index)
    info['index_sha256'] = C.sha256_file(out / 'index.csv')
    C.write_json(out / 'features_info.json', info)
    C.write_json(C.EXP_D / 'runs' / args.run_name / f'features_{tag}.json', info)
    (out / 'DONE').write_text('ok')
    print(f'samples={len(index)} seconds={elapsed:.0f} peak_gpu={info["peak_gpu_mib"]:.0f} MiB', flush=True)
    print('FEATURES DONE', flush=True)


if __name__ == '__main__':
    main()
