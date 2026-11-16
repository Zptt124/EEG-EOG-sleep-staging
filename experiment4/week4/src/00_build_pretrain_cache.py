"""第 0 步：生成预训练用的训练池缓存（CPU，约 1–2 分钟；组员在自己电脑上跑一次即可）。

与第三周 D 的 02_build_pretrain_cache.py 处理完全相同，生成的 signals_f16.npy / epoch_index.csv
应与 D 的缓存逐字节相同（SHA256 写死在 w4d_common.py，不一致时报错）。
缓存包含训练池全部 54 人；held-out 的 8 人在训练时按受试者过滤，不在这里剔除。

只处理 split=train 的 54 人；只读取 NPZ 的 signal 和 epoch_index，不读取 labels。
输出（SLEEP_PRETRAIN_CACHE 指定的目录）：
  signals_f16.npy     (N, 2, 6000) float16，已做 100->200 Hz、uV/100、staged mu-law
  epoch_index.csv     row, sample_id, subject_id, record_id, epoch_index（无标签）
  cache_manifest.json 配置、样本数、float16 误差、SHA256
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import w4d_common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    out = C.CACHE_DIR
    if (out / 'cache_manifest.json').exists() and not args.overwrite:
        C.fail(f'{out} already complete; use --overwrite to rebuild')
    out.mkdir(parents=True, exist_ok=True)

    recs = C.split_records()
    train = sorted(r for r, (_, s) in recs.items() if s == 'train')
    missing = [r for r in train if not (C.DATA / f'{r}.npz').exists()]
    if missing:
        C.fail(f'Missing NPZ files under {C.DATA} (set SLEEP_DATA): {missing[:5]} ...')

    counts = {}
    for r in train:
        with np.load(C.DATA / f'{r}.npz', allow_pickle=False) as z:
            counts[r] = len(z['epoch_index'])
    n_total = sum(counts.values())
    print(f'train records {len(train)}, epochs {n_total}', flush=True)

    tmp = out / 'signals_f16.partial.npy'
    dest = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=(n_total, 2, 6000))
    index, row = [], 0
    max_err, vmin, vmax, n_clip = 0.0, np.inf, -np.inf, 0
    start = time.perf_counter()
    for i, r in enumerate(train):
        with np.load(C.DATA / f'{r}.npz', allow_pickle=False) as z:
            sig = z['signal']            # (n, 2, 3000) uV；不读取 labels
            ep = z['epoch_index']
        n = len(ep)
        if sig.shape != (n, 2, 3000):
            C.fail(f'{r}: unexpected signal shape {sig.shape}')
        for m in (0, 1):
            x = C.transform_uv(sig[:, m, :])
            x16 = x.astype(np.float16)
            if not np.isfinite(x16).all():
                C.fail(f'{r}: float16 overflow')
            max_err = max(max_err, float(np.abs(x16.astype(np.float32) - x).max()))
            vmin, vmax = min(vmin, float(x.min())), max(vmax, float(x.max()))
            n_clip += int((np.abs(x) > 1).sum())
            dest[row:row + n, m] = x16
        subj = recs[r][0]
        for e in ep:
            index.append(dict(row=len(index), sample_id=f'{r}_{int(e):04d}', subject_id=subj,
                              record_id=r, epoch_index=int(e)))
        row += n
        print(f'[{i + 1:3d}/{len(train)}] {r} epochs={n} elapsed={time.perf_counter() - start:.0f}s', flush=True)
    dest.flush()
    del dest
    if row != n_total:
        C.fail('Row count mismatch')
    tmp.replace(out / 'signals_f16.npy')
    C.write_csv(out / 'epoch_index.csv', index)

    print('computing SHA256 ...', flush=True)
    sha_sig = C.sha256_file(out / 'signals_f16.npy')
    sha_idx = C.sha256_file(out / 'epoch_index.csv')
    manifest = dict(
        version='W3_D_pretrain_cache_v01', built_by='week4 00_build_pretrain_cache.py',
        split_sha256=C.SPLIT_SHA, upstream_commit=C.UPSTREAM_COMMIT,
        split_used='train only (54 subjects)', labels_read=False,
        n_records=len(train), n_subjects=len({recs[r][0] for r in train}), n_epochs=n_total,
        n_units=n_total * 2 * C.N_WIN, unit_definition='(row, modality, 5s window k); u=(row*2+m)*6+k',
        modalities={'0': 'EEG Fpz-Cz', '1': 'EOG horizontal'},
        transform='resample_poly(up=2,down=1) 100->200 Hz; uV/100; official staged_mu_law(mu=255)',
        storage_dtype='float16', float16_max_abs_error=max_err, value_min=vmin, value_max=vmax,
        fraction_abs_gt_1=n_clip / (n_total * 2 * 6000),
        sha256_signals=sha_sig, sha256_epoch_index=sha_idx,
        build_seconds=time.perf_counter() - start)
    C.write_json(out / 'cache_manifest.json', manifest)
    if sha_sig != C.CACHE_SIGNALS_SHA or sha_idx != C.CACHE_INDEX_SHA:
        C.fail(f'Cache built but SHA256 differs from the team reference '
               f'(signals {sha_sig[:12]}..., index {sha_idx[:12]}...); check NPZ version / scipy version')
    print(f'n_epochs {n_total}; SHA256 matches team reference', flush=True)
    print('CACHE DONE', flush=True)


if __name__ == '__main__':
    main()
