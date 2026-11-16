"""导出 val MSE 的固定窗清单（CPU，约 1 分钟），供 B 独立复算、全组核对。

全组的正式 val MSE 用 D 这版（B 于 11-16 确认，见 PR #21 说明）。本脚本把 ValSet 用到的窗、t、噪声配方
原样写成清单，B 的 32 号脚本改成读取它即可复算出同样的 MSE。

输出（exp_d_pretrain/manifests/）：
  d_val_windows_w4_v01.csv   每行一个 val 窗：val_row, unit_id, subject_id, record_id, sample_id, epoch_index,
                             modality(eeg/eog), window_k, t_b0 ... t_b9（10 个 t 段里各自固定的 t）
  d_val_w4_v01.json          名单 SHA、选窗/t/噪声的种子与配方、units/t/noise 的 SHA256、MSE 的计算口径
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w4d_common as C
from w4d_train import BUCKET, N_BUCKET, STOP_BUCKETS, VAL_SEED, ValSet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--heldout-csv', required=True, help='B: manifests/pretrain_split_w4_v01.csv')
    args = p.parse_args()
    heldout, ho_info = C.load_heldout(args.heldout_csv)
    if not ho_info['official_w4_v01']:
        C.fail('Export only from the official team list pretrain_split_w4_v01.csv')
    pool = C.TrainPool(load_to_ram=False)
    val = ValSet(pool, heldout, device='cpu')

    rows = []
    for i, u in enumerate(val.units):
        row = int(u) // (2 * C.N_WIN)
        m = (int(u) // C.N_WIN) % 2
        r = pool.index[row]
        d = dict(val_row=i, unit_id=int(u), subject_id=r['subject_id'], record_id=r['record_id'],
                 sample_id=r['sample_id'], epoch_index=int(r['epoch_index']), modality=C.MOD_NAMES[m],
                 window_k=int(u) % C.N_WIN)
        d.update({f't_b{b}': int(val.t_np[i, b]) for b in range(N_BUCKET)})
        rows.append(d)
    out_csv = C.MANIFESTS / 'd_val_windows_w4_v01.csv'
    C.write_csv(out_csv, rows)

    info = dict(
        version='d_val_w4_v01', owner='D', adopted_by_team='2026-11-16 (B, PR #21)',
        heldout=ho_info,
        selection=f'for each held-out subject (sorted), for modality in (eeg, eog): '
                  f'np.random.default_rng({VAL_SEED}) choice of {val.fingerprint["per_subject_per_modality"]} '
                  f'5-s windows without replacement from all windows of that subject, then sorted '
                  f'(one rng shared across subjects/modalities in this order)',
        t_rule=f't_b = {BUCKET}*b + rng.integers(0, {BUCKET}) for b = 0..{N_BUCKET - 1} (same rng, after selection)',
        noise=val.fingerprint['noise_recipe'] + '; noise[i, b] belongs to val_row i, bucket b',
        preprocessing='same as W3 cache: 100->200 Hz resample_poly(2,1), uV/100, official staged_mu_law; '
                      'window k = samples [1000k, 1000(k+1))',
        mse='per (val_row, bucket): mean over 1000 samples of (v_pred - v)^2, v-prediction, DDPM 50 steps '
            'squaredcos_cap_v2; x_t = add_noise(x0, noise, t); main condition = no EEG condition, EMA weights',
        report='{eeg,eog}_tLO_HI = mean over that modality\'s rows for bucket; {eeg,eog}_all = mean over all buckets',
        stop_metric=f'mean over (eeg, eog) of mean over buckets {list(STOP_BUCKETS)} (t in [5, 25))',
        n_rows=len(rows), fingerprint=val.fingerprint,
        sha256_windows_csv=C.sha256_file(out_csv), cache_sha256=pool.cfg['sha256_signals'])
    C.write_json(C.MANIFESTS / 'd_val_w4_v01.json', info)
    for k in ('n_rows', 'sha256_windows_csv'):
        print(f'{k}: {info[k]}', flush=True)
    for k, v in val.fingerprint.items():
        print(f'{k}: {v}', flush=True)
    print('VAL MANIFEST DONE', flush=True)


if __name__ == '__main__':
    main()
