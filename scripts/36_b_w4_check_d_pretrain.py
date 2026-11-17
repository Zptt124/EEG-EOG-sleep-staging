"""Step W4-B6: 独立复算 D 公共预训练脚本（PR #23）的 held-out val MSE，并核对错时配对表。

两个子命令：

  mse    按 D 在 manifests/d_val_w4_v01.json 里写的配方，用 B 自己的代码从头重建 2,048 个 val 窗、t 和噪声：
         - 训练池行序从 A 的 NPZ 重建（split=train 记录按编号排序，段按 NPZ 顺序），不读 D 的缓存；
         - held-out 名单读 B 的 manifests/pretrain_split_w4_v01.csv；
         - 单元编号、t、噪声的 SHA256 与 D 登记的值比对，窗表与 D 的 d_val_windows_w4_v01.csv 逐行比对；
         - x0 从 NPZ 重新做 200 Hz + µV/100 + staged mu-law，再按 D 的缓存取 float16 精度；
         - 前向用官方 Wavenet.forward（不经过 D 的 w4_forward），不给 EEG 条件；
         - 每个 checkpoint × 权重（ema/online）的 22 个分段值和 stop_metric 与 D 的 val_mse.csv 逐格比较。

  pairs  调用 D 的 build_shift_pairs，在 B 从 NPZ 重建的窗起点上重算配对表，SHA256 与 D 登记的 npy 比对；
         再用 B 自己的检查逐条核对 C 的规则：同一记录、|Δ| ∈ [60, 300] 秒、两个方向、在候选里均匀抽取。

用法：
  python scripts/36_b_w4_check_d_pretrain.py mse --vendor <vendor/EEGDM> --d-week4 <experiment4/week4> \
      --ckpt-dir <解压后的 checkpoint 目录> --out reports/week4/d_val_recheck
  python scripts/36_b_w4_check_d_pretrain.py pairs --vendor <vendor/EEGDM> --d-week4 <experiment4/week4> \
      --out reports/week4/d_val_recheck
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / 'data' / 'processed'
SPLIT_CSV = BASE / 'manifests' / 'split_subjects.csv'
HELDOUT_CSV = BASE / 'manifests' / 'pretrain_split_w4_v01.csv'

_spec = importlib.util.spec_from_file_location('mse32', Path(__file__).with_name('32_b_w4_pretrain_mse.py'))
M32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M32)

N_WIN, WIN, N_BUCKET, BUCKET = 6, 1000, 10, 5
STOP_BUCKETS = [1, 2, 3, 4]
VAL_SEED = 20261114
PER_SUBJECT = 128
MOD_NAMES = ('eeg', 'eog')


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rebuild_index():
    """从 NPZ 重建训练池行序：split=train 的记录按编号排序，每条记录内按 NPZ 的段顺序。"""
    split = list(csv.DictReader(open(SPLIT_CSV, encoding='utf-8')))
    recs = sorted((rec, r['subject_id']) for r in split if r['split'] == 'train' for rec in r['record_ids'].split(';'))
    rows = []
    for rec, subj in recs:
        with np.load(PROCESSED / f'{rec}.npz') as z:
            ep = z['epoch_index']
        rows += [(rec, subj, int(e), i) for i, e in enumerate(ep)]
    idx = pd.DataFrame(rows, columns=['record_id', 'subject_id', 'epoch_index', 'npz_row'])
    idx['sample_id'] = idx['record_id'] + '_' + idx['epoch_index'].map('{:04d}'.format)
    return idx


def compare_index(idx, d_week4):
    d = pd.read_csv(d_week4 / 'manifests' / 'pretrain_epoch_index.csv', dtype={'epoch_index': int})
    cols = ['sample_id', 'subject_id', 'record_id', 'epoch_index']
    same = len(d) == len(idx) and (d['row'].to_numpy() == np.arange(len(d))).all() and \
        all((d[c].to_numpy() == idx[c].to_numpy()).all() for c in cols)
    return dict(n_rows=len(idx), n_records=int(idx['record_id'].nunique()), matches_d_epoch_index=bool(same))


def heldout_subjects():
    h = sorted(r['subject_id'] for r in csv.DictReader(open(HELDOUT_CSV, encoding='utf-8'))
               if r['pretrain_role'] == 'heldout_val')
    if len(h) != 8:
        sys.exit('held-out 名单不是 8 人')
    return h


def rebuild_val(idx, heldout):
    """按 D 的 d_val_w4_v01.json 配方重建单元、t、噪声。单元 u = (row*2 + m)*6 + k。"""
    rng = np.random.default_rng(VAL_SEED)
    subj = idx['subject_id'].to_numpy()
    units = []
    for s in heldout:
        rows = np.nonzero(subj == s)[0]
        for m in (0, 1):
            cand = (rows[:, None] * 2 * N_WIN + m * N_WIN + np.arange(N_WIN)[None, :]).reshape(-1)
            units.append(np.sort(rng.choice(cand, size=PER_SUBJECT, replace=False)))
    units = np.concatenate(units).astype(np.int64)
    t = (np.arange(N_BUCKET)[None, :] * BUCKET + rng.integers(0, BUCKET, size=(len(units), N_BUCKET))).astype(np.int64)
    g = torch.Generator().manual_seed(VAL_SEED)
    noise = torch.randn((len(units), N_BUCKET, 1, WIN), generator=g)
    row, m, k = units // (2 * N_WIN), (units // N_WIN) % 2, units % N_WIN
    tab = idx.iloc[row].reset_index(drop=True)[['subject_id', 'record_id', 'sample_id', 'epoch_index', 'npz_row']]
    tab.insert(0, 'unit_id', units)
    tab['modality_id'], tab['modality'], tab['window_k'] = m, np.array(MOD_NAMES)[m], k
    return tab, t, noise


def check_val(tab, t, noise, d_week4):
    meta = json.loads((d_week4 / 'manifests' / 'd_val_w4_v01.json').read_text(encoding='utf-8'))
    fp = meta['fingerprint']
    wpath = d_week4 / 'manifests' / 'd_val_windows_w4_v01.csv'
    d = pd.read_csv(wpath)
    cols = ['unit_id', 'subject_id', 'record_id', 'sample_id', 'epoch_index', 'modality', 'window_k']
    res = dict(
        units_sha256_match=sha_arr(tab['unit_id'].to_numpy().astype('<i8')) == fp['units_sha256'],
        t_sha256_match=sha_arr(t.astype('<i8')) == fp['t_sha256'],
        noise_sha256_match=sha_arr(noise.numpy().astype('<f4')) == fp['noise_sha256'],
        windows_csv_sha256_match=M32.sha256_file(wpath) == meta['sha256_windows_csv'],
        windows_csv_rows_match=bool(len(d) == len(tab) and all((d[c].to_numpy() == tab[c].to_numpy()).all() for c in cols)
                                    and (d[[f't_b{b}' for b in range(N_BUCKET)]].to_numpy() == t).all()),
        heldout_csv_sha256_match=M32.sha256_file(HELDOUT_CSV) == meta['heldout']['sha256'],
    )
    return res


def build_x0(tab):
    """NPZ → 200 Hz → µV/100 → staged mu-law → float16（与 D 的缓存同精度）→ 第 k 个 5 秒窗。"""
    from scipy.signal import resample_poly
    from src.util import staged_mu_law
    x0 = np.empty((len(tab), 1, WIN), dtype=np.float32)
    for rec, g in tab.groupby('record_id', sort=True):
        with np.load(PROCESSED / f'{rec}.npz') as z:
            ei, sig = z['epoch_index'], z['signal']
        r = g['npz_row'].to_numpy()
        if not (ei[r] == g['epoch_index'].to_numpy()).all():
            sys.exit(f'{rec}: NPZ 段号不对应')
        x = sig[r, g['modality_id'].to_numpy()].astype(np.float32)
        x = resample_poly(x, 2, 1, axis=-1).astype(np.float32)
        x = staged_mu_law(x / 100).astype(np.float32).astype(np.float16).astype(np.float32)
        k = g['window_k'].to_numpy()
        x0[g.index.to_numpy(), 0] = np.take_along_axis(x, k[:, None] * WIN + np.arange(WIN)[None, :], axis=1)
    return x0


@torch.no_grad()
def eval_per(net, sch, x0, m, t, noise, batch):
    n = len(x0)
    out = np.empty((n, N_BUCKET), dtype=np.float64)
    x0, m, t = torch.from_numpy(x0), torch.from_numpy(m).long()[:, None], torch.from_numpy(t)
    for b in range(N_BUCKET):
        for s in range(0, n, batch):
            sl = slice(s, s + batch)
            tb = t[sl, b][:, None]
            e = noise[sl, b]
            xt = sch.add_noise(x0[sl], e, tb)
            v = sch.get_velocity(x0[sl], e, tb)
            out[sl, b] = ((net(xt, tb, m[sl]) - v) ** 2).mean(dim=(1, 2)).double().numpy()
    return out


def summarize(per, m):
    out, stop = {}, []
    for mi, name in enumerate(MOD_NAMES):
        pm = per[m == mi]
        for b in range(N_BUCKET):
            out[f'{name}_t{b * BUCKET:02d}_{(b + 1) * BUCKET:02d}'] = float(pm[:, b].mean())
        out[f'{name}_all'] = float(pm.mean())
        out[f'{name}_t05_25'] = float(pm[:, STOP_BUCKETS].mean())
        stop.append(out[f'{name}_t05_25'])
    out['stop_metric'] = float(np.mean(stop))
    return out


def cmd_mse(args):
    d_week4 = Path(args.d_week4)
    M32.add_vendor(args.vendor)
    from model.diffusion_model import Wavenet
    from diffusers import DDPMScheduler
    torch.set_num_threads(args.threads) if args.threads else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    idx = rebuild_index()
    idx_check = compare_index(idx, d_week4)
    heldout = heldout_subjects()
    tab, t, noise = rebuild_val(idx, heldout)
    val_check = check_val(tab, t, noise, d_week4)
    print(json.dumps(dict(index=idx_check, val=val_check), indent=1), flush=True)
    hard = {k: v for k, v in val_check.items() if k != 'noise_sha256_match'}
    if not idx_check['matches_d_epoch_index'] or not all(hard.values()):
        sys.exit('重建的行序或 val 集与 D 登记的不一致，停止')
    if not val_check['noise_sha256_match']:
        # torch 的 CPU float32 randn 在 x86（AVX2 向量化 Box-Muller）和 ARM（通用实现）上末位可能不同；
        # 种子、形状、顺序一致时差异在 1e-7 量级。继续算，最终以 MSE 与 D 的差判断。
        print(f'警告：噪声 SHA256 与 D 不同（本机 {platform.machine()}，torch {torch.__version__}），'
              '多半是 CPU 随机数实现的平台差异；继续计算', flush=True)
    x0 = build_x0(tab)
    m = tab['modality_id'].to_numpy()
    sch = DDPMScheduler(**M32.NOISE_SCH_KWARGS)

    d_val = pd.read_csv(d_week4 / 'runs' / args.run / 'val_mse.csv')
    d_ck = pd.read_csv(d_week4 / 'runs' / args.run / 'checkpoints.csv')
    metric_cols = [c for c in d_val.columns if c[:4] in ('eeg_', 'eog_') and c != 'eeg_cond'] + ['stop_metric']
    rows, ck_info = [], []
    for _, ck in d_ck.iterrows():
        path = Path(args.ckpt_dir) / ck['file']
        sha = M32.sha256_file(path)
        if sha != ck['sha256']:
            sys.exit(f'{path.name}: SHA256 与 D 的 checkpoints.csv 不一致')
        state = torch.load(path, map_location='cpu', weights_only=False)
        ck_info.append(dict(step=int(ck['step']), file=path.name, sha256=sha))
        for w in args.weights:
            net = Wavenet(**state['model_kwargs'])
            net.load_state_dict(state['ema_model' if w == 'ema' else 'model'], strict=True)
            net.eval()
            ts = time.time()
            per = eval_per(net, sch, x0, m, t, noise, args.batch)
            mine = summarize(per, m)
            ref = d_val[(d_val['step'] == ck['step']) & (d_val['weights'] == w) & (d_val['eeg_cond'] == 0)]
            if len(ref) != 1:
                sys.exit(f'val_mse.csv 里找不到 step {ck["step"]} {w}')
            ref = ref.iloc[0]
            diffs = {c: mine[c] - float(ref[c]) for c in metric_cols}
            rows.append(dict(step=int(ck['step']), weights=w, stop_metric_b=mine['stop_metric'],
                             stop_metric_d=float(ref['stop_metric']),
                             max_abs_diff=max(abs(v) for v in diffs.values()),
                             max_rel_diff=max(abs(v) / float(ref[c]) for c, v in diffs.items()),
                             **{f'b_{c}': mine[c] for c in metric_cols}))
            print(f'step {ck["step"]:>6} {w:6s} stop B {mine["stop_metric"]:.6f} D {ref["stop_metric"]:.6f} '
                  f'max|diff| {rows[-1]["max_abs_diff"]:.2e} ({time.time() - ts:.0f}s)', flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'val_mse_recheck.csv', index=False, float_format='%.6g', lineterminator='\n')
    run = dict(run=args.run, rule='D d_val_w4_v01 (rebuilt independently by B)', index_check=idx_check,
               val_check=val_check, checkpoints=ck_info, weights=args.weights,
               forward='official Wavenet.forward, no EEG condition', x0='NPZ -> transform -> float16 round',
               max_abs_diff=float(res['max_abs_diff'].max()), max_rel_diff=float(res['max_rel_diff'].max()),
               device='cpu', torch=torch.__version__, platform=platform.platform(),
               seconds=round(time.time() - t0, 1))
    (out / 'val_mse_recheck.json').write_text(json.dumps(run, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'max|diff| {run["max_abs_diff"]:.2e}, max rel {run["max_rel_diff"]:.2e}')


def cmd_pairs(args):
    d_week4 = Path(args.d_week4)
    M32.add_vendor(args.vendor)
    sys.path.insert(0, str(d_week4 / 'src'))
    import w4d_common as DC
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summ_d = json.loads((d_week4 / 'manifests' / 'shift_pairs_60_300_v01_summary.json').read_text(encoding='utf-8'))
    idx = rebuild_index()
    idx_check = compare_index(idx, d_week4)
    if not idx_check['matches_d_epoch_index']:
        sys.exit('重建的行序与 D 的 pretrain_epoch_index.csv 不一致')

    class Pool:  # build_shift_pairs 只用到 window_start_sec() 和 record
        record = idx['record_id'].to_numpy()

        @staticmethod
        def window_start_sec():
            return (idx['epoch_index'].to_numpy()[:, None] * 30 + np.arange(N_WIN)[None, :] * 5).reshape(-1)

    pair, gap = DC.build_shift_pairs(Pool)
    tmp = out / '_shift_pairs_rebuilt.npy'
    np.save(tmp, pair)
    npy_sha = M32.sha256_file(tmp)
    tmp.unlink()

    # B 自己的规则检查（不用 D 的代码）
    rec_w = np.repeat(idx['record_id'].to_numpy(), N_WIN)
    start = Pool.window_start_sec()
    ok = pair >= 0
    tgt = np.nonzero(ok)[0]
    c = pair[tgt]
    g = start[c] - start[tgt]
    same_rec = bool((rec_w[c] == rec_w[tgt]).all())
    in_range = bool(((np.abs(g) >= 60) & (np.abs(g) <= 300)).all())
    gap_consistent = bool((g == gap[tgt]).all())
    # 均匀性：每个目标窗的候选按时间排序，记录被选中的名次 r ∈ {0..n-1}。
    # 随机化 PIT：(r + U)/n，U ~ U(0,1) 独立；抽取均匀时它严格服从 U(0,1)（候选数少的窗也成立）。
    jitter = np.random.default_rng(0).random(len(tgt))
    ranks = np.empty(len(tgt))
    ncand = np.empty(len(tgt), dtype=np.int64)
    bounds = np.flatnonzero(np.r_[True, rec_w[1:] != rec_w[:-1], True])
    pos = 0
    no_cand_ok = True
    for a, b in zip(bounds[:-1], bounds[1:]):
        st = start[a:b]
        for i in range(a, b):
            d = st - start[i]
            cand = np.nonzero((np.abs(d) >= 60) & (np.abs(d) <= 300))[0] + a
            if not ok[i]:
                no_cand_ok &= len(cand) == 0
                continue
            j = np.searchsorted(cand, pair[i])
            if j >= len(cand) or cand[j] != pair[i]:
                sys.exit(f'窗 {i} 的条件窗不在候选里')
            ranks[pos], ncand[pos] = (j + jitter[pos]) / len(cand), len(cand)
            pos += 1
    from scipy.stats import binomtest, chisquare, kstest
    hist = np.histogram(ranks, bins=20, range=(0, 1))[0]
    chi = chisquare(hist)
    res = dict(
        index_check=idx_check,
        n_windows=int(len(pair)), n_paired=int(ok.sum()), coverage=float(ok.mean()),
        rebuilt_npy_sha256=npy_sha, d_npy_sha256=summ_d['sha256_npy'], npy_sha256_match=npy_sha == summ_d['sha256_npy'],
        summary_counts_match=bool(int(ok.sum()) == summ_d['n_paired'] and int((g < 0).sum()) == summ_d['n_negative_gap']
                                  and int((g > 0).sum()) == summ_d['n_positive_gap']),
        same_record=same_rec, gap_in_60_300=in_range, gap_matches_returned=gap_consistent,
        unpaired_have_no_candidate=bool(no_cand_ok),
        n_negative_gap=int((g < 0).sum()), n_positive_gap=int((g > 0).sum()),
        n_candidates_min=int(ncand.min()), n_candidates_median=float(np.median(ncand)),
        rank_uniformity_chi2_20bins=dict(stat=float(chi.statistic), p=float(chi.pvalue)),
        rank_uniformity_ks_p=float(kstest(ranks, 'uniform').pvalue),
        direction_binomial_p=float(binomtest(int((g > 0).sum()), len(g), 0.5).pvalue),
        note='名次用随机化 PIT (r+U)/n（U 的 seed 0），均匀抽取时严格服从 U(0,1)',
    )
    (out / 'shift_pairs_recheck.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('mse', 'pairs'):
        s = sub.add_parser(name)
        s.add_argument('--vendor', required=True)
        s.add_argument('--d-week4', default=str(BASE / 'experiment4' / 'week4'))
        s.add_argument('--out', default=str(BASE / 'reports' / 'week4' / 'd_val_recheck'))
    s = sub.choices['mse']
    s.add_argument('--ckpt-dir', required=True)
    s.add_argument('--run', default='W4_D_main_m64x12_seed0_v01')
    s.add_argument('--weights', nargs='+', default=['ema', 'online'], choices=['ema', 'online'])
    s.add_argument('--batch', type=int, default=512)
    s.add_argument('--threads', type=int, default=0)
    args = p.parse_args()
    {'mse': cmd_mse, 'pairs': cmd_pairs}[args.cmd](args)


if __name__ == '__main__':
    main()
