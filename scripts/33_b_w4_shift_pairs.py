"""Step W4-B3: 错时 EEG 条件配对与覆盖率检查（B 组，跨模态三组之“错时 EEG”）。

规则（experiment3/week4/cross_modal_protocol.md）：
  - 目标窗 = 一个 5 秒 EOG 窗 (record, epoch_index, k)；k 为 30 秒段内第 k 个 5 秒窗，起点 start = 30·epoch + 5·k 秒。
  - 候选条件窗 = 同一记录、完整有效的 Fpz-Cz 5 秒窗，且 60 ≤ |start_c − start_t| ≤ 300 秒（两端都含）。
    “完整有效” = 所在 30 秒段在 A 的预处理 NPZ 中存在（NPZ 已裁到入睡前后 30 分钟、去掉 M/? 段，
    这是数据集层面的既有筛选）；配对本身只看时间戳和窗是否存在，不读睡眠标签。
  - 从候选中均匀抽 1 个，记录目标窗 ID、条件窗 ID 与有符号间隔（条件 − 目标，正值表示条件窗在后）。
    没有候选的目标窗条件置空（A/B 两组都置空），登记数量。

抽样是逐窗确定的：u = splitmix64(窗键 ⊕ seed) / 2^64，取第 floor(u·n_候选) 个候选。
因此任何窗的配对只取决于它自己和所在记录，与窗清单的范围、顺序无关：D 锁定的训练窗清单只要是这些窗的子集，
结果就和全表一致，A、B、D 也可以在各自的数据加载器里调用 pair_windows() 现场计算。

用法：
  python scripts/33_b_w4_shift_pairs.py                       # 46 人预训练池全部窗 + 8 名 held-out 的固定验证窗
  python scripts/33_b_w4_shift_pairs.py --windows <D 的窗清单.csv>   # 需含 record_id, epoch_index, window_k 三列

输出：
  cache/w4_shift_pairs/shift_pairs_pretrain_pool_w4_v01.csv   全部训练池窗的配对（约 68 万行，不入库，可由脚本重建）
  reports/week4/b_shift_pairs/shift_pairs_val_windows_w4_v01.csv  held-out 固定验证窗（EEG/EOG 共用位置）的配对
  reports/week4/b_shift_pairs/coverage_summary.json             覆盖率、候选数、方向与间隔分布、表 SHA256
  reports/week4/b_shift_pairs/coverage_by_subject.csv
  reports/week4/b_shift_pairs/offset_hist.csv                   每个有符号间隔（5 秒网格，±60…±300）的计数
"""
import argparse
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / 'data' / 'processed'
OUT_REPORT = BASE / 'reports' / 'week4' / 'b_shift_pairs'
OUT_CACHE = BASE / 'cache' / 'w4_shift_pairs'

_spec = importlib.util.spec_from_file_location('heldout', Path(__file__).with_name('31_b_w4_pretrain_heldout.py'))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

VERSION = 'w4_v01'
PAIR_SEED = 20261118
MIN_GAP, MAX_GAP = 60, 300      # 秒，|Δstart| 两端都含
EPOCH_SEC, WIN_SEC, N_WIN = 30, 5, 6
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x):
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over='ignore'):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & _MASK
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK
        return x ^ (x >> np.uint64(31))


def window_key(record_id, epoch_index, k):
    """SC4001, 961, 3 -> 唯一 64 位整数（记录号 · 2^24 + 段号 · 8 + k）。"""
    rec = np.uint64(int(record_id[2:]))
    return (rec << np.uint64(24)) + (np.asarray(epoch_index, dtype=np.uint64) << np.uint64(3)) + np.asarray(k, dtype=np.uint64)


def uniform01(record_id, epoch_index, k, seed=PAIR_SEED):
    h = splitmix64(window_key(record_id, epoch_index, k) ^ splitmix64(np.uint64(seed)))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def record_windows(record_id):
    """该记录全部有效 5 秒窗：(epoch_index, k, npz_row, start_sec)，按 start 升序。"""
    with np.load(PROCESSED / f'{record_id}.npz') as z:
        ei = z['epoch_index'].astype(np.int64)
    if (np.diff(ei) <= 0).any():
        sys.exit(f'{record_id}: epoch_index 不是严格递增')
    e = np.repeat(ei, N_WIN)
    k = np.tile(np.arange(N_WIN), len(ei))
    row = np.repeat(np.arange(len(ei)), N_WIN)
    return e, k, row, EPOCH_SEC * e + WIN_SEC * k


def pair_windows(record_id, epoch_index, k, seed=PAIR_SEED, _cache={}):
    """给定记录内若干目标窗，返回 dict(数组)：条件窗 epoch/k/npz_row、有符号间隔、候选数（无候选时 -1 / 0）。"""
    if record_id not in _cache:
        _cache.clear()
        _cache[record_id] = record_windows(record_id)
    ce, ck, crow, cs = _cache[record_id]
    epoch_index, k = np.asarray(epoch_index, dtype=np.int64), np.asarray(k, dtype=np.int64)
    if not np.isin(epoch_index, ce[::N_WIN]).all() or ((k < 0) | (k >= N_WIN)).any():
        sys.exit(f'{record_id}: 目标窗所在段不在 NPZ 中')
    t = EPOCH_SEC * epoch_index + WIN_SEC * k
    lo1, hi1 = np.searchsorted(cs, t - MAX_GAP, 'left'), np.searchsorted(cs, t - MIN_GAP, 'right')
    lo2, hi2 = np.searchsorted(cs, t + MIN_GAP, 'left'), np.searchsorted(cs, t + MAX_GAP, 'right')
    n_before, n_after = hi1 - lo1, hi2 - lo2
    n = n_before + n_after
    u = uniform01(record_id, epoch_index, k, seed)
    j = np.minimum((u * n).astype(np.int64), np.maximum(n - 1, 0))
    idx = np.where(j < n_before, lo1 + j, lo2 + (j - n_before))
    has = n > 0
    idx = np.where(has, idx, 0)
    return dict(cond_epoch_index=np.where(has, ce[idx], -1), cond_window_k=np.where(has, ck[idx], -1),
                cond_npz_row=np.where(has, crow[idx], -1), signed_offset_sec=np.where(has, cs[idx] - t, 0),
                n_candidates=n, n_before=n_before, n_after=n_after)


def pair_table(targets):
    """targets: DataFrame[record_id, epoch_index, window_k, (subject_id)] -> 加上配对列。"""
    parts = []
    for rec, g in targets.groupby('record_id', sort=True):
        p = pair_windows(rec, g['epoch_index'].to_numpy(), g['window_k'].to_numpy())
        g = g.copy()
        for c, v in p.items():
            g[c] = v
        parts.append(g)
    out = pd.concat(parts).sort_index()
    has = out['n_candidates'] > 0
    out['target_window_id'] = [f'{r}_{e:04d}_k{k}' for r, e, k in zip(out.record_id, out.epoch_index, out.window_k)]
    out['cond_window_id'] = np.where(has, [f'{r}_{e:04d}_k{k}' for r, e, k in
                                          zip(out.record_id, out.cond_epoch_index.clip(lower=0), out.cond_window_k.clip(lower=0))], '')
    out['cond_status'] = np.where(has, 'shifted', 'null_no_candidate')
    return out


def pool_targets(role):
    split = pd.read_csv(H.OUT_SPLIT)
    rows = []
    for r in split[split.pretrain_role == role].itertuples():
        for rec in r.record_ids.split(';'):
            e, k, _, _ = record_windows(rec)
            rows.append(pd.DataFrame(dict(subject_id=r.subject_id, record_id=rec, epoch_index=e, window_k=k)))
    return pd.concat(rows, ignore_index=True)


def coverage(df, name):
    has = df['n_candidates'] > 0
    off = df.loc[has, 'signed_offset_sec']
    nc = df['n_candidates']
    return dict(
        table=name, n_targets=int(len(df)), n_paired=int(has.sum()), n_null_no_candidate=int((~has).sum()),
        coverage=float(has.mean()),
        n_candidates=dict(min=int(nc.min()), p1=float(nc.quantile(0.01)), median=float(nc.median()), max=int(nc.max()),
                          full_both_sides=int(2 * ((MAX_GAP - MIN_GAP) // WIN_SEC + 1))),
        one_sided=dict(only_before=int(((df.n_before > 0) & (df.n_after == 0)).sum()),
                       only_after=int(((df.n_before == 0) & (df.n_after > 0)).sum()),
                       reduced_candidates=int(((nc > 0) & (nc < 2 * ((MAX_GAP - MIN_GAP) // WIN_SEC + 1))).sum())),
        offset=dict(frac_condition_after=float((off > 0).mean()), mean_abs_sec=float(off.abs().mean()),
                    min=int(off.min()), max=int(off.max()), median_abs_sec=float(off.abs().median()),
                    at_boundary_abs60=int((off.abs() == MIN_GAP).sum()), at_boundary_abs300=int((off.abs() == MAX_GAP).sum())))


def write_csv(df, path):
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator='\n')
    txt = buf.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding='utf-8')
    return hashlib.sha256(txt.encode()).hexdigest()


COLS = ['subject_id', 'record_id', 'epoch_index', 'window_k', 'target_window_id', 'cond_status', 'cond_window_id',
        'cond_epoch_index', 'cond_window_k', 'cond_npz_row', 'signed_offset_sec', 'n_candidates', 'n_before', 'n_after']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', help='外部窗清单 CSV（record_id, epoch_index, window_k）；给出时只配对这些窗')
    ap.add_argument('--out', help='--windows 模式的输出 CSV')
    args = ap.parse_args()

    if args.windows:
        t = pd.read_csv(args.windows)
        miss = {'record_id', 'epoch_index', 'window_k'} - set(t.columns)
        if miss:
            sys.exit(f'窗清单缺少列 {miss}')
        if 'subject_id' not in t:
            t['subject_id'] = t['record_id'].str[:5]
        heldout = set(pd.read_csv(H.OUT_SPLIT).query("pretrain_role == 'heldout_val'").subject_id)
        if set(t.subject_id) & heldout:
            print('警告：窗清单含 held-out 受试者', sorted(set(t.subject_id) & heldout))
        p = pair_table(t[['subject_id', 'record_id', 'epoch_index', 'window_k']].drop_duplicates())
        out = Path(args.out or 'shift_pairs_custom.csv')
        sha = write_csv(p[COLS], out)
        print(json.dumps(coverage(p, out.name), indent=2, ensure_ascii=False))
        print(f'已写入 {out}（sha256 {sha[:16]}…）')
        return

    pool = pair_table(pool_targets('pretrain_train'))
    vw = pd.read_csv(H.OUT_WINDOWS)
    vpos = vw[vw.modality == 'eog'][['subject_id', 'record_id', 'epoch_index', 'window_k']].reset_index(drop=True)
    val = pair_table(vpos)

    pool_path = OUT_CACHE / f'shift_pairs_pretrain_pool_{VERSION}.csv'
    val_path = OUT_REPORT / f'shift_pairs_val_windows_{VERSION}.csv'
    pool_sha, val_sha = write_csv(pool[COLS], pool_path), write_csv(val[COLS], val_path)

    by_subj = pool.groupby('subject_id').agg(n_targets=('n_candidates', 'size'),
                                             n_null=('n_candidates', lambda s: int((s == 0).sum())),
                                             median_candidates=('n_candidates', 'median'),
                                             frac_condition_after=('signed_offset_sec', lambda s: float((s > 0).mean())))
    by_subj['coverage'] = 1 - by_subj['n_null'] / by_subj['n_targets']
    by_subj.reset_index().to_csv(OUT_REPORT / 'coverage_by_subject.csv', index=False, float_format='%.6g', lineterminator='\n')
    has = pool.n_candidates > 0
    grid = np.r_[np.arange(-MAX_GAP, -MIN_GAP + 1, WIN_SEC), np.arange(MIN_GAP, MAX_GAP + 1, WIN_SEC)]
    cnt = pool.loc[has, 'signed_offset_sec'].value_counts().reindex(grid, fill_value=0)
    pd.DataFrame(dict(signed_offset_sec=grid, count=cnt.to_numpy())).to_csv(
        OUT_REPORT / 'offset_hist.csv', index=False, lineterminator='\n')

    summary = dict(version=VERSION, owner='B', seed=PAIR_SEED, gap_sec=[MIN_GAP, MAX_GAP], gap_inclusive=True,
                   offset_sign='条件窗起点 − 目标窗起点（秒）', sampler='splitmix64(window_key ^ splitmix64(seed))，逐窗确定',
                   validity='条件窗所在 30 秒段存在于 A 的预处理 NPZ；不读睡眠标签',
                   inputs={H.OUT_SPLIT.name: H.sha256_file(H.OUT_SPLIT), H.OUT_WINDOWS.name: H.sha256_file(H.OUT_WINDOWS)},
                   pretrain_pool=coverage(pool, pool_path.name), val_windows=coverage(val, val_path.name),
                   files={pool_path.name: pool_sha, val_path.name: val_sha})
    (OUT_REPORT / 'coverage_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary['pretrain_pool'], indent=2, ensure_ascii=False))
    print('val:', summary['val_windows']['coverage'], summary['val_windows']['n_null_no_candidate'])


if __name__ == '__main__':
    main()
