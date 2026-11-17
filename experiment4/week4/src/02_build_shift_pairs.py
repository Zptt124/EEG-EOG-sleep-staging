"""生成错时（shifted）EEG 条件的配对表（C 的跨模态规则，CPU，约 1 分钟）。

规则（experiment3/week4/cross_modal_protocol.md）：对每个目标窗，从同一记录、完整有效、起点间隔
|Δ| ∈ [60, 300] 秒的候选窗中，用固定种子均匀抽一个；记录目标窗 ID、条件窗 ID 和有符号间隔；
没有候选的目标窗置空并登记数量。睡眠标签不参与。

配对按“窗位置”（同一 30 秒段的同一 5 秒窗）做，与模态无关；训练时只有 EOG 目标窗会用到它。
覆盖整个 54 人缓存（held-out 的人也有配对，供带条件的 val 使用）。

输出（SLEEP_W4_BIG/shift_pairs/）：
  shift_pairs_60_300_v01.npy      [n_rows*6] int64，条件窗位置，-1 = 无候选（训练脚本读这个）
  shift_pairs_60_300_v01.csv.gz   target_window_id, cond_window_id, signed_gap_sec（窗 ID = sample_id_k）
  shift_pairs_60_300_v01_summary.json  覆盖率、方向、间隔分布、SHA256
"""
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import w4d_common as C


def main():
    if C.PAIRS_NPY.exists():
        C.fail(f'{C.PAIRS_NPY} already exists; delete it first if you really want to rebuild')
    pool = C.TrainPool(load_to_ram=False)
    pair, gap = C.build_shift_pairs(pool)
    C.PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(C.PAIRS_NPY, pair)

    sid = np.repeat(np.array([r['sample_id'] for r in pool.index]), C.N_WIN)
    k = np.tile(np.arange(C.N_WIN), pool.n_rows)
    wid = np.char.add(np.char.add(sid, '_'), k.astype(str))
    csv_path = C.PAIRS_NPY.with_suffix('.csv.gz')
    with gzip.open(csv_path, 'wt', encoding='utf-8', newline='') as f:
        f.write('target_window_id,cond_window_id,signed_gap_sec\n')
        for i in range(len(pair)):
            if pair[i] >= 0:
                f.write(f'{wid[i]},{wid[pair[i]]},{gap[i]}\n')
            else:
                f.write(f'{wid[i]},,\n')

    ok = pair >= 0
    g = gap[ok]
    # |Δ| 分段：[60,120) [120,180) [180,240) [240,300]，正负方向分开
    hist_edges = [60, 120, 180, 240, 301]
    hist_neg = np.histogram(-g[g < 0], bins=hist_edges)[0]
    hist_pos = np.histogram(g[g > 0], bins=hist_edges)[0]
    summ = dict(
        rule='same record, |start gap| in [60, 300] s, both directions, uniform over candidate 5-s windows',
        pair_seed=C.PAIR_SEED, min_sec=C.SHIFT_MIN_SEC, max_sec=C.SHIFT_MAX_SEC,
        n_windows=int(len(pair)), n_paired=int(ok.sum()), n_no_candidate=int((~ok).sum()),
        coverage=float(ok.mean()), n_negative_gap=int((g < 0).sum()), n_positive_gap=int((g > 0).sum()),
        gap_abs_mean=float(np.abs(g).mean()), gap_abs_median=float(np.median(np.abs(g))),
        n_at_boundary_60=int((np.abs(g) == 60).sum()), n_at_boundary_300=int((np.abs(g) == 300).sum()),
        abs_gap_hist_edges_sec=hist_edges, hist_counts_negative=hist_neg.tolist(), hist_counts_positive=hist_pos.tolist(),
        cache_sha256=pool.cfg['sha256_signals'],
        sha256_npy=C.sha256_file(C.PAIRS_NPY), sha256_csv_gz=C.sha256_file(csv_path))
    C.write_json(C.PAIRS_NPY.parent / (C.PAIRS_NPY.stem + '_summary.json'), summ)
    C.write_json(C.MANIFESTS / (C.PAIRS_NPY.stem + '_summary.json'), summ)
    for kk, v in summ.items():
        print(f'{kk}: {v}', flush=True)
    print('SHIFT PAIRS DONE', flush=True)


if __name__ == '__main__':
    main()
