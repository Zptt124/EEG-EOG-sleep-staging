"""新一轮第 1 周 · 时序线：把每夜切成无缺口的连续段（A0/A1/A2 窗口与 A2 置换共用）。

输入：data/processed/<record>.npz 的 epoch_index（只读训练池 54 人与固定 dev 8 人；测试集 NPZ 不打开，不读任何标签）。
缺口定义：同一记录内相邻两段 epoch_index 之差 > 1。不跨记录（夜）。
角色：pretrain32 / orig14 / heldout8（W4 预训练 held-out，只缓存特征）/ dev8，来自 44 号名单与 W4 预训练划分。

输出（--out 目录）：
  contiguous_segments_t1_v01.csv   subject_id,record_id,role,segment,start_epoch,end_epoch,n_epochs
                                   （start/end 为原始 epoch_index，含两端；segment 在记录内从 0 编号）
  contiguous_segments_t1_v01.json  规则、来源 SHA、各角色的记录数 / 段数 / 段长分布 / 缺口数、输出 SHA
可选 --check-features <W4 特征目录>：核对特征 index.csv 中每条记录的 epoch 集合与 NPZ 一致。

用法：
  python scripts/45_b_t1_contiguous_segments.py --out manifests \
      [--check-features cache/w4_xmodal_dl/w4_d_main30k_feat_pool1s_v01/W4_D_main_m64x12_seed0_v01/step30000_ema_model_pool1s]
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
NPZ_DIR = BASE / 'data' / 'processed'
ROSTERS = BASE / 'manifests' / 'temporal_rosters_t1_v01.csv'
SPLIT = BASE / 'manifests' / 'split_subjects.csv'
PRETRAIN = BASE / 'manifests' / 'pretrain_split_w4_v01.csv'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def segments(ep):
    """升序 epoch_index -> [(start, end)]，含两端。"""
    ep = np.asarray(ep)
    if (np.diff(ep) <= 0).any():
        raise SystemExit('epoch_index 不是严格升序')
    cut = np.flatnonzero(np.diff(ep) > 1)
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(ep) - 1]
    return [(int(ep[s]), int(ep[e])) for s, e in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check-features')
    args = ap.parse_args()
    out = Path(args.out)

    split = {r['subject_id']: r for r in read(SPLIT)}
    role = {}
    for r in read(ROSTERS):
        if r['role'] == 'pretrain32':
            role[r['subject_id']] = 'pretrain32'
        elif r['role'] in ('label', 'eval9'):
            role[r['subject_id']] = 'orig14'
        elif r['role'] == 'dev8':
            role[r['subject_id']] = 'dev8'
    for r in read(PRETRAIN):
        if r['pretrain_role'] == 'heldout_val':
            role[r['subject_id']] = 'heldout8'
    assert sum(v == 'orig14' for v in role.values()) == 14 and sum(v == 'dev8' for v in role.values()) == 8
    assert sum(v == 'pretrain32' for v in role.values()) == 32 and sum(v == 'heldout8' for v in role.values()) == 8
    assert not any(split[s]['split'] == 'test' for s in role), '名单里出现测试集受试者'

    rows, record_epochs = [], {}
    for s in sorted(role):
        for rec in split[s]['record_ids'].split(';'):
            with np.load(NPZ_DIR / f'{rec}.npz') as z:
                ep = z['epoch_index'].astype(np.int64)   # 只读 epoch_index
            record_epochs[rec] = set(ep.tolist())
            for k, (a, b) in enumerate(segments(ep)):
                rows.append(dict(subject_id=s, record_id=rec, role=role[s], segment=k,
                                 start_epoch=a, end_epoch=b, n_epochs=b - a + 1))

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'contiguous_segments_t1_v01.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    summary = {}
    by_role = defaultdict(list)
    for r in rows:
        by_role[r['role']].append(r)
    for k, rs in sorted(by_role.items()):
        n = np.array([r['n_epochs'] for r in rs])
        recs = {r['record_id'] for r in rs}
        summary[k] = dict(subjects=len({r['subject_id'] for r in rs}), records=len(recs), segments=len(rs),
                          gaps=len(rs) - len(recs), records_with_gap=len(recs) - sum(
                              1 for rec in recs if sum(r['record_id'] == rec for r in rs) == 1),
                          epochs=int(n.sum()), seg_len_min=int(n.min()), seg_len_median=float(np.median(n)),
                          seg_len_max=int(n.max()), segments_shorter_than_5=int((n < 5).sum()),
                          epochs_in_segments_shorter_than_5=int(n[n < 5].sum()))

    feat_check = None
    if args.check_features:
        idx = read(Path(args.check_features) / 'index.csv')
        fe = defaultdict(set)
        for r in idx:
            fe[r['record_id']].add(int(r['epoch_index']))
        bad = {rec: dict(only_npz=len(record_epochs[rec] - e), only_features=len(e - record_epochs[rec]))
               for rec, e in fe.items() if e != record_epochs.get(rec)}
        feat_check = dict(features_dir=str(Path(args.check_features)), records=len(fe), n_rows=len(idx),
                          index_sha256=sha256(Path(args.check_features) / 'index.csv'),
                          mismatched_records=bad)

    info = dict(script='scripts/45_b_t1_contiguous_segments.py',
                rule='same record; gap = consecutive epoch_index difference > 1; start/end inclusive',
                sources=dict(rosters=dict(path=str(ROSTERS.relative_to(BASE)), sha256=sha256(ROSTERS)),
                             split_subjects=dict(path=str(SPLIT.relative_to(BASE)), sha256=sha256(SPLIT)),
                             pretrain_split=dict(path=str(PRETRAIN.relative_to(BASE)), sha256=sha256(PRETRAIN)),
                             npz='data/processed/<record>.npz (epoch_index only; labels not read; test not opened)'),
                summary=summary, feature_index_check=feat_check, output_sha256=sha256(csv_path))
    (out / 'contiguous_segments_t1_v01.json').write_text(json.dumps(info, indent=1, ensure_ascii=False) + '\n',
                                                         encoding='utf-8')
    for k, v in summary.items():
        print(k, v)
    if feat_check:
        print('feature index check:', feat_check['records'], 'records,', len(feat_check['mismatched_records']),
              'mismatched', feat_check['mismatched_records'])


if __name__ == '__main__':
    main()
