"""新一轮第 2 周 · B 统一评分器：旧头 20 个 + 原人另一夜新头 9 个，在 dev8 回放或预留 8 人上统一推理。

推理直接复用第一周 trainer（experiment3/week1_budget_20261126/train.py）的 load_bundle 与 score：
特征回执、index、数组 SHA 的核对相同；输入 = (特征 − 头自带 mean) / 头自带 std，Linear(384, 5)，softmax，argmax。

--heads：CSV，列 roster, arm, dir（arm ∈ add_people / deepen / other_night；dir 下有 head.pt、normalization.npz）。

两种模式：
  --cohort dev8      开发回放。给 --reference-csv（列 roster, arm, predictions）时逐头核对样本键、argmax 与概率差；
                     --write-lock 写出锁定清单（每个头的 head.pt / normalization.npz SHA-256）。
  --cohort heldout8  预留 8 人（第一周 subject_roles.csv 的 held_out，即 W4 held-out 8 人）。必须给 --lock，
                     且头集合与锁定清单完全一致（10 份 add_people + 10 份 deepen + 9 份 other_night = 29 个，
                     05 没有 other_night），每个文件 SHA 与清单相同；输出目录已存在则拒绝（只评一次）。
                     本脚本只写逐段预测与清单，不计算任何指标；配对表由 52 号单独生成。
输出（--out）：<roster>/<arm>/predictions_<cohort>.csv、scoring_manifest.json，dev8 模式另有 replay_check.csv。

用法：
  python scripts/51_b_w2_unified_scorer.py --cohort dev8 --heads heads.csv --reference-csv refs.csv \
      --bundle <D 共享缓存> --data <C 交接包 data> --write-lock lock.json --out runs/w2_replay
  python scripts/51_b_w2_unified_scorer.py --cohort heldout8 --heads heads.csv --lock lock.json \
      --bundle <D 共享缓存> --data <C 交接包 data> --out runs/w2_heldout8
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent.parent
WEEK1 = BASE / 'experiment3' / 'week1_budget_20261126'
FEATURE_INFO = '392122c602030d8dcfa903a8d5a5f55fd582f0e09dc6d086248c86bac4267bdf'
ARMS = ('add_people', 'deepen', 'other_night')
ROSTERS = [f'{i:02d}' for i in range(1, 11)]
EXPECTED = {(r, a) for r in ROSTERS for a in ARMS if not (r == '05' and a == 'other_night')}


def week1():
    sys.path.insert(0, str(WEEK1))
    spec = importlib.util.spec_from_file_location('w1train', WEEK1 / 'train.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    import protocol
    return m, protocol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cohort', required=True, choices=['dev8', 'heldout8'])
    ap.add_argument('--heads', required=True, type=Path)
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--old-plan', type=Path, default=WEEK1 / 'rosters')
    ap.add_argument('--feature-info-sha256', default=FEATURE_INFO)
    ap.add_argument('--reference-csv', type=Path)
    ap.add_argument('--write-lock', type=Path)
    ap.add_argument('--lock', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    tr, P = week1()

    heads = pd.read_csv(args.heads, dtype=str)
    heads['roster'] = heads['roster'].str.zfill(2)
    if heads.duplicated(['roster', 'arm']).any() or not set(heads['arm']) <= set(ARMS):
        sys.exit('heads.csv 有重复或未知的 arm')
    files = {}
    for h in heads.itertuples():
        d = Path(h.dir)
        files[h.roster, h.arm] = {f: P.sha256(d / f) for f in ('head.pt', 'normalization.npz')}

    if args.cohort == 'heldout8':
        if not args.lock:
            sys.exit('heldout8 必须给 --lock')
        lock = json.loads(args.lock.read_text(encoding='utf-8'))
        if not lock.get('locked'):
            sys.exit('锁定清单未标记 locked: true')
        want = {tuple(k.split('/')): v for k, v in lock['heads'].items()}
        if set(want) != EXPECTED or set(files) != EXPECTED:
            sys.exit(f'头集合不完整或与清单不一致：缺 {sorted(EXPECTED - set(files))}，多 {sorted(set(files) - EXPECTED)}')
        bad = [k for k in EXPECTED if files[k] != want[k]]
        if bad:
            sys.exit(f'SHA 与锁定清单不同：{bad}')
        if args.out.exists():
            sys.exit(f'{args.out} 已存在：预留 8 人只评一次')

    P.require(P.sha256(args.old_plan / 'manifest.json') ==
              json.loads((BASE / 'experiment3/week2_budget_20261127/rosters/manifest.json').read_text())
              ['first_week_plan_sha256'], '第一周名单与第二周匹配清单不对应')
    tr.load_plan(args.old_plan, args.data)
    rows, positions, arrays = tr.load_bundle(args.bundle, args.feature_info_sha256, args.old_plan, args.data)
    roles = {r['subject_id']: r['role'] for r in P.read_csv(args.old_plan / 'subject_roles.csv')}
    target = 'held_out' if args.cohort == 'heldout8' else 'dev'
    members = {s for s, role in roles.items() if role == target}
    ids = [i for i, r in enumerate(rows) if r['subject_id'] in members]
    P.require(len(members) == 8, f'{target} 不是 8 人')
    if args.cohort == 'heldout8':
        P.require({rows[i]['role'] for i in ids} == {'heldout_val'} and len(ids) == 18897, '预留 8 人行数或角色不符')
    else:
        P.require(len(ids) == 21186, 'dev8 行数不符')

    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=args.cohort == 'dev8')
    manifest = dict(script='scripts/51_b_w2_unified_scorer.py', cohort=args.cohort, subjects=sorted(members),
                    n_epochs=len(ids), feature_info_sha256=args.feature_info_sha256, code=tr.code_hashes(),
                    torch=torch.__version__, numpy=np.__version__, metrics_computed=False, test_used=False,
                    lock_sha256=(P.sha256(args.lock) if args.lock else None), heads={})
    for h in heads.sort_values(['roster', 'arm']).itertuples():
        d, out = Path(h.dir), args.out / h.roster / h.arm
        out.mkdir(parents=True, exist_ok=True)
        z = np.load(d / 'normalization.npz')
        model = torch.nn.Linear(384, 5)
        model.load_state_dict(torch.load(d / 'head.pt', map_location='cpu', weights_only=True))
        model.eval()
        tr.score(out, args.cohort, ids, rows, arrays, model, z['mean'], z['std'])
        manifest['heads'][f'{h.roster}/{h.arm}'] = dict(
            **{f'{k}_sha256': v for k, v in files[h.roster, h.arm].items()},
            predictions_sha256=P.sha256(out / f'predictions_{args.cohort}.csv'))
    if args.cohort == 'dev8':
        manifest.pop('metrics_computed')   # dev8 回放只核对，不出指标
    (args.out / 'scoring_manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n',
                                                    encoding='utf-8')

    if args.cohort == 'dev8' and args.reference_csv:
        refs = pd.read_csv(args.reference_csv, dtype=str)
        refs['roster'] = refs['roster'].str.zfill(2)
        res = []
        for r in refs.itertuples():
            a = pd.read_csv(args.out / r.roster / r.arm / 'predictions_dev8.csv')
            b = pd.read_csv(r.predictions)
            keys = list(P.KEY)
            pc = list(tr.PROBS)
            same = a[keys].astype(str).equals(b[keys].astype(str))
            res.append(dict(roster=r.roster, arm=r.arm, keys_identical=same,
                            argmax_exact=bool(same and (a['y_pred'] == b['y_pred']).all()),
                            max_abs_probability_diff=float(np.abs(a[pc].to_numpy() - b[pc].to_numpy()).max()) if same else None))
        rc = pd.DataFrame(res)
        rc.to_csv(args.out / 'replay_check.csv', index=False, lineterminator='\n')
        print(rc.to_string(index=False))
        print('replay all exact:', bool(rc['argmax_exact'].all()))
    if args.write_lock:
        args.write_lock.write_text(json.dumps(dict(
            locked=False, note='第 8 天 9 个新头齐备并经核对后，把 locked 改为 true 再用于 heldout8',
            feature_info_sha256=args.feature_info_sha256,
            heads={f'{r}/{a}': v for (r, a), v in sorted(files.items())},
            complete=set(files) == EXPECTED, missing=sorted(f'{r}/{a}' for r, a in EXPECTED - set(files))),
            indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'{args.cohort}: {len(heads)} heads scored, {len(ids)} epochs each')


if __name__ == '__main__':
    main()
