"""D 独立复核：新一轮第 1 周时序线（A0 / A1 / A2 × 2 份名单 × 3 head seed）的 8 / 9 / 17 人评分和短 N1。

只读 A 交付的逐段预测（reports/w1_temporal/runs/*/predictions_{dev8,eval9}.csv）、B 的名单和连续段文件、
D 共享缓存的 index.csv（取真值和样本键做核对）。指标全部自己实现，不调用 A/B 的 44–50 号脚本。

核对内容：
  1. 数据边界：每个 run 的 dev8 / eval9 覆盖 = 缓存里对应受试者的全部 epoch；样本键、真值与缓存一致；
     三个条件同名单同 seed 的行序和真值一致；概率合法（非负、和为 1）、y_pred = argmax。
     名单：label 5 人 ⊂ 原 14 人；eval9 = 原 14 人 − label；dev8 = 固定 dev；A1/A2 预训练 32 人与 14 人、held-out、dev 不相交。
  2. 每个 run：Macro-F1、N1 F1、N1 AP（sklearn average_precision_score），三个评分集 dev8 / eval9 / merged17。
  3. 短 N1：真值中长度 1–2 的连续 N1 段（同一记录、epoch_index 连续）；召回 = 被预测为 N1 的比例，
     另报被判成 W / N2 的比例。段的边界同时试"按记录内 epoch 连续"和"按 B 的连续段文件"两种定义。
  4. 配对差：逐 seed 差值及 3 seed 均值；把 3 个 seed 的预测拼在一起后按受试者 bootstrap（1000 次）求 95% 区间。
输出：<out>/per_run.csv、paired.csv、short_n1.csv、checks.json，与 A 的 eval/ 表逐项对比的 compare_with_A.csv。
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

CONDS = ('A0', 'A1', 'A2')
ROSTERS = ('1', '2')
SEEDS = (0, 1, 2)
GROUPS = ('dev8', 'eval9', 'merged17')
PROBS = ('p_W', 'p_N1', 'p_N2', 'p_N3', 'p_REM')
CONTRASTS = (('A1', 'A0'), ('A1', 'A2'), ('A2', 'A0'))


def fail(msg):
    print(f'ERROR: {msg}', file=sys.stderr, flush=True)
    sys.exit(1)


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_dir(runs, cond, roster, seed):
    prefix = 'T1_A' if cond == 'A1' or (cond == 'A0' and roster == '1') else 'T1_B'
    d = runs / f'{prefix}_{cond}_r{roster}_seed{seed}_v01'
    if not d.exists():
        fail(f'Missing run directory {d.name}')
    return d


def load_preds(d, group):
    rows = read_csv(d / f'predictions_{group}.csv')
    p = np.array([[float(r[k]) for k in PROBS] for r in rows])
    return dict(sample_id=np.array([r['sample_id'] for r in rows]),
                subject=np.array([r['subject_id'] for r in rows]),
                record=np.array([r['record_id'] for r in rows]),
                epoch=np.array([int(r['epoch_index']) for r in rows]),
                y=np.array([int(r['y_true']) for r in rows]),
                pred=np.array([int(r['y_pred']) for r in rows]), p=p)


def concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def metrics(y, p, pred=None):
    pred = p.argmax(1) if pred is None else pred
    return dict(macro_f1=f1_score(y, pred, labels=range(5), average='macro', zero_division=0),
                n1_f1=f1_score(y == 1, pred == 1, zero_division=0),
                n1_ap=average_precision_score(y == 1, p[:, 1]))


def short_n1_mask(d, seg_lookup=None):
    """真值中长度 1–2 的 N1 段。seg_lookup=None：同一记录内 epoch_index 连续即算连续；否则还要同一连续段。"""
    mask = np.zeros(len(d['y']), bool)
    order = np.lexsort((d['epoch'], d['record']))
    rec, ep, y = d['record'][order], d['epoch'][order], d['y'][order]
    seg = np.array([seg_lookup(r, e) for r, e in zip(rec, ep)]) if seg_lookup else np.zeros(len(y), int)
    i = 0
    while i < len(y):
        if y[i] != 1:
            i += 1
            continue
        j = i
        while (j + 1 < len(y) and y[j + 1] == 1 and rec[j + 1] == rec[i] and ep[j + 1] == ep[j] + 1
               and seg[j + 1] == seg[i]):
            j += 1
        if j - i + 1 <= 2:
            mask[order[i:j + 1]] = True
        i = j + 1
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, help='checkout of feat/a-w1-temporal')
    ap.add_argument('--cache-index', required=True, help='D shared cache index.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--boot-seed', type=int, default=0)
    args = ap.parse_args()
    repo, out = Path(args.repo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = repo / 'reports/w1_temporal/runs'
    checks = {}

    # ---- 名单与边界 ----
    roster_rows = read_csv(repo / 'manifests/temporal_rosters_t1_v01.csv')
    roles = defaultdict(lambda: defaultdict(set))
    for r in roster_rows:
        roles[r['roster']][r['role']].add(r['subject_id'])
    seg_rows = read_csv(repo / 'manifests/contiguous_segments_t1_v01.csv')
    seg_roles = defaultdict(set)
    for r in seg_rows:
        seg_roles[r['role']].add(r['subject_id'])
    orig14 = seg_roles['orig14']
    dev = seg_roles['dev8']
    for k in ROSTERS:
        ro = roles[k]
        if not (len(ro['label']) == 5 and ro['label'] <= orig14 and ro['eval9'] == orig14 - ro['label']
                and ro['dev8'] == dev and len(dev) == 8):
            fail(f'Roster {k} sets are inconsistent')
    pre = seg_roles['pretrain32']
    if len(pre) != 32 or pre & (orig14 | dev | seg_roles['heldout8']):
        fail('pretrain32 overlaps the 14 labelled, dev or held-out subjects')
    checks['rosters'] = dict(label_r1=sorted(roles['1']['label']), label_r2=sorted(roles['2']['label']),
                             label_overlap=sorted(roles['1']['label'] & roles['2']['label']),
                             pretrain32_disjoint=True)

    segs = defaultdict(list)
    for r in seg_rows:
        segs[r['record_id']].append((int(r['start_epoch']), int(r['end_epoch']), int(r['segment'])))

    def seg_lookup(rec, ep):
        for s, e, g in segs[rec]:
            if s <= ep <= e:
                return g
        fail(f'Epoch {rec} {ep} not in any contiguous segment')

    # ---- 缓存真值 ----
    cache = {r['sample_id']: (r['subject_id'], r['record_id'], int(r['epoch_index']), int(r['y_true']))
             for r in read_csv(args.cache_index)}
    by_subject = defaultdict(set)
    for sid, (subj, _, _, _) in cache.items():
        by_subject[subj].add(sid)

    # ---- 读预测并核对 ----
    P = {}
    for k in ROSTERS:
        for s in SEEDS:
            for c in CONDS:
                d = run_dir(runs, c, k, s)
                for g in ('dev8', 'eval9'):
                    x = load_preds(d, g)
                    want = set().union(*(by_subject[u] for u in roles[k][g]))
                    if set(x['sample_id']) != want or len(x['sample_id']) != len(want):
                        fail(f'{d.name} {g}: coverage differs from the cache ({len(x["sample_id"])} vs {len(want)})')
                    for i, sid in enumerate(x['sample_id']):
                        subj, rec, ep, y = cache[sid]
                        if (subj, rec, ep, y) != (x['subject'][i], x['record'][i], x['epoch'][i], x['y'][i]):
                            fail(f'{d.name} {g}: key or label mismatch at {sid}')
                    if (x['p'] < 0).any() or np.abs(x['p'].sum(1) - 1).max() > 1e-4:
                        fail(f'{d.name} {g}: invalid probabilities')
                    if (x['p'].argmax(1) != x['pred']).mean() > 0:
                        fail(f'{d.name} {g}: y_pred differs from argmax')
                    P[c, k, s, g] = x
                P[c, k, s, 'merged17'] = concat(P[c, k, s, 'dev8'], P[c, k, s, 'eval9'])
            for g in GROUPS:
                ref = P['A0', k, s, g]['sample_id']
                if any(not np.array_equal(P[c, k, s, g]['sample_id'], ref) for c in CONDS):
                    fail(f'Row order differs across conditions (roster {k} seed {s} {g})')
    checks['predictions'] = 'coverage, keys, labels, probabilities and argmax all consistent with the D cache'
    print('checks passed', flush=True)

    # ---- 每个 run 的指标 ----
    per_run, M = [], {}
    for (c, k, s, g), x in sorted(P.items()):
        m = metrics(x['y'], x['p'], x['pred'])
        M[c, k, s, g] = m
        per_run.append(dict(cond=c, roster=k, seed=s, group=g, n=len(x['y']),
                            **{kk: round(v, 6) for kk, v in m.items()}))
    write_csv(out / 'per_run.csv', per_run)

    # ---- 短 N1 ----
    short_rows = []
    for (c, k, s, g), x in sorted(P.items()):
        for mode, lk in (('record_consecutive', None), ('b_segments', seg_lookup)):
            mk = short_n1_mask(x, lk)
            pr = x['pred'][mk]
            short_rows.append(dict(cond=c, roster=k, seed=s, group=g, definition=mode, n_short=int(mk.sum()),
                                   recall=round(float((pr == 1).mean()), 6), to_W=round(float((pr == 0).mean()), 6),
                                   to_N2=round(float((pr == 2).mean()), 6)))
    write_csv(out / 'short_n1.csv', short_rows)

    # ---- 配对差：逐 seed + 拼 seed 后按受试者 bootstrap ----
    rng = np.random.default_rng(args.boot_seed)
    paired = []
    for k in ROSTERS:
        for g in GROUPS:
            pooled = {c: {key: np.concatenate([P[c, k, s, g][key] for s in SEEDS]) for key in ('y', 'p', 'subject')}
                      for c in CONDS}
            subj = pooled['A0']['subject']
            uniq = np.unique(subj)
            idx = {u: np.flatnonzero(subj == u) for u in uniq}
            boots = [np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(args.n_boot)]
            full_m = {c: metrics(pooled[c]['y'], pooled[c]['p']) for c in CONDS}
            boot_m = {c: [metrics(pooled[c]['y'][i], pooled[c]['p'][i]) for i in boots] for c in CONDS}
            for a, b in CONTRASTS:
                row = dict(roster=k, group=g, contrast=f'{a}-{b}')
                for met in ('n1_ap', 'macro_f1', 'n1_f1'):
                    ds = [100 * (M[a, k, s, g][met] - M[b, k, s, g][met]) for s in SEEDS]
                    full = 100 * (full_m[a][met] - full_m[b][met])
                    bd = np.array([100 * (x[met] - z[met]) for x, z in zip(boot_m[a], boot_m[b])])
                    row.update({f'{met}_per_seed': ' '.join(f'{v:+.2f}' for v in ds),
                                f'{met}_seed_mean': round(float(np.mean(ds)), 3),
                                f'{met}_pooled': round(full, 3),
                                f'{met}_lo': round(float(np.percentile(bd, 2.5)), 3),
                                f'{met}_hi': round(float(np.percentile(bd, 97.5)), 3),
                                f'{met}_frac_pos': round(float((bd > 0).mean()), 3)})
                paired.append(row)
            print(f'roster {k} {g} done', flush=True)
    write_csv(out / 'paired.csv', paired)

    # ---- 和 A 的表对比 ----
    cmp_rows = []
    a_sum = {(r['roster'], r['group'], r['contrast']): r for r in read_csv(repo / 'reports/w1_temporal/eval/summary.csv')}
    for r in paired:
        a = a_sum.get((r['roster'], r['group'], r['contrast']))
        if a:
            for met, ak in (('n1_ap', 'mean_d_n1_ap'), ('macro_f1', 'mean_d_macro'), ('n1_f1', 'mean_d_n1_f1')):
                cmp_rows.append(dict(source='summary.csv', roster=r['roster'], group=r['group'], contrast=r['contrast'],
                                     metric=met, A=float(a[ak]), D=r[f'{met}_seed_mean'],
                                     abs_diff=round(abs(float(a[ak]) - r[f'{met}_seed_mean']), 4)))
    a_pd = read_csv(repo / 'reports/w1_temporal/eval/paired_dev8.csv')
    for a in a_pd:
        k = a['budget'].lstrip('r')
        r = next(x for x in paired if x['roster'] == k and x['group'] == 'dev8' and x['contrast'] == a['contrast'])
        for met, ak in (('n1_ap', 'delta_n1_ap'), ('macro_f1', 'delta_macro_f1'), ('n1_f1', 'delta_n1_f1')):
            for suf, rk in (('', 'pooled'), ('_lo', 'lo'), ('_hi', 'hi')):
                cmp_rows.append(dict(source='paired_dev8.csv', roster=k, group='dev8', contrast=a['contrast'],
                                     metric=f'{met}_{rk}', A=round(100 * float(a[ak + suf]), 3), D=r[f'{met}_{rk}'],
                                     abs_diff=round(abs(100 * float(a[ak + suf]) - r[f'{met}_{rk}']), 4)))
    a_sn = read_csv(repo / 'reports/w1_temporal/eval/short_n1.csv')
    for a in a_sn:
        for mode in ('record_consecutive', 'b_segments'):
            r = next(x for x in short_rows if x['cond'] == a['cond'] and x['roster'] == a['roster']
                     and str(x['seed']) == a['seed'] and x['group'] == a['group'] and x['definition'] == mode)
            for met in ('n_short', 'recall', 'to_W', 'to_N2'):
                cmp_rows.append(dict(source=f'short_n1.csv/{mode}', roster=a['roster'], group=a['group'],
                                     contrast=f'{a["cond"]}_seed{a["seed"]}', metric=met, A=float(a[met]), D=r[met],
                                     abs_diff=round(abs(float(a[met]) - r[met]), 6)))
    write_csv(out / 'compare_with_A.csv', cmp_rows)
    summ = defaultdict(lambda: [0, 0.0])
    for r in cmp_rows:
        key = (r['source'], r['metric'].split('_lo')[0].split('_hi')[0])
        summ[key][0] += 1
        summ[key][1] = max(summ[key][1], r['abs_diff'])
    checks['compare_with_A_max_abs_diff'] = {f'{s} | {m}': dict(n=n, max_abs_diff=d) for (s, m), (n, d) in summ.items()}
    with open(out / 'checks.json', 'w', encoding='utf-8') as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)
    for k, v in checks['compare_with_A_max_abs_diff'].items():
        print(f'{k}: n={v["n"]} max_abs_diff={v["max_abs_diff"]}', flush=True)
    print('RECHECK DONE', flush=True)


if __name__ == '__main__':
    main()
