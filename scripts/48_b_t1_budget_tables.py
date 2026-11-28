"""新一轮第 1 周 · 预算线：B 重算两套预算主表（36 人主表与固定 dev 8 人），同一受试者跨名单共用 bootstrap 权重。

输入：C/D 用 experiment3/week1_budget_20261126/train.py 跑出的 <runs>/<名单>/paired/{base,add_people,deepen}/
      （predictions_pool36.csv、predictions_dev8.csv、config.json）与 <名单>/paired/summary.json，以及名单目录 rosters/。
逐份核对：
  1. 三组预测的样本键（sample_id, subject_id, record_id, epoch_index）顺序与真值完全相同；y_pred = 概率 argmax；
     评估人 = evaluation_subjects.csv 对应 cohort；dev8 真值与 val_reference 一致。
  2. 原始 Macro-F1（五类等权，sklearn 口径，无预测的类 F1 记 0）重算后与 summary.json 一致，三个差值（pp）一致。
  3. config：更新步数 base 300、加人 450、加深 450；全部名单的特征回执、计划、代码指纹、seed、轮数相同。
主表（每个 cohort 一张）：每份名单的三组 Macro-F1 与 加人−加深、加人−起点、加深−起点（pp），以及跨名单平均。
不确定性：每个受试者每次抽一个 Poisson(1) 权重，在所有名单里共用（同一受试者跨名单共享 bootstrap 权重），
          用加权混淆矩阵算 Macro-F1，1000 次，rng seed 0；报告逐名单与跨名单平均差的 95% 区间。
方向核对：同号名单数；逐人剔除（从所有名单中去掉一人）后平均差的最小值、最大值与方向是否不变；
          最大、最小配对差；每人被评估的名单数。复验参考信号：|平均差| ≥ 1 pp、≥ 7/10 份同向、逐人剔除方向稳定；
          dev8 与 36 人主表方向相反时记为待复核。
输出（--out）：main_table_pool36.csv、main_table_dev8.csv、checks.csv、evaluator_counts.csv、loso_pool36.csv、
              loso_dev8.csv、summary.json

用法：
  python scripts/48_b_t1_budget_tables.py --runs <C 与 D 的 runs 根目录> \
      --plan experiment3/week1_budget_20261126/rosters --out reports/t1/b_budget_tables
  （C 与 D 的输出若在不同目录，可多次给 --runs，按名单号合并）
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('base', 'add_people', 'deepen')
COHORTS = ('pool36', 'dev8')
KEY = ('sample_id', 'subject_id', 'record_id', 'epoch_index')
PROBS = ('p_W', 'p_N1', 'p_N2', 'p_N3', 'p_REM')
UPDATES = {'base': 300, 'add_people': 450, 'deepen': 450}
CONTRASTS = {'add_minus_deepen': ('add_people', 'deepen'), 'add_gain': ('add_people', 'base'),
             'deepen_gain': ('deepen', 'base')}
BASE = Path(__file__).resolve().parent.parent
N_BOOT, TOL = 1000, 1e-9


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def macro_f1_cm(cm):
    """cm [..., 5, 5]（行真值、列预测）-> 五类等权 Macro-F1；分母为 0 的类记 0（与 sklearn zero_division=0 一致）。"""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(-2) - tp
    fn = cm.sum(-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den > 0)
    return f1.mean(-1)


def subject_cms(table, subjects):
    y = np.array([int(r['y_true']) for r in table])
    p = np.array([int(r['y_pred']) for r in table])
    s = np.array([r['subject_id'] for r in table])
    out = np.zeros((len(subjects), 5, 5))
    for j, sub in enumerate(subjects):
        m = s == sub
        np.add.at(out[j], (y[m], p[m]), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', action='append', required=True)
    ap.add_argument('--plan', default=str(BASE / 'experiment3' / 'week1_budget_20261126' / 'rosters'))
    ap.add_argument('--rosters', nargs='*', default=[f'{i:02d}' for i in range(1, 11)])
    ap.add_argument('--dev-reference', default=str(BASE / 'manifests' / 'val_reference_v1.csv'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    plan, out = Path(args.plan), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev_truth = {r['sample_id']: int(r['y_true']) for r in read_csv(args.dev_reference)}

    run_dir = {}
    for root in args.runs:
        for r in args.rosters:
            d = Path(root) / r / 'paired'
            if d.exists():
                if r in run_dir:
                    raise SystemExit(f'名单 {r} 在多个 --runs 目录里都有')
                run_dir[r] = d
    if missing := [r for r in args.rosters if r not in run_dir]:
        raise SystemExit(f'缺名单输出: {missing}')

    checks, macro, cms, members, fingerprints = [], {}, {}, {}, {}
    for r in args.rosters:
        d = run_dir[r]
        summ = json.loads((d / 'summary.json').read_text(encoding='utf-8'))
        ev_sub = {c: sorted(x['subject_id'] for x in read_csv(plan / r / 'evaluation_subjects.csv') if x['cohort'] == c)
                  for c in COHORTS}
        chk = dict(roster=r)
        for arm in ARMS:
            cfg = json.loads((d / arm / 'config.json').read_text(encoding='utf-8'))
            chk[f'updates_{arm}'] = cfg['updates']
            fingerprints.setdefault('feature_info_sha256', set()).add(cfg['feature_info_sha256'])
            fingerprints.setdefault('plan_sha256', set()).add(cfg['plan_sha256'])
            fingerprints.setdefault('code', set()).add(json.dumps(cfg['code'], sort_keys=True))
            fingerprints.setdefault('seed_epochs', set()).add((cfg['seed'], cfg['epochs']))
        chk['updates_ok'] = all(chk[f'updates_{a}'] == UPDATES[a] for a in ARMS)
        for c in COHORTS:
            tabs = {a: read_csv(d / a / f'predictions_{c}.csv') for a in ARMS}
            keys = [tuple(x[k] for k in KEY) + (x['y_true'],) for x in tabs['base']]
            chk[f'{c}_keys_identical'] = all([tuple(x[k] for k in KEY) + (x['y_true'],) for x in tabs[a]] == keys
                                             for a in ARMS)
            chk[f'{c}_argmax_ok'] = all(int(x['y_pred']) == int(np.argmax([float(x[p]) for p in PROBS]))
                                        for a in ARMS for x in tabs[a])
            chk[f'{c}_subjects_ok'] = sorted({x['subject_id'] for x in tabs['base']}) == ev_sub[c]
            chk[f'{c}_n_subjects'] = len(ev_sub[c])
            if c == 'dev8':
                chk['dev8_truth_ok'] = (len(tabs['base']) == len(dev_truth) and
                                        all(dev_truth.get(x['sample_id']) == int(x['y_true']) for x in tabs['base']))
            members[r, c] = ev_sub[c]
            for a in ARMS:
                cms[r, c, a] = subject_cms(tabs[a], ev_sub[c])
                macro[r, c, a] = float(macro_f1_cm(cms[r, c, a].sum(0)))
                chk[f'{c}_d_macro_{a}'] = macro[r, c, a] - summ[c]['macro_f1'][a]
            for name, key in (('add_minus_deepen', 'add_people_minus_deepen_pp'), ('add_gain', 'add_people_gain_pp'),
                              ('deepen_gain', 'deepen_gain_pp')):
                a, b = CONTRASTS[name]
                chk[f'{c}_d_{name}_pp'] = 100 * (macro[r, c, a] - macro[r, c, b]) - summ[c][key]
        chk['all_ok'] = bool(chk['updates_ok'] and all(chk[f'{c}_{k}'] for c in COHORTS
                                                       for k in ('keys_identical', 'argmax_ok', 'subjects_ok'))
                             and chk['dev8_truth_ok'] and all(abs(v) < TOL for k, v in chk.items() if '_d_' in k))
        checks.append(chk)
    fp_ok = {k: len(v) == 1 for k, v in fingerprints.items()}
    write_csv(out / 'checks.csv', checks)

    rng = np.random.default_rng(0)
    summary = dict(script='scripts/48_b_t1_budget_tables.py', rosters=args.rosters,
                   runs={r: str(run_dir[r]) for r in args.rosters}, n_all_ok=sum(c['all_ok'] for c in checks),
                   fingerprints_identical=fp_ok,
                   feature_info_sha256=sorted(fingerprints['feature_info_sha256']),
                   bootstrap=f'{N_BOOT} 次 Poisson(1) 受试者权重，同一受试者跨名单共用，rng seed 0', cohorts={})
    for c in COHORTS:
        universe = sorted(set().union(*(members[r, c] for r in args.rosters)))
        upos = {s: i for i, s in enumerate(universe)}
        wts = rng.poisson(1.0, size=(N_BOOT, len(universe))).astype(float)

        def diffs(weights, a, b):
            """weights [B, U] -> [B, n_rosters] 的 Macro-F1 差（pp）。"""
            res = []
            for r in args.rosters:
                ix = [upos[s] for s in members[r, c]]
                wa = np.einsum('bs,sij->bij', weights[:, ix], cms[r, c, a])
                wb = np.einsum('bs,sij->bij', weights[:, ix], cms[r, c, b])
                res.append(100 * (macro_f1_cm(wa) - macro_f1_cm(wb)))
            return np.stack(res, axis=1)

        table, csum = [], {}
        point = {k: np.array([100 * (macro[r, c, a] - macro[r, c, b]) for r in args.rosters])
                 for k, (a, b) in CONTRASTS.items()}
        boot = {k: diffs(wts, a, b) for k, (a, b) in CONTRASTS.items()}
        for i, r in enumerate(args.rosters):
            row = dict(roster=r, n_subjects=len(members[r, c]),
                       **{f'macro_f1_{a}': round(macro[r, c, a], 6) for a in ARMS})
            for k in CONTRASTS:
                lo, hi = np.percentile(boot[k][:, i], [2.5, 97.5])
                row.update({f'{k}_pp': round(point[k][i], 4), f'{k}_lo': round(lo, 4), f'{k}_hi': round(hi, 4)})
            table.append(row)
        mrow = dict(roster='mean', n_subjects=len(universe),
                    **{f'macro_f1_{a}': round(float(np.mean([macro[r, c, a] for r in args.rosters])), 6) for a in ARMS})
        for k in CONTRASTS:
            lo, hi = np.percentile(boot[k].mean(1), [2.5, 97.5])
            mrow.update({f'{k}_pp': round(point[k].mean(), 4), f'{k}_lo': round(lo, 4), f'{k}_hi': round(hi, 4)})
        table.append(mrow)
        write_csv(out / f'main_table_{c}.csv', table)

        loso = []
        for s in universe:
            wl = np.ones((1, len(universe)))
            wl[0, upos[s]] = 0
            loso.append(dict(dropped=s, n_rosters=sum(s in members[r, c] for r in args.rosters),
                             **{f'{k}_mean_pp': round(float(diffs(wl, a, b).mean()), 4)
                                for k, (a, b) in CONTRASTS.items()}))
        write_csv(out / f'loso_{c}.csv', loso)
        amd = point['add_minus_deepen']
        lm = np.array([x['add_minus_deepen_mean_pp'] for x in loso])
        sign = np.sign(amd.mean())
        csum[c] = dict(
            universe_subjects=len(universe),
            mean_pp={k: round(float(v.mean()), 4) for k, v in point.items()},
            mean_ci_pp={k: [round(float(x), 4) for x in np.percentile(boot[k].mean(1), [2.5, 97.5])] for k in CONTRASTS},
            add_minus_deepen=dict(n_positive=int((amd > 0).sum()), n_negative=int((amd < 0).sum()),
                                  max_pp=round(float(amd.max()), 4), min_pp=round(float(amd.min()), 4),
                                  loso_min_pp=round(float(lm.min()), 4), loso_max_pp=round(float(lm.max()), 4),
                                  loso_sign_stable=bool(sign != 0 and (np.sign(lm) == sign).all())))
        summary['cohorts'][c] = csum[c]
        if c == 'pool36':
            write_csv(out / 'evaluator_counts.csv',
                      [dict(subject_id=s, n_rosters_evaluated=sum(s in members[r, c] for r in args.rosters))
                       for s in universe])

    p, dv = summary['cohorts']['pool36'], summary['cohorts']['dev8']
    amd = p['add_minus_deepen']
    same_dir = max(amd['n_positive'], amd['n_negative'])
    summary['retest_signal'] = dict(
        abs_mean_ge_1pp=abs(p['mean_pp']['add_minus_deepen']) >= 1.0,
        same_direction_ge_7_of_10=same_dir >= 7, same_direction_count=same_dir,
        loso_sign_stable=amd['loso_sign_stable'],
        dev8_direction_agrees=bool(np.sign(dv['mean_pp']['add_minus_deepen']) == np.sign(p['mean_pp']['add_minus_deepen'])))
    rs = summary['retest_signal']
    summary['retest_signal']['verdict'] = (
        '待复核（dev8 与主表方向相反）' if not rs['dev8_direction_agrees'] else
        '达到复验参考信号' if rs['abs_mean_ge_1pp'] and rs['same_direction_ge_7_of_10'] and rs['loso_sign_stable'] else
        '未达到复验参考信号')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"checks all_ok {summary['n_all_ok']}/{len(checks)}; fingerprints identical {fp_ok}")
    for c in COHORTS:
        s = summary['cohorts'][c]
        print(f"{c}: 加人−加深 {s['mean_pp']['add_minus_deepen']:+.2f} pp {s['mean_ci_pp']['add_minus_deepen']}, "
              f"加人增益 {s['mean_pp']['add_gain']:+.2f}, 加深增益 {s['mean_pp']['deepen_gain']:+.2f}; "
              f"同号 +{s['add_minus_deepen']['n_positive']}/−{s['add_minus_deepen']['n_negative']}, "
              f"LOSO [{s['add_minus_deepen']['loso_min_pp']}, {s['add_minus_deepen']['loso_max_pp']}]")
    print('复验参考信号:', summary['retest_signal'])


if __name__ == '__main__':
    main()
