"""B 独立复算 C 的 11-28 预算复验（只用归档混淆计数，不调用 C 的代码）。

覆盖：Sleep-EDF 开发复验（week3）、Ying 外部复验（week4）、BiGRU（week5）、两档预算（week6）、关灯窗口（window）。
每项做三件事：
  1. 点估计：B 自己的五类 Macro-F1（48 号 macro_f1_cm），与 C 的 summary.json 逐项比较（容差 1e-9）。
  2. 档案自洽：scores.csv 的逐拟合 Macro-F1 与混淆计数重算值比较。
  3. 区间：B 自己写的分层重抽样（名单、名单内抽块、评估者），rng 与 C 不同（seed 1），
     与 C 的区间比较（只应相差蒙特卡洛误差）；另报告"名单固定、只重抽评估者"的区间作敏感性。
输入：--root 指向含 experiment3/ 的仓库根目录。输出（--out）：summary.json、compare.csv。

用法：python scripts/53_b_c_budget_recompute.py --root . --out reports/w3/b_recompute_c
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('t48', HERE / '48_b_t1_budget_tables.py')
t48 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t48)

ARMS = ('deepen', 'other_night', 'add_people')
CON = {'other_minus_deepen': (1, 0), 'new_minus_other': (2, 1), 'new_minus_deepen': (2, 0)}
N_BOOT = 4000


def f1(cnt):
    """cnt [..., 25] -> Macro-F1（%）"""
    return t48.macro_f1_cm(cnt.reshape(*cnt.shape[:-1], 5, 5).astype(float)) * 100


def contrast(v, k):
    a, b = CON[k]
    return v[..., a] - v[..., b]


def boot_hier(cnt, cohorts, rng, fix_rosters=False, n=N_BOOT):
    """cnt [R, D, A, E, 25]；cohorts = 评估者分组的索引列表（组内有放回重抽）。返回 [n, A] 的跨名单、跨抽块平均 F1 差所需的 F1。"""
    R, D, A, E, _ = cnt.shape
    out = {k: np.empty(n) for k in CON}
    for i in range(n):
        w = np.zeros(E)
        for g in cohorts:
            np.add.at(w, rng.choice(g, len(g)), 1)
        F = f1(np.einsum('e,rdaec->rdac', w, cnt))       # [R, D, A]
        rs = np.arange(R) if fix_rosters else rng.integers(R, size=R)
        ds = np.tile(np.arange(D), (R, 1)) if fix_rosters else rng.integers(D, size=(R, D))
        sel = F[rs[:, None], ds]                          # [R, D, A]
        for k in CON:
            out[k][i] = contrast(sel, k).mean()
    return out


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    E3 = a.root / 'experiment3'
    rng = np.random.default_rng(1)
    rows, summ = [], {}

    def cmp(study, item, b, c):
        rows.append(dict(study=study, item=item, b=b, c=c, abs_diff=abs(b - c) if c is not None else None))

    # ---------- Sleep-EDF 开发复验与 BiGRU（同结构） ----------
    for study, d in (('sleepedf_dev', 'week3_budget_replication_20261128'), ('bigru', 'week5_budget_bigru_20261128')):
        cnt = np.load(E3 / d / 'results/confusion_counts.npz')['counts']   # [seed, R, D, A, E, 25]
        cs = json.loads((E3 / d / 'results/summary.json').read_text())
        sc = pd.read_csv(E3 / d / 'results/scores.csv')
        prim = cnt[0]
        # scores.csv 自洽（seed 0 全部；其他 seed 只在前 3 份名单）
        F_all = f1(cnt.sum(-2))                                             # [seed, R, D, A]
        dev = []
        for r in sc.itertuples():
            dev.append(abs(F_all[r.seed, r.roster - 1, r.draw - 1, ARMS.index(r.arm)] - r.macro_f1))
        res = dict(scores_rows=len(sc), scores_max_abs_diff=float(max(dev)))
        for cohort, sl in (('combined', slice(None)), ('dev8', slice(0, 8)), ('pretrain_eval8', slice(8, 16))):
            F = f1(prim[..., sl, :].sum(-2))                                 # [R, D, A]
            sub =prim if cohort == 'combined' else prim[..., sl, :]
            groups = [np.arange(8), np.arange(8, 16)] if cohort == 'combined' else [np.arange(8)]
            bh = boot_hier(sub, groups, rng)
            bf = boot_hier(sub, groups, rng, fix_rosters=True, n=1000)
            res[cohort] = {}
            for k in CON:
                m = float(contrast(F, k).mean())
                rv = contrast(F, k).mean(1)
                c = cs[cohort][k]
                res[cohort][k] = dict(mean_pp=m, ci_b=ci(bh[k]), ci_c=c['interval_95_pp'], ci_people_only=ci(bf[k]),
                                      positive_rosters=int((rv > 0).sum()))
                cmp(study, f'{cohort}.{k}.mean', m, c['mean_pp'])
                cmp(study, f'{cohort}.{k}.positive_rosters', int((rv > 0).sum()), c['positive_rosters'])
                for j, e in enumerate(('lo', 'hi')):
                    cmp(study, f'{cohort}.{k}.ci_{e}', res[cohort][k]['ci_b'][j], c['interval_95_pp'][j])
        summ[study] = res

    # ---------- Ying 外部复验 ----------
    d = E3 / 'week4_budget_external_ying/results'
    cnt = np.stack([np.load(d / f'fold{f}_confusions.npy') for f in range(1, 5)])   # [fold, R, D, A, E, 25]
    cs = json.loads((d / 'summary.json').read_text())
    proto = json.loads((d / 'protocol.json').read_text())
    F = f1(cnt.sum(-2))                                                           # [fold, R, D, A]
    names = {'other_minus_deepen': 'other_night_minus_deepen', 'new_minus_other': 'add_people_minus_other_night',
             'new_minus_deepen': 'add_people_minus_deepen'}
    boots = [boot_hier(cnt[f], [np.arange(cnt.shape[4])], rng) for f in range(4)]
    bootf = [boot_hier(cnt[f], [np.arange(cnt.shape[4])], rng, fix_rosters=True, n=1000) for f in range(4)]
    res = dict(primary_preset=proto['primary'],
               primary_in_all_protocol_snapshots=sorted({json.loads(p.read_text())['primary'] for p in d.glob('protocol*.json')}),
               epochs=int(cnt[:, 0, 0, 0].sum()))
    for k in CON:
        m = float(contrast(F, k).mean())
        fm = contrast(F, k).mean(axis=(1, 2))
        bh = np.mean([b[k] for b in boots], axis=0)
        bf = np.mean([b[k] for b in bootf], axis=0)
        c = cs['contrasts'][names[k]]
        res[k] = dict(mean_pp=m, ci_b=ci(bh), ci_c=c['95_interval_pp'], ci_people_only=ci(bf),
                      fold_means=fm.tolist(), positive_rosters=int((contrast(F, k).mean(2) > 0).sum()))
        cmp('ying', f'{k}.mean', m, c['mean_pp'])
        cmp('ying', f'{k}.positive_rosters', res[k]['positive_rosters'], c['positive_rosters'])
        for j in range(4):
            cmp('ying', f'{k}.fold{j + 1}', float(fm[j]), c['fold_means_pp'][j])
        for j, e in enumerate(('lo', 'hi')):
            cmp('ying', f'{k}.ci_{e}', res[k]['ci_b'][j], c['95_interval_pp'][j])
    p = res['new_minus_deepen']
    res['decision_b'] = ('external_support' if p['mean_pp'] >= 1 and p['ci_b'][0] > 0 and (np.array(p['fold_means']) > 0).sum() >= 3
                         else 'below_practical_target' if p['ci_b'][1] < 1 else 'inconclusive')
    res['decision_c'] = cs['decision']
    summ['ying'] = res

    # ---------- 两档预算 ----------
    d = E3 / 'week6_budget_extension_20261128/results'
    z = np.load(d / 'confusion_counts.npz')
    hi, rosters = z['counts'], z['rosters']                                       # [28, D, A, E, 25]
    lo = np.load(E3 / 'week3_budget_replication_20261128/results/confusion_counts.npz')['counts'][0][rosters - 1]
    cs = json.loads((d / 'summary.json').read_text())
    sc = pd.read_csv(d / 'results' / 'scores.csv') if (d / 'results/scores.csv').exists() else pd.read_csv(d / 'scores.csv')
    Fl, Fh = f1(lo.sum(-2)), f1(hi.sum(-2))
    ix = {r: i for i, r in enumerate(rosters)}
    dl = max(abs(Fl[ix[r.roster], r.draw - 1, ARMS.index(r.arm)] - r.macro_f1_low) for r in sc.itertuples())
    dh = max(abs(Fh[ix[r.roster], r.draw - 1, ARMS.index(r.arm)] - r.macro_f1_high) for r in sc.itertuples())
    res = dict(rosters=len(rosters), scores_max_abs_diff_low=float(dl), scores_max_abs_diff_high=float(dh))
    for k in CON:
        for tag, v in (('low', contrast(Fl, k).mean()), ('high', contrast(Fh, k).mean()),
                       ('high_minus_low', (contrast(Fh, k) - contrast(Fl, k)).mean())):
            c = cs['contrasts'][k][tag]
            cv = c['mean_pp'] if isinstance(c, dict) else c
            res[f'{k}.{tag}'] = float(v)
            cmp('budget_extension', f'{k}.{tag}.mean', float(v), cv)
    summ['budget_extension'] = res

    # ---------- 关灯窗口 ----------
    d = E3 / 'window_budget_c_20261128/results'
    z = np.load(d / 'paired_confusions.npz')
    cnt, exact = z['counts'], z['exact_600']                                      # [R, D, W, A, E, 25], [R, D]
    cs = json.loads((d / 'summary.json').read_text())
    F = f1(cnt.sum(-2))                                                           # [R, D, W, A]
    nd = F[..., 2] - F[..., 0]                                                    # 新人 − 原夜 [R, D, W]
    res = {}
    for tag, mask in (('exact_600', exact), ('fixed_600_attempts', np.ones_like(exact))):
        keep = mask.any(1)
        per = [nd[r][mask[r]].mean(0) for r in np.flatnonzero(keep)]              # 每名单 [W]
        per = np.array(per)
        vals = dict(old=per[:, 0].mean(), lights_off=per[:, 1].mean(), did=(per[:, 1] - per[:, 0]).mean())
        c = cs[tag]
        res[tag] = dict(rosters=int(keep.sum()), draw_pairs=int(mask.sum()), **{k: float(v) for k, v in vals.items()})
        cmp('window', f'{tag}.rosters', int(keep.sum()), c['rosters'])
        cmp('window', f'{tag}.draw_pairs', int(mask.sum()), c['draw_pairs'])
        for k, ck in (('old', 'old_newcomer_minus_original_night'), ('lights_off', 'lights_off_newcomer_minus_original_night'),
                      ('did', 'paired_difference_of_differences')):
            cmp('window', f'{tag}.{k}.mean', float(vals[k]), c[ck]['mean_pp'])
    summ['window'] = res

    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / 'compare.csv', index=False, lineterminator='\n')
    exact_items = df[~df['item'].str.contains('ci_')]
    summ['max_abs_diff_point_estimates'] = float(exact_items['abs_diff'].max())
    summ['max_abs_diff_ci_endpoints'] = float(df[df['item'].str.contains('ci_')]['abs_diff'].max())
    summ['bootstrap_b'] = f'{N_BOOT} 次，分层重抽（名单、名单内抽块、评估者），rng seed 1；people_only 为名单与抽块固定，1000 次'
    (a.out / 'summary.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('point estimates max |diff|:', summ['max_abs_diff_point_estimates'])
    print('CI endpoints max |diff|:', summ['max_abs_diff_ci_endpoints'])


if __name__ == '__main__':
    main()
