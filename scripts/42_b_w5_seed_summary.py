"""Step W5-B1: 跨模态三组补 head seed 3、4 后的汇总（固定三套 backbone，下游 EOG-only linear probe）。

输入（均由已有脚本生成，本脚本只读不训练）：
  --metrics    39 号复评输出的 metrics.csv（30 个 run：3 组 × 2 预算 × seed 0–4）
  --paired     39 号复评输出的 paired.csv（seed 0–4 合并，原始与 α 调整后）
  --per-seed   35 号输出的 paired.csv（含逐 seed 差值列）
  --ref-paired W4 的 seed 0–2 配对表（reports/week4/xmodal_reeval/b_formal/paired.csv）
输出（--out）：
  seed_spread.csv    每组 × 预算：各指标跨 5 个 head seed 的均值、SD、最小、最大，另列 seed 0–2 均值
  seed_direction.csv 每个配对 × 预算 × 指标：逐 seed 差值、正负计数
  pooled_compare.csv 每个配对 × 预算 × 指标：seed 0–2 与 seed 0–4 合并后的点估计、95% 区间、区间是否含 0
"""
import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

ROLE = {'xm_sync': 'sync', 'xm_shift': 'shift', 'xm_none': 'none'}
METRICS = [('macro_f1', 'Macro-F1'), ('n1_ap', 'N1 AP'), ('n1_f1', '原始 N1 F1'),
           ('n1_f1_adjusted', '调整后 N1 F1'), ('macro_f1_adjusted', '调整后 Macro-F1'), ('n1_alpha', 'α')]
PER_SEED = [('per_seed_delta_macro_f1', 'Macro-F1'), ('per_seed_delta_n1_ap', 'N1 AP'),
            ('per_seed_delta_n1_f1', '原始 N1 F1'), ('per_seed_delta_n1_f1_adjusted', '调整后 N1 F1'),
            ('per_seed_delta_macro_f1_adjusted', '调整后 Macro-F1')]
CONTRASTS = ['sync-none', 'shift-none', 'sync-shift']
BUDGETS = ['label10', 'label25']


def read(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write(p, rows):
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--paired', required=True)
    ap.add_argument('--per-seed', required=True)
    ap.add_argument('--ref-paired', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 跨 seed 离散度
    met = read(a.metrics)
    alpha_key = 'alpha' if 'alpha' in met[0] else 'n1_alpha'
    adj_mf1_key = 'macro_f1_adjusted' if 'macro_f1_adjusted' in met[0] else 'macro_f1_adjust'
    keymap = {'n1_alpha': alpha_key, 'macro_f1_adjusted': adj_mf1_key}
    spread = []
    for arm in ROLE:
        for b in BUDGETS:
            rs = sorted([r for r in met if r['arm'] == arm and r['budget'] == b], key=lambda r: int(r['seed']))
            seeds = [int(r['seed']) for r in rs]
            assert seeds == [0, 1, 2, 3, 4], (arm, b, seeds)
            for k, name in METRICS:
                v = [float(r[keymap.get(k, k)]) for r in rs]
                spread.append(dict(group=ROLE[arm], budget=b, metric=k, label=name,
                                   mean_0_4=round(mean(v), 6), sd_0_4=round(stdev(v), 6),
                                   min=round(min(v), 6), max=round(max(v), 6),
                                   mean_0_2=round(mean(v[:3]), 6),
                                   values=' '.join(f'{x:.4f}' for x in v)))
            spread.append(dict(group=ROLE[arm], budget=b, metric='selected_epoch', label='E*',
                               mean_0_4=mean(int(r['selected_epoch']) for r in rs) if 'selected_epoch' in rs[0] else '',
                               sd_0_4='', min='', max='', mean_0_2='',
                               values=' '.join(r.get('selected_epoch', '') for r in rs)))
    write(out / 'seed_spread.csv', spread)

    # 2. 逐 seed 方向
    ps = read(a.per_seed)
    direction = []
    for r in ps:
        assert r['seeds'].split() == ['0', '1', '2', '3', '4'], r['seeds']
        for k, name in PER_SEED:
            v = [float(x) for x in r[k].split()]
            direction.append(dict(contrast=r['contrast'], budget=r['budget'], metric=name,
                                  per_seed=' '.join(f'{x:+.4f}' for x in v),
                                  n_pos=sum(x > 0 for x in v), n_neg=sum(x < 0 for x in v),
                                  n_pos_0_2=sum(x > 0 for x in v[:3]), n_neg_0_2=sum(x < 0 for x in v[:3])))
    write(out / 'seed_direction.csv', direction)

    # 3. 合并区间：seed 0–2 与 0–4
    new = {(r['contrast'], r['scores'], r['budget']): r for r in read(a.paired)}
    old = {(r['contrast'], r['scores'], r['budget']): r for r in read(a.ref_paired)}
    comp = []
    for (c, s, b), rn in new.items():
        ro = old[(c, s, b)]
        assert ro['seeds'].split() == ['0', '1', '2'] and rn['seeds'].split() == ['0', '1', '2', '3', '4']
        for k in ['delta_macro_f1', 'delta_n1_f1', 'delta_n1_ap']:
            row = dict(contrast=c, scores=s, budget=b, metric=k)
            for tag, r in (('0_2', ro), ('0_4', rn)):
                lo, hi = float(r[k + '_lo']), float(r[k + '_hi'])
                row[f'{tag}_delta'] = float(r[k])
                row[f'{tag}_lo'] = lo
                row[f'{tag}_hi'] = hi
                row[f'{tag}_excl0'] = lo > 0 or hi < 0
            comp.append(row)
    write(out / 'pooled_compare.csv', comp)

    n_new = sum(r['0_4_excl0'] for r in comp)
    n_old = sum(r['0_2_excl0'] for r in comp)
    flips = [r for r in comp if r['0_2_excl0'] != r['0_4_excl0']]
    print(f'区间不含 0：seed 0–2 {n_old} 格，seed 0–4 {n_new} 格（共 {len(comp)} 格）')
    for r in flips:
        print(f"  变化 {r['contrast']} {r['scores']} {r['budget']} {r['metric']}: "
              f"{100*r['0_2_delta']:+.2f} [{100*r['0_2_lo']:+.2f},{100*r['0_2_hi']:+.2f}] -> "
              f"{100*r['0_4_delta']:+.2f} [{100*r['0_4_lo']:+.2f},{100*r['0_4_hi']:+.2f}]")
    print(f'已写入 {out}')


if __name__ == '__main__':
    main()
