"""新一轮第 2 周 · A 复算(第 9 天职责):串起开发预测,复算各阶段 F1,重点 N3 贡献与 N1 AP。

输入:
  - 第一周三条件(base / deepen / add_people)的 pool36 与 dev8 逐段预测:
    C 01-05 与 D 06-10 的 Week1 结果包(本地 cache/w1_budget/)
  - 第二周九份名单的原人另一夜(other_night)预测:
    A(01/07/09,runs/)、B(02/08)、C(03/10)、D(04/06) 的 Week2 交付(本地 cache/w2_budget/)
输出(--out):
  per_class.csv        每名单 × 组 × 条件 × 五类的 F1(pp 粒度的问题直接看原始值)
  summary.csv          每名单 × 组 × 条件:Macro-F1、N1 AP、N1 F1、N3 F1
  contrasts.csv        主比较 other_night−deepen、add_people−other_night 的逐类贡献与合计
  recompute_run.json   口径与来源登记

用法:
  python scripts/50_a_w2_recompute.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import BASE
import w2_common as w

ev = w.ev
CLASSES = ('W', 'N1', 'N2', 'N3', 'REM')
CONDS = ('base', 'deepen', 'add_people', 'other_night')
WEEK1_C = BASE / 'cache/w1_budget/c01_05/x'
WEEK1_D = BASE / 'cache/w1_budget/d06_10/x/d_budget_week1_results_v01'
W2 = {
    '01': BASE / 'runs/01/other_night',
    '07': BASE / 'runs/07/other_night',
    '09': BASE / 'runs/09/other_night',
    '02': BASE / 'cache/w2_budget/b/x_b_budget_week2_other_night_v01/b_budget_week2_other_night_v01/02/other_night',
    '08': BASE / 'cache/w2_budget/b/x_b_budget_week2_other_night_v01/b_budget_week2_other_night_v01/08/other_night',
    '03': BASE / 'cache/w2_budget/c/x_C_week2_matching_and_results_20261127/results/03/other_night',
    '10': BASE / 'cache/w2_budget/c/x_C_week2_matching_and_results_20261127/results/10/other_night',
    '04': BASE / 'cache/w2_budget/d/x_d_budget_week2_other_night_v01/d_budget_week2_other_night_v01/04/other_night',
    '06': BASE / 'cache/w2_budget/d/x_d_budget_week2_other_night_v01/d_budget_week2_other_night_v01/06/other_night',
}


def week1_dir(roster, cond):
    if roster <= '05':
        return WEEK1_C / roster / 'paired' / cond
    return WEEK1_D / roster / 'paired' / cond


def metrics_of(df):
    y, p = df['y_true'].to_numpy(), df['y_pred'].to_numpy()
    m = ev.compute_metrics(y, p)
    y1 = (y == 1)
    return dict(macro=m['macro_f1'],
                n1_ap=float(average_precision_score(y1, df['p_N1'].to_numpy())),
                n1_f1=m['n1_f1'],
                per_class={c: m['per_class_f1'][c] for c in CLASSES})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(BASE / 'reports' / 'w2_budget' / 'a_recompute'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, pc_rows = [], []
    for roster in [f'{i:02d}' for i in range(1, 11)]:
        for group in ('pool36', 'dev8'):
            for cond in CONDS:
                if cond == 'other_night' and roster == '05':
                    continue  # 05 无另一夜(SC452 单夜)
                d = W2[roster] if cond == 'other_night' else week1_dir(roster, cond)
                f = d / f'predictions_{group}.csv'
                if not f.exists():
                    raise SystemExit(f'缺预测 {f}')
                df = pd.read_csv(f)
                m = metrics_of(df)
                rows.append(dict(roster=roster, group=group, cond=cond, **{k: v for k, v in m.items() if k != 'per_class'}))
                pc_rows.append(dict(roster=roster, group=group, cond=cond, **m['per_class']))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / 'summary.csv', index=False, float_format='%.6g')
    pc = pd.DataFrame(pc_rows)
    pc.to_csv(out / 'per_class.csv', index=False, float_format='%.6g')

    # 主比较的逐类贡献:other_night − deepen 与 add_people − other_night
    piv = pc.pivot_table(index=['roster', 'group', 'cond'], values=list(CLASSES)).reset_index()
    contrasts = []
    for roster in piv.roster.unique():
        for group in ('pool36', 'dev8'):
            sub = piv[(piv.roster == roster) & (piv.group == group)].set_index('cond')
            if 'other_night' not in sub.index:
                continue
            for name, a, b in (('other_night-deepen', 'other_night', 'deepen'),
                               ('add_people-other_night', 'add_people', 'other_night')):
                row = dict(roster=roster, group=group, contrast=name)
                for c in CLASSES:
                    row[f'd_{c}'] = (sub.loc[a, c] - sub.loc[b, c]) * 100
                row['d_macro'] = sum(row[f'd_{c}'] for c in CLASSES) / 5
                sa = summ[(summ.roster == roster) & (summ.group == group)].set_index('cond')
                row['d_n1_ap'] = (sa.loc[a, 'n1_ap'] - sa.loc[b, 'n1_ap']) * 100
                contrasts.append(row)
    cdf = pd.DataFrame(contrasts)
    cdf.to_csv(out / 'contrasts.csv', index=False, float_format='%.6g')

    (out / 'recompute_run.json').write_text(json.dumps(dict(
        script='scripts/50_a_w2_recompute.py', sources='week1 C/D 结果包 + week2 A/B/C/D 交付',
        conditions=CONDS, groups=['pool36', 'dev8'],
        note='Macro 为五类等权(官方 evaluate v1);N1 AP 用 p_N1;05 无另一夜'), indent=2, ensure_ascii=False) + '\n')

    print(cdf.round(3).to_string(index=False))
    print(f'已写入 {out}')


if __name__ == '__main__':
    main()
