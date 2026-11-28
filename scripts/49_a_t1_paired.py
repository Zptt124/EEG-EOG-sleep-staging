"""新一轮第 1 周 · 时序线配对比较(A 第 4 天汇总)。

按任务书:每份名单、每个评估组(dev8 主评 / eval9 / merged17),对 A1−A0 与 A1−A2
逐 seed 算配对差,再取 3 seed 均值与范围;同时给 seed 合并后按受试者 bootstrap 1000
的 95% 区间(复用 29 号 paired,口径与 W5 相同)。指标:N1 AP(主判据)、Macro-F1、
原始 N1 F1、短 N1(连续 1–2 段)召回与 W/N2 错报。

用法:
  python scripts/49_a_t1_paired.py --root runs --out reports/w1_temporal/eval
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import BASE, RUNS
import w2_common as w

ev = w.ev


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


E29 = _load('eval29', '29_b_w3_evaluate.py')
T = _load('t1c', 't1_temporal_common.py')

ROSTERS = BASE / 'manifests' / 'temporal_rosters_t1_v01.csv'
SEGMENTS = BASE / 'manifests' / 'contiguous_segments_t1_v01.csv'
PROB = w.PROB_COLS


def load_preds(path):
    return pd.read_csv(path)


def n1_ap(df):
    y1 = (df['y_true'] == 1).to_numpy()
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y1, df['p_N1'].to_numpy()))


def short_n1_rows(df, segments):
    """短 N1 段的行下标(连续 1–2 段,不跨缺口)。"""
    seg = T.load_segments(SEGMENTS)
    rows = df.to_dict('records')
    by_rec = {}
    for i, r in enumerate(rows):
        by_rec.setdefault(r['record_id'], []).append((int(r['epoch_index']), i))
    short = set()
    for rec, entries in by_rec.items():
        for a, b in seg.get(rec, []):
            run = sorted(e for e, i in entries if a <= e <= b)
            n1s = [i for e, i in sorted(entries) if a <= e <= b and rows[i]['y_true'] == 1]
            k = 0
            while k < len(n1s):
                m = k
                while m + 1 < len(n1s) and n1s[m + 1] == n1s[m] + 1:
                    m += 1
                if m - k + 1 <= 2:
                    short.update(n1s[k:m + 1])
                k = m + 1
    return sorted(short)


def group_metrics(df, segments):
    out = dict(n1_ap=n1_ap(df),
               macro_f1=ev.compute_metrics(df['y_true'].to_numpy(), df['y_pred'].to_numpy())['macro_f1'],
               n1_f1=ev.compute_metrics(df['y_true'].to_numpy(), df['y_pred'].to_numpy())['n1_f1'])
    short = short_n1_rows(df, segments)
    yp = df['y_pred'].to_numpy()
    out['short_n1'] = dict(n_short=len(short),
                           recall=float((yp[short] == 1).mean()) if short else None,
                           to_W=float((yp[short] == 0).mean()) if short else None,
                           to_N2=float((yp[short] == 2).mean()) if short else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--root', default=str(RUNS))
    ap.add_argument('--out', default=str(BASE / 'reports' / 'w1_temporal' / 'eval'))
    ap.add_argument('--segments', default=str(SEGMENTS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    preds = {}
    for d in sorted(Path(args.root).glob('T1_*_r*_seed*_v01')):
        m = re.fullmatch(r'T1_(\w)_(A\d)_r(\d)_seed(\d)_v\d+', d.name)
        if m and (d / 'predictions_dev8.csv').exists():
            preds[(m[2], m[3], m[4])] = dict(dev8=load_preds(d / 'predictions_dev8.csv'),
                                             eval9=load_preds(d / 'predictions_eval9.csv'))
    assert len(preds) == 18, f'应有 18 次拟合,实际 {len(preds)}'

    rows_out, seed_rows = [], []
    for roster in ('1', '2'):
        for group in ('dev8', 'eval9', 'merged17'):
            for contrast in (('A1', 'A0'), ('A1', 'A2')):
                c, r = contrast
                per_seed = []
                for seed in ('0', '1', '2'):
                    a = preds[(c, roster, seed)][group if group != 'merged17' else 'dev8']
                    b = preds[(r, roster, seed)][group if group != 'merged17' else 'dev8']
                    if group == 'merged17':
                        ae = preds[(c, roster, seed)]['eval9']
                        be = preds[(r, roster, seed)]['eval9']
                        a = pd.concat([a, ae], ignore_index=True)
                        b = pd.concat([b, be], ignore_index=True)
                    if not (a['sample_id'].equals(b['sample_id']) and a['y_true'].equals(b['y_true'])):
                        raise SystemExit(f'{roster} {group} {c}-{r} seed {seed}: 样本键或真值不一致')
                    per_seed.append(dict(seed=int(seed),
                                         d_n1_ap=(n1_ap(a) - n1_ap(b)) * 100,
                                         d_macro=(ev.compute_metrics(a['y_true'], a['y_pred'])['macro_f1']
                                                  - ev.compute_metrics(b['y_true'], b['y_pred'])['macro_f1']) * 100,
                                         d_n1_f1=(ev.compute_metrics(a['y_true'], a['y_pred'])['n1_f1']
                                                  - ev.compute_metrics(b['y_true'], b['y_pred'])['n1_f1']) * 100))
                d = pd.DataFrame(per_seed)
                seed_rows.append(dict(roster=roster, group=group, contrast=f'{c}-{r}',
                                      d_n1_ap=' '.join(f'{v:+.2f}' for v in d.d_n1_ap),
                                      d_macro=' '.join(f'{v:+.2f}' for v in d.d_macro),
                                      d_n1_f1=' '.join(f'{v:+.2f}' for v in d.d_n1_f1)))
                rows_out.append(dict(roster=roster, group=group, contrast=f'{c}-{r}',
                                     mean_d_n1_ap=d.d_n1_ap.mean(), range_d_n1_ap=(d.d_n1_ap.min(), d.d_n1_ap.max()),
                                     mean_d_macro=d.d_macro.mean(), mean_d_n1_f1=d.d_n1_f1.mean()))
    per_seed_df = pd.DataFrame(seed_rows)
    per_seed_df.to_csv(out / 'per_seed.csv', index=False, float_format='%.6g')
    summ = pd.DataFrame(rows_out)
    summ.to_csv(out / 'summary.csv', index=False, float_format='%.6g')

    # seed 合并的按受试者 bootstrap(dev8 主判据,复用 29 号 paired)
    reg_rows = []
    for (c, r, s), g in preds.items():
        reg_rows.append(dict(run_id=f'T1_X_{c}_r{r}_seed{s}', group='A', arm=c, budget=f'r{r}',
                             seed=int(s), predictions=''))
    reg = pd.DataFrame(reg_rows, columns=E29.COLS)
    preds_e29 = {f'T1_X_{c}_r{r}_seed{s}': g['dev8'] for (c, r, s), g in preds.items()}
    prow = []
    for contrast in (('A1', 'A0'), ('A1', 'A2')):
        p, _ = E29.paired(reg, preds_e29, *contrast)
        for row in p:
            row['contrast'] = f'{contrast[0]}-{contrast[1]}'
        prow += p
    pd.DataFrame(prow).to_csv(out / 'paired_dev8.csv', index=False, float_format='%.6g')

    # 短 N1 统计(每条件 × 名单,dev8 与 merged17)
    sn = []
    for (c, r, s), g in preds.items():
        for group in ('dev8', 'merged17'):
            df = g[group if group != 'merged17' else 'dev8']
            if group == 'merged17':
                df = pd.concat([df, g['eval9']], ignore_index=True)
            m = group_metrics(df, args.segments)
            sn.append(dict(cond=c, roster=r, seed=s, group=group, **m['short_n1']))
    pd.DataFrame(sn).to_csv(out / 'short_n1.csv', index=False, float_format='%.6g')

    print(summ.round(3).to_string(index=False))
    print(per_seed_df.to_string(index=False))
    print(f'已写入 {out}')


if __name__ == '__main__':
    main()
