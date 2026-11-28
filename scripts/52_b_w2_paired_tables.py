"""新一轮第 2 周 · B 两张配对表（输入 = 51 号统一评分的逐段预测）。

表一｜复验第一周结论：10 份名单（01–10），新人新夜（add_people）− 原人原夜（deepen）。
表二｜拆分人和夜：9 份名单（01–04、06–10），
      换夜收益 = 原人另一夜（other_night）− 原人原夜（deepen）；
      新人较另一夜 = 新人新夜（add_people）− 原人另一夜（other_night）。
指标：原始五类 Macro-F1（混淆矩阵口径，无预测的类 F1 记 0），差值为百分点；名单内配对差，跨名单等权平均。
不确定性：每名评分受试者每次抽一个 Poisson(1) 权重，所有名单共用（同一受试者跨名单共享 bootstrap 权重），
          加权混淆矩阵重算，1000 次，rng seed 0；报告逐名单与跨名单平均差的 95% 区间。
方向与稳健：同号名单数；逐人剔除（从所有名单去掉一人）后平均差的范围；最大、最小配对差。
敏感性：另报告"名单与受试者一起重抽"的区间（ci95_rosters_people；名单有放回重抽，rng seed 1），
      口径与 C 的扩大复验相同；主区间只重抽受试者，条件是这几份名单。
判读参考（任务书建议，第 6 天前冻结）：实际意义 1 pp；等效带 ±1 pp，整个 95% 区间落入带内才判"等效"。
核对：每个头的样本键、真值与其他头逐行相同；y_pred = 概率 argmax；评分人数 = 8（dev8 或预留 8 人）。
输出（--out）：table1.csv、table2.csv、loso.csv、checks.csv、summary.json

用法：
  python scripts/52_b_w2_paired_tables.py --scored runs/w2_heldout8 --cohort heldout8 --out reports/w2/b_tables_heldout8
  （开发核查：--scored runs/w2_replay --cohort dev8）
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

KEY = ['sample_id', 'subject_id', 'record_id', 'epoch_index', 'y_true']
PROBS = ['p_W', 'p_N1', 'p_N2', 'p_N3', 'p_REM']
T1_ROSTERS = [f'{i:02d}' for i in range(1, 11)]
T2_ROSTERS = [r for r in T1_ROSTERS if r != '05']
TABLES = {
    'table1': (T1_ROSTERS, {'add_minus_deepen': ('add_people', 'deepen')}),
    'table2': (T2_ROSTERS, {'other_minus_deepen': ('other_night', 'deepen'),
                            'add_minus_other': ('add_people', 'other_night')}),
}
BAND, MEANINGFUL, N_BOOT = 1.0, 1.0, 1000


def verdict(mean, lo, hi):
    if -BAND < lo and hi < BAND:
        return '等效（区间落入 ±1 pp）'
    if lo > 0 and mean >= MEANINGFUL:
        return '正向且达到 1 pp'
    if hi < 0 and mean <= -MEANINGFUL:
        return '负向且达到 1 pp'
    if lo > 0 or hi < 0:
        return '方向确定但未达 1 pp'
    return '不确定（区间跨 0 且超出等效带）'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scored', required=True, type=Path)
    ap.add_argument('--cohort', required=True, choices=['dev8', 'heldout8'])
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    preds, checks = {}, []
    ref = None
    for d in sorted(args.scored.glob('*/*/predictions_' + args.cohort + '.csv')):
        roster, arm = d.parent.parent.name, d.parent.name
        df = pd.read_csv(d)
        if ref is None:
            ref = df[KEY]
        ok_keys = df[KEY].astype(str).equals(ref.astype(str))
        ok_arg = bool((df['y_pred'].to_numpy() == df[PROBS].to_numpy().argmax(1)).all())
        checks.append(dict(roster=roster, arm=arm, keys_truth_identical=ok_keys, argmax_ok=ok_arg,
                           n_subjects=df['subject_id'].nunique(), n_epochs=len(df)))
        preds[roster, arm] = df
    subjects = sorted(ref['subject_id'].unique())
    assert len(subjects) == 8, f'评分人数 {len(subjects)}'
    cms = {k: t48.subject_cms(v[['y_true', 'y_pred', 'subject_id']].to_dict('records'), subjects)
           for k, v in preds.items()}
    rng = np.random.default_rng(0)
    W = rng.poisson(1.0, size=(N_BOOT, len(subjects))).astype(float)
    rng_r = np.random.default_rng(1)

    def f1(cm, w=None):
        return t48.macro_f1_cm(cm.sum(0) if w is None else np.einsum('bs,sij->bij', w, cm)) * 100

    out_tables, loso_rows, summ = {}, [], {}
    args.out.mkdir(parents=True, exist_ok=True)
    for tname, (rosters, contrasts) in TABLES.items():
        arms = sorted({a for pair in contrasts.values() for a in pair})
        missing = [(r, a) for r in rosters for a in arms if (r, a) not in cms]
        if missing:
            summ[tname] = dict(status='incomplete', missing=[f'{r}/{a}' for r, a in missing])
            continue
        rows = []
        for r in rosters:
            row = dict(roster=r, **{f'macro_f1_{a}': round(f1(cms[r, a]), 4) for a in arms})
            for cname, (a, b) in contrasts.items():
                d = f1(cms[r, a]) - f1(cms[r, b])
                bd = f1(cms[r, a], W) - f1(cms[r, b], W)
                row.update({f'{cname}_pp': round(d, 4), f'{cname}_lo': round(np.percentile(bd, 2.5), 4),
                            f'{cname}_hi': round(np.percentile(bd, 97.5), 4)})
            rows.append(row)
        mean = dict(roster='mean', **{f'macro_f1_{a}': round(np.mean([f1(cms[r, a]) for r in rosters]), 4) for a in arms})
        summ[tname] = dict(status='complete', rosters=rosters, contrasts={})
        for cname, (a, b) in contrasts.items():
            diffs = np.array([f1(cms[r, a]) - f1(cms[r, b]) for r in rosters])
            bd = np.array([f1(cms[r, a], W) - f1(cms[r, b], W) for r in rosters])      # [名单, N_BOOT]
            bmean = bd.mean(0)
            lo, hi = np.percentile(bmean, [2.5, 97.5])
            ri = rng_r.integers(len(rosters), size=(N_BOOT, len(rosters)))
            both = bd[ri, np.arange(N_BOOT)[:, None]].mean(1)
            mean.update({f'{cname}_pp': round(diffs.mean(), 4), f'{cname}_lo': round(lo, 4), f'{cname}_hi': round(hi, 4)})
            loso = []
            for j, s in enumerate(subjects):
                keep = [k for k in range(len(subjects)) if k != j]
                m = np.mean([f1(cms[r, a][keep]) - f1(cms[r, b][keep]) for r in rosters])
                loso.append(m)
                loso_rows.append(dict(table=tname, contrast=cname, dropped=s, mean_pp=round(m, 4)))
            summ[tname]['contrasts'][cname] = dict(
                mean_pp=round(diffs.mean(), 4), ci95=[round(lo, 4), round(hi, 4)],
                ci95_rosters_people=[round(x, 4) for x in np.percentile(both, [2.5, 97.5])],
                positive=int((diffs > 0).sum()), negative=int((diffs < 0).sum()), n_rosters=len(rosters),
                max=dict(roster=rosters[int(diffs.argmax())], pp=round(diffs.max(), 4)),
                min=dict(roster=rosters[int(diffs.argmin())], pp=round(diffs.min(), 4)),
                loso_range=[round(min(loso), 4), round(max(loso), 4)],
                loso_sign_stable=bool(all(np.sign(x) == np.sign(diffs.mean()) for x in loso)),
                verdict=verdict(diffs.mean(), lo, hi))
        rows.append(mean)
        out_tables[tname] = pd.DataFrame(rows)
        out_tables[tname].to_csv(args.out / f'{tname}.csv', index=False, lineterminator='\n')
    pd.DataFrame(loso_rows).to_csv(args.out / 'loso.csv', index=False, lineterminator='\n')
    pd.DataFrame(checks).to_csv(args.out / 'checks.csv', index=False, lineterminator='\n')
    manifest = args.scored / 'scoring_manifest.json'
    (args.out / 'summary.json').write_text(json.dumps(dict(
        script='scripts/52_b_w2_paired_tables.py', cohort=args.cohort, subjects=subjects,
        scoring_manifest_sha256=(t48.sha256(manifest) if manifest.exists() else None),
        all_checks_ok=bool(all(c['keys_truth_identical'] and c['argmax_ok'] and c['n_subjects'] == 8 for c in checks)),
        bootstrap=f'{N_BOOT} 次 Poisson(1) 受试者权重，跨名单共用，rng seed 0',
        band_pp=BAND, meaningful_pp=MEANINGFUL, tables=summ), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    for t, s in summ.items():
        print(t, json.dumps(s, ensure_ascii=False))


if __name__ == '__main__':
    main()
