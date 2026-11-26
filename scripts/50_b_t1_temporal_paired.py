"""新一轮第 1 周 · 时序线配对复评：A0 / A1 / A2 两两比较（A1−A0 主比较，A1−A2 置换对照，A2−A0 附带）。

输入：--runs 下 46 号的 run 目录 T1_<组>_<条件>_r<名单>_seed<s>_<tag>/（predictions_dev8.csv、predictions_eval9.csv、config.json）。
核对：同名单同评分集的所有 run 样本键与真值逐行相同；y_pred = argmax；更新步数 = 名单 1 2,820 / 名单 2 2,490；
      同名单内 norm_stats 与特征指纹相同；A1/A2 的 --init 来自各自唯一一份预训练权重。
指标：Macro-F1、N1 AP、短 N1 召回（真值连续 1–2 段的 N1）、W/N2 判成 N1 的比例。
配对：每份名单、每个评分集分别做；3 个 seed 的指标先取平均再求差；
      不确定性 = 按受试者有放回重抽 1000 次（rng seed 0），dev 8 人为主判据，9 人、17 人只看方向。
      17 人由 dev8 与 eval9 的逐段预测合并后重算，不是两个指标平均。两份名单不合并。
输出（--out）：checks.csv、metrics_by_run.csv、paired.csv、summary.json

用法：
  python scripts/50_b_t1_temporal_paired.py --runs runs/t1_final [--runs <A 的目录>] --out reports/t1/b_temporal_paired
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_temporal_common as T  # noqa: E402
import w2_common as w  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
PROB = w.PROB_COLS
KEYS = ['sample_id', 'subject_id', 'record_id', 'epoch_index', 'y_true']
UPDATES = {'1': 2820, '2': 2490}
CONTRASTS = [('A1', 'A0'), ('A1', 'A2'), ('A2', 'A0')]
RUN_RE = re.compile(r'T1_(?P<group>[^_]+)_(?P<cond>A[012])_r(?P<roster>[12])_seed(?P<seed>\d+)_(?P<tag>.+)$')


def metrics(y, yp, p1, short, wn2):
    return dict(macro_f1=f1_score(y, yp, labels=range(5), average='macro', zero_division=0) * 100,
                n1_ap=average_precision_score(y == 1, p1) * 100,
                short_n1_recall=(yp[short] == 1).mean() * 100,
                wn2_to_n1_rate=(yp[wn2] == 1).mean() * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', action='append', required=True)
    ap.add_argument('--segments', default=str(BASE / 'manifests' / 'contiguous_segments_t1_v01.csv'))
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    runs = {}
    for root in args.runs:
        for d in sorted(Path(root).iterdir()):
            m = RUN_RE.match(d.name)
            if m and (d / 'predictions_dev8.csv').exists():
                runs[(m['cond'], m['roster'], int(m['seed']))] = d
    segs = T.load_segments(args.segments)
    checks, rows_m, paired, summ = [], [], [], {}
    rng = np.random.default_rng(0)
    for roster in ('1', '2'):
        conds = sorted({c for c, r, _ in runs if r == roster})
        seeds = {c: sorted(s for cc, r, s in runs if cc == c and r == roster) for c in conds}
        if not conds:
            continue
        cfg = {k: json.loads((d / 'config.json').read_text(encoding='utf-8')) for k, d in runs.items() if k[1] == roster}
        norm = {v['norm'].get('sha256') for v in cfg.values()}
        feat = {json.dumps(v['features']['files'], sort_keys=True) for v in cfg.values()}
        inits = {}
        for (c, _, _), v in cfg.items():
            if v['init']:
                inits.setdefault(c, set()).add(v['init']['sha256'])
        checks.append(dict(roster=roster, check='norm_stats_identical', ok=len(norm) == 1 and None not in norm))
        checks.append(dict(roster=roster, check='features_identical', ok=len(feat) == 1))
        checks.append(dict(roster=roster, check='one_pretrained_weight_per_cond',
                           ok=all(len(s) == 1 for s in inits.values())))
        checks.append(dict(roster=roster, check='updates', ok=all(v['updates'] == UPDATES[roster] for v in cfg.values())))

        preds = {}
        for k, d in runs.items():
            if k[1] != roster:
                continue
            preds[k] = {s: pd.read_csv(d / f'predictions_{s}.csv') for s in ('dev8', 'eval9')}
        ref = next(iter(preds.values()))
        for k, pr in preds.items():
            for s in ('dev8', 'eval9'):
                df = pr[s]
                ok_keys = df[KEYS].equals(ref[s][KEYS])
                ok_arg = bool((df['y_pred'].to_numpy() == df[PROB].to_numpy().argmax(1)).all())
                checks.append(dict(roster=roster, check=f'{k[0]}_seed{k[2]}_{s}_keys_argmax', ok=ok_keys and ok_arg))
        for s in ('dev8', 'eval9', 'eval17'):
            base = pd.concat([ref['dev8'], ref['eval9']], ignore_index=True) if s == 'eval17' else ref[s]
            y = base['y_true'].to_numpy(int)
            short = T.short_n1_runs(base[['record_id', 'epoch_index', 'y_true']].astype(str).to_dict('records'), segs)
            wn2 = np.isin(y, [0, 2])
            subj = base['subject_id'].to_numpy()
            groups = [np.flatnonzero(subj == u) for u in sorted(set(subj))]
            P = {}
            for k, pr in preds.items():
                df = pd.concat([pr['dev8'], pr['eval9']], ignore_index=True) if s == 'eval17' else pr[s]
                P[k] = (df['y_pred'].to_numpy(int), df['p_N1'].to_numpy())
                rows_m.append(dict(roster=roster, eval_set=s, cond=k[0], seed=k[2],
                                   **metrics(y, *P[k], short, wn2)))
            boot_idx = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
                        for _ in range(args.n_boot)]
            for a, b in CONTRASTS:
                if a not in conds or b not in conds:
                    continue

                def cond_mean(c, ii):
                    ms = [metrics(y[ii], P[(c, roster, sd)][0][ii], P[(c, roster, sd)][1][ii], short[ii], wn2[ii])
                          for sd in seeds[c]]
                    return {m: np.mean([x[m] for x in ms]) for m in ms[0]}
                full = np.arange(len(y))
                ma, mb = cond_mean(a, full), cond_mean(b, full)
                bs = [(lambda x, z: {m: x[m] - z[m] for m in x})(cond_mean(a, ii), cond_mean(b, ii)) for ii in boot_idx]
                for m in ma:
                    v = np.array([x[m] for x in bs])
                    paired.append(dict(roster=roster, eval_set=s, contrast=f'{a}-{b}', metric=m,
                                       a=round(ma[m], 4), b=round(mb[m], 4), diff_pp=round(ma[m] - mb[m], 4),
                                       lo=round(np.percentile(v, 2.5), 4), hi=round(np.percentile(v, 97.5), 4),
                                       n_subjects=len(groups), seeds_a=len(seeds[a]), seeds_b=len(seeds[b])))
        summ[roster] = dict(conds=conds, seeds=seeds)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(out / 'checks.csv', index=False, lineterminator='\n')
    pd.DataFrame(rows_m).round(4).to_csv(out / 'metrics_by_run.csv', index=False, lineterminator='\n')
    pdf = pd.DataFrame(paired)
    pdf.to_csv(out / 'paired.csv', index=False, lineterminator='\n')
    T.dump_json(out / 'summary.json', dict(script='scripts/50_b_t1_temporal_paired.py', rosters=summ,
                                           all_checks_ok=bool(all(c['ok'] for c in checks)),
                                           bootstrap=f'{args.n_boot} 次按受试者重抽，rng seed 0，seed 先平均再求差',
                                           test_used=False))
    print('checks ok:', all(c['ok'] for c in checks))
    if len(pdf):
        print(pdf[pdf.eval_set == 'dev8'].to_string(index=False))


if __name__ == '__main__':
    main()
