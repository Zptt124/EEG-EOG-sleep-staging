"""第 7 步：random-init vs pilot-pretrained 配对比较（CPU，几秒钟）。

- 按 head seed 配对：每个 seed 的差值（pretrained - random）及 3 个 seed 的均值/标准差
- 按受试者配对：每个开发验证者先对 3 个 head seed 取平均，再算差值；8 人重抽样 bootstrap 95% CI
  （只有 1 套预训练、8 个开发验证者、开发集反复使用，区间只反映受试者波动，不代表 backbone 训练波动）
输出：analysis/compare_<run_name>/ 下 by_seed.csv、by_subject.csv、summary.json、summary.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import w3d_common as C

KEYS = ['macro_f1', 'n1_f1', 'n1_precision', 'n1_recall', 'n1_ap', 'rem_f1', 'w_f1', 'n2_f1', 'n3_f1']


def flat(m):
    f = m['per_class_f1']
    return dict(macro_f1=m['macro_f1'], n1_f1=f['N1'], n1_precision=m['per_class_precision']['N1'],
                n1_recall=m['per_class_recall']['N1'], n1_ap=m['n1_ap'], rem_f1=f['REM'], w_f1=f['W'],
                n2_f1=f['N2'], n3_f1=f['N3'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=True)
    p.add_argument('--random-tag', default='step0_ema_model')
    p.add_argument('--pretrained-tag', required=True)
    p.add_argument('--suffix', default='std_eegeog')
    p.add_argument('--head-seeds', default='0,1,2')
    p.add_argument('--n-boot', type=int, default=10000)
    args = p.parse_args()

    base = C.EXP_D / 'runs' / args.run_name / 'probe'
    seeds = [int(s) for s in args.head_seeds.split(',')]
    met = {}
    for cond, tag in (('random', args.random_tag), ('pretrained', args.pretrained_tag)):
        for k in seeds:
            path = base / f'{tag}_{args.suffix}_head{k}' / 'metrics.json'
            if not path.exists():
                C.fail(f'Missing {path}')
            met[cond, k] = json.loads(path.read_text(encoding='utf-8'))

    by_seed = []
    for k in seeds:
        a, b = flat(met['random', k]), flat(met['pretrained', k])
        row = dict(head_seed=k)
        for key in KEYS:
            row[f'random_{key}'] = round(a[key], 4)
            row[f'pretrained_{key}'] = round(b[key], 4)
            row[f'delta_{key}'] = round(b[key] - a[key], 4)
        by_seed.append(row)

    subjects = sorted(met['random', seeds[0]]['per_subject'])
    by_subj = []
    for s in subjects:
        row = dict(subject_id=s, n=met['random', seeds[0]]['per_subject'][s]['n'])
        for key in ('macro_f1', 'n1_f1'):
            ra = np.mean([met['random', k]['per_subject'][s][key] for k in seeds])
            pr = np.mean([met['pretrained', k]['per_subject'][s][key] for k in seeds])
            row.update({f'random_{key}': round(ra, 4), f'pretrained_{key}': round(pr, 4),
                        f'delta_{key}': round(pr - ra, 4)})
        by_subj.append(row)

    rng = np.random.default_rng(0)
    summary = dict(run_name=args.run_name, random_tag=args.random_tag, pretrained_tag=args.pretrained_tag,
                   feature_suffix=args.suffix, head_seeds=seeds, n_dev_subjects=len(subjects),
                   caveat='single pretraining run (pilot); 8 dev subjects reused across experiments; '
                          'CI = subject bootstrap of seed-averaged per-subject deltas', results={})
    for key in KEYS:
        d = np.array([r[f'delta_{key}'] for r in by_seed])
        summary['results'][key] = dict(
            random_mean=float(np.mean([r[f'random_{key}'] for r in by_seed])),
            pretrained_mean=float(np.mean([r[f'pretrained_{key}'] for r in by_seed])),
            delta_mean_over_seeds=float(d.mean()), delta_sd_over_seeds=float(d.std(ddof=1)) if len(d) > 1 else None,
            n_seeds_positive=int((d > 0).sum()))
    for key in ('macro_f1', 'n1_f1'):
        d = np.array([r[f'delta_{key}'] for r in by_subj])
        boots = rng.choice(d, size=(args.n_boot, len(d)), replace=True).mean(1)
        summary['results'][key].update(
            per_subject_delta_mean=float(d.mean()), per_subject_ci95=[float(np.percentile(boots, 2.5)),
                                                                      float(np.percentile(boots, 97.5))],
            n_subjects_positive=int((d > 0).sum()))

    out = C.ANALYSIS / f'compare_{args.run_name}_{args.pretrained_tag}_{args.suffix}'
    C.write_csv(out / 'by_seed.csv', by_seed)
    C.write_csv(out / 'by_subject.csv', by_subj)
    C.write_json(out / 'summary.json', summary)
    lines = [f'# random-init vs pretrained ({args.run_name}, {args.pretrained_tag}, {args.suffix})', '',
             '| metric | random | pretrained | delta (mean over seeds) | seeds +/total | subject CI95 |',
             '|---|---|---|---|---|---|']
    for key in KEYS:
        r = summary['results'][key]
        ci = r.get('per_subject_ci95')
        ci_s = f'[{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else ''
        lines.append(f'| {key} | {r["random_mean"]:.4f} | {r["pretrained_mean"]:.4f} | '
                     f'{r["delta_mean_over_seeds"]:+.4f} | {r["n_seeds_positive"]}/{len(seeds)} | {ci_s} |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'saved {out}', flush=True)
    print('COMPARE DONE', flush=True)


if __name__ == '__main__':
    main()
