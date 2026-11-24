"""Step W5-A2: temporal context 三臂配对比较(单 epoch / ±2 拼接 / 小 BiGRU)。

与 35 号(跨模态三组配对)相同的方法:同预算同 seed 的臂逐行对齐,seed 合并后按开发集
受试者成对 bootstrap(29 号 paired,1000 次,rng seed 0)。对比:
  cat−single、bgru−single、bgru−cat;原始预测与 α 调整后预测各出一套。

登记表 CSV 列:run_id,group,arm,budget,seed,predictions(与 29 号 COLS 一致)。
不传 --registry 时从 runs-root 自动收集 runs/W5_A_tctx_*_v01。

用法(仓库根目录):
  python scripts/42_a_w5_temporal_paired.py --out reports/week5/a_temporal_ctx/eval
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
ALIGN_COLS = ['sample_id', 'y_true']


def load_registry(paths):
    reg = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if list(reg.columns) != E29.COLS:
        raise ValueError(f'登记表列应为 {E29.COLS}')
    if reg['run_id'].duplicated().any():
        raise ValueError('run_id 重复')
    if reg[['arm', 'budget', 'seed']].duplicated().any():
        raise ValueError('同一 arm×budget×seed 登记了多份预测')
    return reg


def auto_registry(runs_root):
    rows = []
    for d in sorted(Path(runs_root).glob('W5_A_tctx_*')):
        m = re.fullmatch(r'W5_A_tctx_(.+)_(label\d+)_sseed2024_seed(\d+)_v\d+', d.name)
        if m and (d / 'predictions_val.csv').exists():
            rows.append(dict(run_id=d.name, group='A', arm=f'tctx_{m[1]}', budget=m[2],
                             seed=int(m[3]), predictions=str((d / 'predictions_val.csv').relative_to(BASE))))
    reg = pd.DataFrame(rows, columns=E29.COLS)
    if reg.empty:
        raise ValueError('runs-root 下没有 W5_A_tctx_* run')
    return reg


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else BASE / p


def adjusted_metrics(pred_path):
    m = resolve(pred_path).parent / 'metrics.json'
    if not m.exists():
        return {}
    a = json.loads(m.read_text(encoding='utf-8')).get('n1_adjust')
    if not a:
        return {}
    return dict(n1_alpha=a['alpha'], n1_f1_adjusted=a['n1_f1_adjusted'], macro_f1_adjusted=a['macro_f1_adjusted'])


def adjusted_frame(df, alpha):
    """按 28 号口径重算调整后预测:p_N1 *= alpha,y_pred = argmax(修改后概率)。

    n1adj CSV 与官方校验器不兼容(y_pred 与未修改的概率列不一致),故这里
    现场重算;N1 AP 不受影响(p_N1 单调变换)。"""
    cols = [c for c in df.columns if c.startswith('p_')]
    p = df[cols].to_numpy()
    j = cols.index('p_N1')
    p[:, j] = p[:, j] * alpha
    return df.assign(y_pred=p.argmax(1), p_N1=p[:, j])


def check_alignment(reg, preds):
    done = []
    for (budget, seed), g in reg.groupby(['budget', 'seed']):
        have = set(g['arm'])
        if len(have) < 2:
            continue
        frames = {r.arm: preds[r.run_id].sort_values('sample_id').reset_index(drop=True)[ALIGN_COLS]
                  for r in g.itertuples()}
        ref_arm = sorted(frames)[0]
        for arm, f in frames.items():
            if len(f) != len(frames[ref_arm]):
                raise ValueError(f'{budget} seed {seed}: {arm} 行数 {len(f)} ≠ {ref_arm} {len(frames[ref_arm])}')
            for c in ALIGN_COLS:
                bad = (f[c].astype(str) != frames[ref_arm][c].astype(str))
                if bad.any():
                    raise ValueError(f'{budget} seed {seed}: {arm} 与 {ref_arm} 的 {c} 不一致')
        done.append(dict(budget=budget, seed=int(seed), arms=' '.join(sorted(have)), n_epochs=len(frames[ref_arm])))
    return done


def per_seed_extra(reg, met, comp, ref):
    out = {}
    for budget in sorted(set(reg['budget'])):
        a = met[(met.arm == comp) & (met.budget == budget)].set_index('seed')
        b = met[(met.arm == ref) & (met.budget == budget)].set_index('seed')
        seeds = sorted(set(a.index) & set(b.index))
        if not seeds:
            continue
        row = dict(per_seed_delta_n1_ap=' '.join(f'{a.loc[s, "n1_ap"] - b.loc[s, "n1_ap"]:+.4f}' for s in seeds))
        for k in ('n1_f1_adjusted', 'macro_f1_adjusted'):
            if a.loc[seeds, k].notna().all() and b.loc[seeds, k].notna().all():
                row[f'per_seed_delta_{k}'] = ' '.join(f'{a.loc[s, k] - b.loc[s, k]:+.4f}' for s in seeds)
        out[budget] = row
    return out


def paired_for(reg, preds, comp, ref, met, suffix=''):
    prow, psub = [], []
    for budget in sorted(set(reg['budget'])):
        p, s = E29.paired(reg, preds, comp, ref)
        p = [r for r in p if r['budget'] == budget]
        s = [r for r in s if r['budget'] == budget]
        extra = per_seed_extra(reg, met, comp, ref)
        for row in p:
            row.update(extra.get(budget, {}))
        for row in p:
            row['contrast'] = f'{comp.split("_")[-1]}-{ref.split("_")[-1]}'
        for row in s:
            row['contrast'] = f'{comp.split("_")[-1]}-{ref.split("_")[-1]}'
        prow += p
        psub += s
    out = pd.DataFrame(prow)
    out.insert(0, 'contrast', out.pop('contrast'))
    ps = pd.DataFrame(psub)
    ps.insert(0, 'contrast', ps.pop('contrast'))
    return out, ps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--registry', nargs='+', help='登记表 CSV;缺省自动从 runs-root 收集')
    ap.add_argument('--runs-root', default=str(RUNS))
    ap.add_argument('--out', default=str(BASE / 'reports' / 'week5' / 'a_temporal_ctx' / 'eval'))
    ap.add_argument('--single-arm', default='tctx_single')
    ap.add_argument('--cat-arm', default='tctx_cat')
    ap.add_argument('--bgru-arm', default='tctx_bgru')
    args = ap.parse_args(argv)

    reg = load_registry(args.registry) if args.registry else auto_registry(args.runs_root)
    arms = [args.single_arm, args.cat_arm, args.bgru_arm]
    if set(reg['arm']) != set(arms):
        raise ValueError(f'登记表 arm 集合应为 {sorted(arms)},实际 {sorted(set(reg["arm"]))}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reg.to_csv(out / 'registry.csv', index=False)

    ref = pd.read_csv(w.VAL_REFERENCE)
    preds, rows = {}, []
    for r in reg.to_dict('records'):
        df = ev.load_predictions(resolve(r['predictions']))
        sha = ev.check_against_reference(df, ref)
        preds[r['run_id']] = df
        m = E29.run_metrics(r, df, sha)
        m.update(adjusted_metrics(r['predictions']))
        rows.append(m)
    met = pd.DataFrame(rows)
    aligned = check_alignment(reg, preds)
    pd.DataFrame(aligned).to_csv(out / 'alignment_check.csv', index=False)
    met.to_csv(out / 'metrics.csv', index=False, float_format='%.6g')

    show = [c for c in ['macro_f1', 'n1_ap', 'n1_f1', 'n1_auroc', 'n1_alpha', 'n1_f1_adjusted',
                        'macro_f1_adjusted'] if c in met]
    summ = met.groupby(['arm', 'budget'])[show].agg(['mean', 'min', 'max'])
    summ.columns = [f'{a}_{b}' for a, b in summ.columns]
    summ.insert(0, 'n_seeds', met.groupby(['arm', 'budget']).size())
    summ.reset_index().to_csv(out / 'summary.csv', index=False, float_format='%.6g')

    pairs = [('cat', 'single'), ('bgru', 'single'), ('bgru', 'cat')]
    for tag, suffix in (('', ''), ('_adjusted', '_adjusted')):
        preds_use = preds
        if suffix:
            preds_use = {}
            for run_id in preds:
                alpha = adjusted_metrics(reg[reg.run_id == run_id]['predictions'].iloc[0]).get('n1_alpha')
                if alpha is None:
                    raise ValueError(f'{run_id} 缺 α')
                preds_use[run_id] = adjusted_frame(preds[run_id], alpha)
        prow, psub = [], []
        for c, r in pairs:
            p, s = paired_for(reg, preds_use, dict(cat=args.cat_arm, single=args.single_arm, bgru=args.bgru_arm)[c],
                              dict(cat=args.cat_arm, single=args.single_arm, bgru=args.bgru_arm)[r], met)
            prow += p.to_dict('records')
            psub += s.to_dict('records')
        pd.DataFrame(prow).to_csv(out / f'paired{suffix}.csv', index=False, float_format='%.6g')
        pd.DataFrame(psub).to_csv(out / f'paired_by_subject{suffix}.csv', index=False, float_format='%.6g')

    meta = dict(script='scripts/42_a_w5_temporal_paired.py', arms=dict(single=args.single_arm,
                cat=args.cat_arm, bgru=args.bgru_arm), n_runs=len(reg), alignment=aligned,
                val_reference_sha256=w.sha256_file(w.VAL_REFERENCE),
                bootstrap=f'{E29.N_BOOT} 次,按开发集受试者成对重采样,seed 合并,rng seed 0',
                code_commit=w.code_version())
    (out / 'eval_run.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(met.groupby(['arm', 'budget'])[show].mean().round(4).to_string())
    cols = ['contrast', 'budget', 'seeds'] + [f'delta_{k}{x}' for k in ('macro_f1', 'n1_ap') for x in ('', '_lo', '_hi')]
    print(pd.DataFrame(prow)[cols].round(4).to_string(index=False))
    print(f'已写入 {out}')


if __name__ == '__main__':
    main()
