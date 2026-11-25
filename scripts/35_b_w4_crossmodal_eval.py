"""Step W4-B5: 跨模态三组统一复评（同窗 EEG / 错时 EEG / 无 EEG，下游均为 EOG-only）。

按 experiment3/week4/cross_modal_protocol.md「评估与交付」：
  1. 每份 11 列预测先用官方 evaluate.py v1 对照 manifests/val_reference_v1.csv 核对覆盖与真值。
  2. 同一 budget、同一 head seed 的三组预测逐行对齐：按 sample_id 排序后 sample_id、subject_id、record_id、
     epoch_index（时间 = 30·epoch_index 秒）与 y_true 必须完全一致，否则报错停止。
  3. 绝对分数：Macro-F1、N1 AP、N1 F1（原始）、N1 AUROC；若 run 目录有 34 号脚本写的 metrics.json，
     另列训练侧 OOF 选的 α、调整后 N1 F1 与调整后 Macro-F1。
  4. 三个配对差：同窗−无 EEG、错时−无 EEG、同窗−错时。各 seed 合并后按开发集受试者成对 bootstrap（1000 次）
     报差值与 95% 区间（Macro-F1、N1 F1、N1 AP，与 29 号脚本同一实现），另报逐 seed 差值（含 N1 AP 与调整后指标）。
开发集 8 人反复使用；区间只反映受试者抽样，不含 backbone、head seed 与选模的不确定性。

登记表 CSV 列：run_id,group,arm,budget,seed,predictions（同 29 号脚本；predictions 相对仓库根或绝对路径）。

用法：
  python scripts/35_b_w4_crossmodal_eval.py --registry <A.csv> <B.csv> <D.csv> \
      --sync-arm xm_sync --shift-arm xm_shift --none-arm xm_none --out reports/week4/b_crossmodal
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import BASE
import w2_common as w

ev = w.ev
ALIGN_COLS = ['sample_id', 'subject_id', 'record_id', 'epoch_index', 'y_true']

_s = importlib.util.spec_from_file_location('eval29', Path(__file__).with_name('29_b_w3_evaluate.py'))
E29 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E29)


def load_registry(paths):
    reg = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if list(reg.columns) != E29.COLS:
        raise ValueError(f'登记表列应为 {E29.COLS}')
    if reg['run_id'].duplicated().any():
        raise ValueError('run_id 重复')
    if reg[['arm', 'budget', 'seed']].duplicated().any():
        raise ValueError('同一 arm×budget×seed 登记了多份预测')
    return reg


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else BASE / p


def adjusted_metrics(pred_path):
    """34 号脚本在 run 目录写的 metrics.json 中的 N1 调整结果；没有则返回空。"""
    m = resolve(pred_path).parent / 'metrics.json'
    if not m.exists():
        return {}
    a = json.loads(m.read_text(encoding='utf-8')).get('n1_adjust')
    if not a:
        return {}
    return dict(n1_alpha=a['alpha'], n1_f1_adjusted=a['n1_f1_adjusted'], macro_f1_adjusted=a['macro_f1_adjusted'])


def check_alignment(reg, preds, arms):
    """同 budget×seed 的三组逐行对齐；返回已核对的 (budget, seed) 列表。"""
    done = []
    for (budget, seed), g in reg[reg['arm'].isin(arms)].groupby(['budget', 'seed']):
        have = set(g['arm'])
        if have != set(arms):
            print(f'提示：{budget} seed {seed} 只有 {sorted(have)}，缺 {sorted(set(arms) - have)}，不参与该 seed 的三组配对')
        frames = {r.arm: preds[r.run_id].sort_values('sample_id').reset_index(drop=True)[ALIGN_COLS] for r in g.itertuples()}
        ref_arm = sorted(frames)[0]
        for arm, f in frames.items():
            if len(f) != len(frames[ref_arm]):
                raise ValueError(f'{budget} seed {seed}: {arm} 行数 {len(f)} ≠ {ref_arm} {len(frames[ref_arm])}')
            for c in ALIGN_COLS:
                bad = (f[c].astype(str) != frames[ref_arm][c].astype(str))
                if bad.any():
                    i = int(np.flatnonzero(bad)[0])
                    raise ValueError(f'{budget} seed {seed}: {arm} 与 {ref_arm} 的 {c} 在第 {i} 行不一致 '
                                     f'({f[c].iloc[i]} vs {frames[ref_arm][c].iloc[i]})')
        done.append(dict(budget=budget, seed=int(seed), arms=' '.join(sorted(have)), n_epochs=len(frames[ref_arm])))
    return done


def per_seed_extra(reg, met, comp, ref):
    """逐 seed 的 N1 AP 与调整后指标差（29 号脚本只给 Macro-F1、N1 F1 的逐 seed 差）。"""
    out = {}
    for budget in sorted(set(reg['budget'])):
        a = met[(met.arm == comp) & (met.budget == budget)].set_index('seed')
        b = met[(met.arm == ref) & (met.budget == budget)].set_index('seed')
        seeds = sorted(set(a.index) & set(b.index))
        row = dict(per_seed_delta_n1_ap=' '.join(f'{a.loc[s, "n1_ap"] - b.loc[s, "n1_ap"]:+.4f}' for s in seeds))
        for k in ('n1_f1_adjusted', 'macro_f1_adjusted'):
            if k in met and a.loc[seeds, k].notna().all() and b.loc[seeds, k].notna().all():
                row[f'per_seed_delta_{k}'] = ' '.join(f'{a.loc[s, k] - b.loc[s, k]:+.4f}' for s in seeds)
        out[budget] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--registry', nargs='+', required=True)
    ap.add_argument('--sync-arm', default='xm_sync')
    ap.add_argument('--shift-arm', default='xm_shift')
    ap.add_argument('--none-arm', default='xm_none')
    ap.add_argument('--out', default=str(BASE / 'reports' / 'week4' / 'b_crossmodal'))
    args = ap.parse_args(argv)

    arms = {'sync': args.sync_arm, 'shift': args.shift_arm, 'none': args.none_arm}
    reg = load_registry(args.registry)
    reg = reg[reg['arm'].isin(arms.values())].reset_index(drop=True)
    if set(reg['arm']) != set(arms.values()):
        raise ValueError(f'登记表缺少 arm：{sorted(set(arms.values()) - set(reg["arm"]))}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

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
    met.insert(3, 'group_role', met['arm'].map({v: k for k, v in arms.items()}))
    aligned = check_alignment(reg, preds, list(arms.values()))
    pd.DataFrame(aligned).to_csv(out / 'alignment_check.csv', index=False)

    met.to_csv(out / 'metrics.csv', index=False, float_format='%.6g')
    show = [c for c in ['macro_f1', 'n1_ap', 'n1_f1', 'n1_auroc', 'n1_alpha', 'n1_f1_adjusted', 'macro_f1_adjusted',
                        'n1_true_to_W', 'n1_true_to_N2'] if c in met]
    summ = met.groupby(['group_role', 'arm', 'budget'])[show].agg(['mean', 'min', 'max'])
    summ.columns = [f'{a}_{b}' for a, b in summ.columns]
    summ.insert(0, 'n_seeds', met.groupby(['group_role', 'arm', 'budget']).size())
    summ.reset_index().to_csv(out / 'summary.csv', index=False, float_format='%.6g')

    pairs = [('sync', 'none'), ('shift', 'none'), ('sync', 'shift')]
    prow, psub = [], []
    for c, r in pairs:
        p, s = E29.paired(reg, preds, arms[c], arms[r])
        extra = per_seed_extra(reg, met, arms[c], arms[r])
        for row in p:
            row.update(extra[row['budget']])
            row['contrast'] = f'{c}-{r}'
        for row in s:
            row['contrast'] = f'{c}-{r}'
        prow += p
        psub += s
    paired = pd.DataFrame(prow)
    paired.insert(0, 'contrast', paired.pop('contrast'))
    paired.to_csv(out / 'paired.csv', index=False, float_format='%.6g')
    ps = pd.DataFrame(psub)
    ps.insert(0, 'contrast', ps.pop('contrast'))
    ps.to_csv(out / 'paired_by_subject.csv', index=False, float_format='%.6g')

    meta = dict(script='scripts/35_b_w4_crossmodal_eval.py', arms=arms, registries=[str(p) for p in args.registry],
                n_runs=len(reg), alignment=aligned, val_reference_sha256=w.sha256_file(w.VAL_REFERENCE),
                bootstrap=f'{E29.N_BOOT} 次，按开发集受试者成对重采样，seed 合并，rng seed 0', code_commit=w.code_version())
    (out / 'eval_run.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(met.groupby(['group_role', 'budget'])[show].mean().round(4).to_string())
    cols = ['contrast', 'budget', 'seeds'] + [f'delta_{k}{x}' for k in ('macro_f1', 'n1_ap') for x in ('', '_lo', '_hi')]
    print(paired[cols].round(4).to_string(index=False))
    print(f'已写入 {out}')


if __name__ == '__main__':
    main()
