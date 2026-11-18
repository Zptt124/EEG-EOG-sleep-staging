"""Step W4-B8: 统一复评 C 第四周四格 12 个 run（PR #25，Release c-fourcell-w4-v01-20261119）。

登记表 reports/week4/c_eval_registry_w4.csv（run_id,group,arm,budget,seed,predictions，路径相对仓库根）。
每个 run：
  1. 11 列开发集预测用官方 evaluate.py v1 对照 manifests/val_reference_v1.csv 核对 sample_id 与 y_true（29 号 load）；
     y_pred 是否等于概率 argmax；预测文件 SHA256 与 C 的 artifact_sha256.json 一致。
  2. B 的统一指标（29 号 run_metrics）：Macro-F1、各阶段 F1、N1 P/R/F1、N1 AP/AUROC、N1 去向。
  3. 训练侧 OOF（oof_train.csv）：受试者必须正好是该预算的训练名单、不含开发集；
     E* 处的 OOF Macro-F1 / N1 F1 与 C 的 training_side_summary.json 比对（C 未交逐轮曲线，E* 本身无法重选）。
  4. N1 α：B 用 28 号的网格与并列规则在 OOF 上独立重选，用于开发集；α 和调整后指标与 C 的 metrics.json 比对。
  5. 与 C 的 metrics.json 比对原始 Macro-F1、N1 F1、N1 AP。
配对（同 budget 同 seed，seed 合并后按 8 名开发受试者 bootstrap 1000 次，29 号 paired）：
  默认 linear 与 LFT 各自的 pretrained − random_init，pretrained 与 random_init 各自的 LFT − linear。
  同一组配对另在 α 调整后的预测上再算一遍（arm 名加 @adj），看 N1 F1 / Macro-F1 在门槛校正后的差。
  点估计与 C 的 paired_comparisons.json 比对（区间的重采样次数和种子不同，只比较是否同向、量级相近）。
输出（--out，默认 reports/week4/c_reeval）：
  metrics.csv、summary.csv、paired.csv、paired_by_subject.csv、check_vs_c.csv、reeval_run.json

用法：
  python scripts/38_b_w4_c_reeval.py                       # PR #25 合并后，在仓库根直接运行
  python scripts/38_b_w4_c_reeval.py --root <C 分支导出目录>  # 合并前
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import BASE
import w2_common as w

ev = w.ev
PROB = w.PROB_COLS


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


E29 = _load('eval29', '29_b_w3_evaluate.py')
A28 = _load('adjust28', '28_b_n1_score_adjust.py')

DEFAULT_PAIRS = ['linear_pretrained:linear_random_init', 'lft_pretrained:lft_random_init',
                 'lft_pretrained:linear_pretrained', 'lft_random_init:linear_random_init']
TOL = 1e-9


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def budget_subjects(budget):
    rows = csv.DictReader(open(BASE / 'manifests' / 'label_subsets_v2.csv', encoding='utf-8'))
    return {r['subject_id'] for r in rows if r[budget] == '1'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(BASE), help='预测路径的根目录（C 分支导出目录或仓库根）')
    ap.add_argument('--registry', default='reports/week4/c_eval_registry_w4.csv', help='相对 --root')
    ap.add_argument('--c-dir', default='experiment3/week4/fourcell_v01', help='相对 --root')
    ap.add_argument('--pairs', nargs='*', default=DEFAULT_PAIRS)
    ap.add_argument('--out', default=str(BASE / 'reports' / 'week4' / 'c_reeval'))
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    cdir = root / args.c_dir
    out.mkdir(parents=True, exist_ok=True)

    reg = pd.read_csv(root / args.registry)
    if reg['run_id'].duplicated().any():
        raise SystemExit('登记表 run_id 重复')
    c_sha = json.loads((cdir / 'artifact_sha256.json').read_text(encoding='utf-8'))
    c_sha = c_sha.get('files', c_sha)
    c_train = json.loads((cdir / 'training_side_summary.json').read_text(encoding='utf-8'))
    dev_subjects = set(pd.read_csv(w.VAL_REFERENCE)['subject_id'])

    preds, preds_adj, metrics, checks = {}, {}, [], []
    for r in reg.to_dict('records'):
        pred_path = root / r['predictions']
        df, sha = E29.load(pred_path.resolve())
        preds[r['run_id']] = df
        run_dir = pred_path.parent
        rel = str(pred_path.relative_to(cdir)) if pred_path.is_relative_to(cdir) else r['predictions']
        sha_c = c_sha.get(rel) or c_sha.get(r['predictions'])
        argmax_ok = bool((df[PROB].to_numpy().argmax(1) == df['y_pred'].to_numpy()).all())
        m = E29.run_metrics(r, df, sha)

        oof = pd.read_csv(run_dir / 'oof_train.csv')
        oof_subj = set(oof['subject_id'])
        want = budget_subjects(r['budget'])
        oof_m = ev.compute_metrics(oof['y_true'].to_numpy(), oof['y_pred'].to_numpy())
        alpha, oof_adj, oof_raw = A28.pick_alpha(oof)
        adj = A28.adjusted(df, alpha)
        adj_m = ev.compute_metrics(adj['y_true'].to_numpy(), adj['y_pred'].to_numpy())
        preds_adj[r['run_id'] + '@adj'] = adj
        m.update(selected_epoch=None, oof_macro_f1=oof_m['macro_f1'], oof_n1_f1=oof_m['n1_f1'],
                 alpha=alpha, oof_n1_f1_raw=oof_raw, oof_n1_f1_adjusted=oof_adj,
                 n1_f1_adjusted=adj_m['n1_f1'], macro_f1_adjusted=adj_m['macro_f1'],
                 n1_precision_adjusted=adj_m['n1_precision'], n1_recall_adjusted=adj_m['n1_recall'])

        cm = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
        seed_key, arm = f"seed{r['seed']}", r['arm']
        ct = c_train['seeds'][seed_key]['arms'][arm]
        m['selected_epoch'] = ct['epoch']
        chk = dict(run_id=r['run_id'],
                   reference_ok=True, y_pred_is_argmax=argmax_ok,
                   pred_sha256_match=(sha256_file(pred_path) == sha_c) if sha_c else None,
                   oof_subjects_ok=oof_subj == want and not (oof_subj & dev_subjects),
                   oof_sha256_match=sha256_file(run_dir / 'oof_train.csv') == cm.get('oof_sha256'),
                   d_macro_f1=m['macro_f1'] - cm['macro_f1'], d_n1_f1=m['n1_f1'] - cm['n1_f1'],
                   d_n1_ap=m['n1_ap'] - cm['n1_ap'],
                   d_oof_macro_f1=oof_m['macro_f1'] - ct['oof_macro_f1'], d_oof_n1_f1=oof_m['n1_f1'] - ct['oof_n1_f1'],
                   alpha_b=alpha, alpha_c=cm['alpha'], alpha_rel_diff=abs(alpha - cm['alpha']) / cm['alpha'],
                   d_n1_f1_adjusted=adj_m['n1_f1'] - cm['adjusted']['n1_f1'],
                   d_macro_f1_adjusted=adj_m['macro_f1'] - cm['adjusted']['macro_f1'])
        chk['all_match'] = bool(chk['y_pred_is_argmax'] and chk['pred_sha256_match'] is not False
                                and chk['oof_subjects_ok'] and chk['oof_sha256_match']
                                and all(abs(chk[k]) < TOL for k in chk if k.startswith('d_'))
                                and chk['alpha_rel_diff'] < 1e-9)
        metrics.append(m)
        checks.append(chk)
        print(f"{r['run_id']}: macroF1 {m['macro_f1']:.4f} N1 AP {m['n1_ap']:.4f} alpha {alpha:.3f} "
              f"adj N1 F1 {adj_m['n1_f1']:.4f}  match={chk['all_match']}", flush=True)

    met = pd.DataFrame(metrics)
    met.to_csv(out / 'metrics.csv', index=False, float_format='%.6f', lineterminator='\n')
    chk = pd.DataFrame(checks)
    chk.to_csv(out / 'check_vs_c.csv', index=False, float_format='%.3g', lineterminator='\n')
    cols = ['macro_f1', 'n1_f1', 'n1_ap', 'n1_auroc', 'n1_f1_adjusted', 'macro_f1_adjusted', 'alpha', 'selected_epoch']
    g = met.groupby(['group', 'arm', 'budget'])
    summ = g[cols].agg(['mean', 'min', 'max'])
    summ.columns = [f'{a}_{b}' for a, b in summ.columns]
    summ.insert(0, 'n_seeds', g.size())
    summ.reset_index().to_csv(out / 'summary.csv', index=False, float_format='%.6f', lineterminator='\n')

    rows, subj = [], []
    for p in args.pairs:
        comp, ref = p.split(':')
        a, b = E29.paired(reg, preds, comp, ref)
        rows += a
        subj += b
    # α 调整后的配对：同一套 bootstrap，看 N1 F1 / Macro-F1 在门槛校正后的差（N1 AP 不受 α 影响，与原始相同）
    reg_adj = reg.assign(run_id=reg['run_id'] + '@adj', arm=reg['arm'] + '@adj')
    for p in args.pairs:
        comp, ref = p.split(':')
        a, b = E29.paired(reg_adj, preds_adj, comp + '@adj', ref + '@adj')
        rows += a
        subj += b
    pr = pd.DataFrame(rows)
    c_pairs = json.loads((cdir / 'paired_comparisons.json').read_text(encoding='utf-8'))
    name_map = {('linear_pretrained', 'linear_random_init'): 'linear_pretrained_minus_random',
                ('lft_pretrained', 'lft_random_init'): 'lft_pretrained_minus_random',
                ('lft_pretrained', 'linear_pretrained'): 'pretrained_lft_minus_linear',
                ('lft_random_init', 'linear_random_init'): 'random_lft_minus_linear'}
    for k in ('macro_f1', 'n1_f1'):
        pr[f'c_delta_{k}'] = [c_pairs.get(name_map.get((x, y), ''), {}).get(k, {}).get('point')
                              for x, y in zip(pr['comparison'], pr['reference'])]
        pr[f'c_delta_{k}_lo'] = [c_pairs.get(name_map.get((x, y), ''), {}).get(k, {}).get('lo')
                                 for x, y in zip(pr['comparison'], pr['reference'])]
        pr[f'c_delta_{k}_hi'] = [c_pairs.get(name_map.get((x, y), ''), {}).get(k, {}).get('hi')
                                 for x, y in zip(pr['comparison'], pr['reference'])]
    pr = pr.astype({c: float for c in pr.columns if c.startswith('c_delta_')})
    pr.to_csv(out / 'paired.csv', index=False, float_format='%.6f', lineterminator='\n')
    pd.DataFrame(subj).to_csv(out / 'paired_by_subject.csv', index=False, float_format='%.6f', lineterminator='\n')

    point_diff = {}
    for _, x in pr[~pr['comparison'].str.endswith('@adj')].iterrows():
        for k in ('macro_f1', 'n1_f1'):
            if x[f'c_delta_{k}'] is not None and not pd.isna(x[f'c_delta_{k}']):
                point_diff[f"{x['comparison']}-{x['reference']}:{k}"] = float(x[f'delta_{k}'] - x[f'c_delta_{k}'])
    run = dict(registry=args.registry, n_runs=len(reg), n_all_match=int(chk['all_match'].sum()),
               pairs=args.pairs, n_boot=E29.N_BOOT, boot_seed=0,
               paired_point_diff_vs_c=point_diff,
               max_abs_point_diff_vs_c=max((abs(v) for v in point_diff.values()), default=None),
               note='E* 只核对选中轮的 OOF 分数；C 未交逐轮曲线，无法独立重选 E*')
    (out / 'reeval_run.json').write_text(json.dumps(run, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(run, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
