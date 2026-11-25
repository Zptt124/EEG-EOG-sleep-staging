"""Step W4-B9: 统一复评跨模态三组的已交付 run（C 的 6 个独立复评 run；D 的 18 个三 head seed run 同一入口）。

两家都用 B 的 34 号探针，run 目录格式相同（predictions_val.csv、oof_train.csv 可选、inner_cv.csv、config.yaml、metrics.json）。
每个 run：
  1. 11 列开发集预测用官方 evaluate.py v1 对照 manifests/val_reference_v1.csv 核对 sample_id 与 y_true（29 号 load）；
     y_pred 是否等于概率 argmax；若给了 --artifact-sha，预测文件 SHA256 须与交付方登记一致。
  2. B 的统一指标（29 号 run_metrics）：Macro-F1、各阶段 F1、N1 P/R/F1、N1 AP/AUROC、N1 去向。
  3. 选轮：在 run 的逐轮曲线 inner_cv.csv 上按“OOF Macro-F1 最高、并列取最早”重选 E*，须等于 config/metrics 里的 E*；
     config 的五折须按受试者划分、正好覆盖该预算训练名单（27 号同一 fold_seed 的划分），且不含开发集受试者。
  4. 若有 oof_train.csv：OOF 受试者 = 预算名单；E* 处 OOF Macro-F1 须等于曲线值；B 用 28 号网格独立重选 α，
     α 与调整后指标须与 metrics.json 一致。没有 OOF 时 α 取交付方的值，只核对调整后指标（标记 alpha_source=deliverer）。
  5. 与 metrics.json 比对原始 Macro-F1、N1 F1、N1 AP、N1 AUROC。
三组配对（35 号同一实现：seed 合并后按 8 名开发受试者成对 bootstrap 1000 次，rng seed 0）：
  同窗−无 EEG、错时−无 EEG、同窗−错时，原始与 α 调整后（arm 名加 @adj）各一遍；若给了 --ref-paired，
  点估计与交付方的配对表逐项相减（区间若同为 1000 次、seed 0、同一实现，应逐位一致）。
输出（--out）：metrics.csv、summary.csv、paired.csv、paired_by_subject.csv、alignment_check.csv、check.csv、reeval_run.json

用法：
  python scripts/39_b_w4_xmodal_reeval.py --root <C 分支导出目录> \
      --registry experiment3/week4/crossmodal_final_v01/c_crossmodal_registry.csv \
      --artifact-sha experiment3/week4/crossmodal_final_v01/artifact_sha256.json \
      --ref-paired experiment3/week4/crossmodal_final_v01/comparison/paired.csv --out reports/week4/xmodal_reeval/c
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import BASE
import w2_common as w

ev = w.ev
PROB = w.PROB_COLS
TOL = 1e-9
ALPHA_TOL = 1e-5  # 交付方 α 表可能只保留 6 位有效数字；网格相邻点相差约 0.7%
ARMS = {'sync': 'xm_sync', 'shift': 'xm_shift', 'none': 'xm_none'}
CONTRASTS = [('sync', 'none'), ('shift', 'none'), ('sync', 'shift')]


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


E29 = _load('eval29', '29_b_w3_evaluate.py')
A28 = _load('adjust28', '28_b_n1_score_adjust.py')
P27 = _load('probe27', '27_b_w3_linear_probe.py')
E35 = _load('eval35', '35_b_w4_crossmodal_eval.py')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(BASE), help='登记表与预测路径的根目录')
    ap.add_argument('--registry', required=True, help='相对 --root；列 run_id,group,arm,budget,seed,predictions')
    ap.add_argument('--artifact-sha', help='相对 --root；交付方的 {路径: sha256} JSON')
    ap.add_argument('--ref-paired', help='相对 --root；交付方的配对表（35 号或 29 号格式）')
    ap.add_argument('--ref-adjust', help='相对 --root；metrics.json 没有 n1_adjust 时，交付方的 α 表'
                    '（28 号格式：arm,budget,seed,alpha,oof_n1_f1_raw,oof_n1_f1_adjusted,after_n1_f1,after_macro_f1）')
    ap.add_argument('--arm-map', nargs='*', default=[], help='交付方 arm 名=sync|shift|none，如 a_same030k_eog=sync')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reg = pd.read_csv(root / args.registry, dtype={'seed': int})
    amap = dict(x.split('=') for x in args.arm_map)
    reg['arm'] = reg['arm'].map(lambda a: ARMS[amap[a]] if a in amap else a)
    ref_adj = None
    if args.ref_adjust:
        ref_adj = pd.read_csv(root / args.ref_adjust)
        ref_adj['arm'] = ref_adj['arm'].map(lambda a: ARMS[amap[a]] if a in amap else a)
        ref_adj = ref_adj.set_index(['arm', 'budget', 'seed'])
    if list(reg.columns) != E29.COLS or reg['run_id'].duplicated().any() or reg[['arm', 'budget', 'seed']].duplicated().any():
        raise SystemExit('登记表列不对，或 run_id / arm×budget×seed 重复')
    shas = {}
    if args.artifact_sha:
        shas = json.loads((root / args.artifact_sha).read_text(encoding='utf-8'))
        shas = shas.get('files', shas)
    dev = set(pd.read_csv(w.VAL_REFERENCE)['subject_id'])

    preds, preds_adj, metrics, checks = {}, {}, [], []
    for r in reg.to_dict('records'):
        pred_path = root / r['predictions']
        run_dir = pred_path.parent
        df, sample_sha = E29.load(pred_path.resolve())
        preds[r['run_id']] = df
        m = E29.run_metrics(r, df, sample_sha)
        cm = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
        cfg_path = run_dir / 'config.yaml'
        cfg = (yaml.safe_load(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists()
               else json.loads((run_dir / 'config.json').read_text(encoding='utf-8')))
        want, _ = w.selected_subjects(str(BASE / 'manifests' / 'label_subsets_v2.csv'), r['budget'])
        want = set(want)

        curve = pd.read_csv(run_dir / 'inner_cv.csv')
        best = curve['oof_macro_f1'].max()
        e_b = int(curve.loc[curve['oof_macro_f1'] == best, 'epoch'].min())
        folds = [set(f) for f in cfg['folds']]
        folds_ref = [set(f) for f in P27.subject_folds(sorted(want), min(5, len(want)))]
        folds_ok = (set().union(*folds) == want and sum(map(len, folds)) == len(want) and not (want & dev)
                    and sorted(map(sorted, folds)) == sorted(map(sorted, folds_ref)))

        oof_path = run_dir / 'oof_train.csv'
        chk = dict(run_id=r['run_id'], y_pred_is_argmax=bool((df[PROB].to_numpy().argmax(1) == df['y_pred'].to_numpy()).all()))
        key = r['predictions']
        chk['pred_sha256_match'] = (sha256_file(pred_path) == shas[key]) if key in shas else None
        chk['pred_sha256_config_match'] = sha256_file(pred_path) == cfg.get('predictions_sha256')
        chk.update(epoch_reselected=e_b, epoch_reported=int(cm['selected_epoch']),
                   epoch_match=e_b == int(cm['selected_epoch']) == int(cfg['selected_epoch']),
                   n_curve_epochs=len(curve), folds_ok=bool(folds_ok))
        na = cm.get('n1_adjust')
        from_table = na is None  # 交付方 α 表只有 6 位有效数字，α 相关量按 1e-6 比较
        if na is None and ref_adj is not None:
            x = ref_adj.loc[(r['arm'], r['budget'], int(r['seed']))]
            na = dict(alpha=float(x['alpha']), oof_n1_f1_raw=float(x['oof_n1_f1_raw']),
                      oof_n1_f1_adjusted=float(x['oof_n1_f1_adjusted']), n1_f1_adjusted=float(x['after_n1_f1']),
                      macro_f1_adjusted=float(x['after_macro_f1']))
        if na is None:
            raise SystemExit(f"{r['run_id']}: 没有 n1_adjust，也没给 --ref-adjust")
        if oof_path.exists():
            oof = pd.read_csv(oof_path)
            oof_m = ev.compute_metrics(oof['y_true'].to_numpy(), oof['y_pred'].to_numpy())
            alpha, oof_adj, oof_raw = A28.pick_alpha(oof)
            oof_subj = set(oof['subject_id'])
            fold_sets = {frozenset(g) for g in oof.groupby('fold')['subject_id'].unique()}
            chk.update(alpha_source='B_reselected',
                       oof_subjects_ok=oof_subj == want and not (oof_subj & dev),
                       oof_folds_match_config=fold_sets == {frozenset(f) for f in folds},
                       d_curve_at_estar=float(curve.loc[curve['epoch'] == e_b, 'oof_macro_f1'].iloc[0]) - oof_m['macro_f1'],
                       d_oof_n1_f1_raw=oof_raw - na['oof_n1_f1_raw'], d_oof_n1_f1_adjusted=oof_adj - na['oof_n1_f1_adjusted'])
            m.update(oof_macro_f1=oof_m['macro_f1'], oof_n1_f1=oof_m['n1_f1'])
        else:
            alpha = float(na['alpha'])
            chk.update(alpha_source='deliverer', oof_subjects_ok=None, oof_folds_match_config=None)
        adj = A28.adjusted(df, alpha)
        adj_m = ev.compute_metrics(adj['y_true'].to_numpy(), adj['y_pred'].to_numpy())
        preds_adj[r['run_id'] + '@adj'] = adj
        nadj_path = run_dir / 'predictions_val_n1adj.csv'
        if nadj_path.exists():
            chk['n1adj_file_match'] = bool((pd.read_csv(nadj_path)['y_pred'].to_numpy() == adj['y_pred'].to_numpy()).all())
        m.update(selected_epoch=e_b, n1_alpha=alpha, n1_f1_adjusted=adj_m['n1_f1'], macro_f1_adjusted=adj_m['macro_f1'],
                 n1_precision_adjusted=adj_m['n1_precision'], n1_recall_adjusted=adj_m['n1_recall'])
        chk.update(alpha_b=alpha, alpha_reported=na['alpha'], alpha_rel_diff=abs(alpha - na['alpha']) / na['alpha'],
                   d_macro_f1=m['macro_f1'] - cm['macro_f1'], d_n1_f1=m['n1_f1'] - cm['n1_f1'],
                   d_n1_ap=(m['n1_ap'] - cm['n1_ap']) if 'n1_ap' in cm else 0.0,
                   d_n1_auroc=(m['n1_auroc'] - cm['n1_auroc']) if 'n1_auroc' in cm else 0.0,
                   d_n1_f1_adjusted=adj_m['n1_f1'] - na['n1_f1_adjusted'],
                   d_macro_f1_adjusted=adj_m['macro_f1'] - na['macro_f1_adjusted'])
        flags = [chk['y_pred_is_argmax'], chk['pred_sha256_match'] is not False, chk['pred_sha256_config_match'],
                 chk['epoch_match'], chk['folds_ok'], chk['oof_subjects_ok'] is not False,
                 chk['oof_folds_match_config'] is not False, chk.get('n1adj_file_match', True), chk['alpha_rel_diff'] < ALPHA_TOL]
        adj_keys = ('d_oof_n1_f1_raw', 'd_oof_n1_f1_adjusted', 'd_n1_f1_adjusted', 'd_macro_f1_adjusted')
        chk['alpha_from_table'] = from_table
        chk['all_match'] = bool(all(flags) and all(abs(v) < (1e-6 if from_table and k in adj_keys else TOL)
                                                   for k, v in chk.items() if k.startswith('d_')))
        metrics.append(m)
        checks.append(chk)
        print(f"{r['run_id']}: macroF1 {m['macro_f1']:.4f} N1 AP {m['n1_ap']:.4f} E* {e_b} alpha {alpha:.3f} "
              f"adj N1 F1 {adj_m['n1_f1']:.4f}  match={chk['all_match']}", flush=True)

    met = pd.DataFrame(metrics)
    met.insert(3, 'group_role', met['arm'].map({v: k for k, v in ARMS.items()}))
    met.to_csv(out / 'metrics.csv', index=False, float_format='%.6f', lineterminator='\n')
    chk = pd.DataFrame(checks)
    chk.to_csv(out / 'check.csv', index=False, float_format='%.3g', lineterminator='\n')
    aligned = E35.check_alignment(reg, preds, list(ARMS.values()))
    pd.DataFrame(aligned).to_csv(out / 'alignment_check.csv', index=False, lineterminator='\n')
    cols = ['macro_f1', 'n1_ap', 'n1_f1', 'n1_auroc', 'n1_alpha', 'n1_f1_adjusted', 'macro_f1_adjusted', 'selected_epoch']
    g = met.groupby(['group_role', 'arm', 'budget'])
    summ = g[cols].agg(['mean', 'min', 'max'])
    summ.columns = [f'{a}_{b}' for a, b in summ.columns]
    summ.insert(0, 'n_seeds', g.size())
    summ.reset_index().to_csv(out / 'summary.csv', index=False, float_format='%.6f', lineterminator='\n')

    rows, subj = [], []
    reg_adj = reg.assign(run_id=reg['run_id'] + '@adj', arm=reg['arm'] + '@adj')
    for tag, rg, pr, suffix in (('raw', reg, preds, ''), ('adjusted', reg_adj, preds_adj, '@adj')):
        for c, r in CONTRASTS:
            a, b = E29.paired(rg, pr, ARMS[c] + suffix, ARMS[r] + suffix)
            for x in a + b:
                x.update(contrast=f'{c}-{r}', scores=tag)
            rows += a
            subj += b
    paired = pd.DataFrame(rows)
    paired = paired[['contrast', 'scores'] + [c for c in paired.columns if c not in ('contrast', 'scores')]]

    diff = {}
    if args.ref_paired:
        ref = pd.read_csv(root / args.ref_paired)
        raw = paired[paired['scores'] == 'raw']
        if 'contrast' not in ref:
            ref['contrast'] = ref['comparison'].map(amap) + '-' + ref['reference'].map(amap)
        for _, x in raw.iterrows():
            y = ref[(ref['contrast'] == x['contrast']) & (ref['budget'] == x['budget'])]
            if len(y) != 1:
                continue
            y = y.iloc[0]
            for k in [c for c in raw.columns if c.startswith('delta_') and c in ref.columns]:
                diff[f"{x['contrast']}:{x['budget']}:{k}"] = float(x[k] - y[k])
    paired.to_csv(out / 'paired.csv', index=False, float_format='%.6f', lineterminator='\n')
    ps = pd.DataFrame(subj)
    ps = ps[['contrast', 'scores'] + [c for c in ps.columns if c not in ('contrast', 'scores')]]
    ps.to_csv(out / 'paired_by_subject.csv', index=False, float_format='%.6f', lineterminator='\n')

    run = dict(script='scripts/39_b_w4_xmodal_reeval.py', registry=args.registry, n_runs=len(reg),
               n_all_match=int(chk['all_match'].sum()), alignment=aligned,
               bootstrap=f'{E29.N_BOOT} 次，按开发集受试者成对重采样，seed 合并，rng seed 0',
               val_reference_sha256=w.sha256_file(w.VAL_REFERENCE), code_commit=w.code_version(),
               ref_paired=args.ref_paired,
               max_abs_diff_vs_ref_paired=max((abs(v) for v in diff.values()), default=None),
               diff_vs_ref_paired=diff)
    (out / 'reeval_run.json').write_text(json.dumps(run, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    show = paired[paired['scores'] == 'raw'][['contrast', 'budget', 'seeds', 'delta_macro_f1', 'delta_macro_f1_lo',
                                               'delta_macro_f1_hi', 'delta_n1_ap', 'delta_n1_ap_lo', 'delta_n1_ap_hi']]
    print(show.round(4).to_string(index=False))
    print(f"all_match {run['n_all_match']}/{len(reg)}  max |Δ vs ref paired| {run['max_abs_diff_vs_ref_paired']}")


if __name__ == '__main__':
    main()
