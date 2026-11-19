"""W4 下游线性 probe（B 第三周统一协议，CPU，每个 run 约 45 秒）。

在第三周 08_bprotocol_probes.py 基础上改成通用版：臂（arm）由命令行给出，不再写死。
直接调用 B 仓库里的函数，只替换“读特征”这一步：
  27_b_w3_linear_probe.py：预算内按受试者 5 折、固定 50 轮、取 OOF Macro-F1 最高轮次 E*、全名单重训 E* 轮、dev 只预测一次
  28_b_n1_score_adjust.py：N1 分数调整，alpha 只在训练侧 OOF 上选
  29_b_w3_evaluate.py：统一指标（含 N1 AP）、按受试者 bootstrap 配对比较
  src/evaluation/evaluate.py v1：官方评分与 val_reference_v1 核对

用法：
  python -u 05_probes.py --tag main_ckpts \\
      --arm d_rand=<features_dir>:eeg,eog:std --arm d_pre030k=<features_dir>:eeg,eog:std \\
      --budgets label25,label10 --seeds 0,1,2 --pair d_pre030k:d_rand
  features_dir = 04_extract_features.py 的输出目录（含 DONE、index.csv、{eeg,eog}_{std,mean,feedback}.npy）。
输出：exp_d_pretrain/runs_bprotocol/W4_D_<arm>_<budget>_sseed2024_seed<k>_v01/（与 B 同格式）
      exp_d_pretrain/analysis/<tag>/：registry、metrics、summary、paired、n1_adjust
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import w4d_common as C


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_arm(text, skip_missing):
    try:
        name, rest = text.split('=', 1)
        d, mods, kind = rest.rsplit(':', 2)
    except ValueError:
        C.fail(f'--arm must look like name=<features_dir>:eeg,eog:std, got {text}')
    d = Path(d).expanduser()
    mods = mods.split(',')
    if not (d / 'DONE').exists():
        if skip_missing:
            print(f'WARNING: skip arm {name}: features not found in {d}', flush=True)
            return None
        C.fail(f'Missing features {d}')
    files = [d / f'{m}_{kind}.npy' for m in mods]
    for f in files:
        if not f.exists():
            C.fail(f'Missing feature file {f}')
    info = json.loads((d / 'features_info.json').read_text(encoding='utf-8'))
    return name, dict(dir=d, files=files, index=d / 'index.csv',
                      desc=f'{info.get("checkpoint")} step {info.get("step")} {info.get("weights")} '
                           f'switches {info.get("switches")}; {"+".join(mods)} {kind}; pool {info.get("pool_sec")} s',
                      checkpoint_sha256=info.get('checkpoint_sha256'))


def load_xy(src, w, chosen):
    rows = w.read_csv_rows(src['index'])
    train, val = w.select_rows(rows, chosen)
    pos = {r['sample_id']: i for i, r in enumerate(rows)}
    out = []
    for part in (train, val):
        idx = np.array([pos[r['sample_id']] for r in part])
        out.append(np.concatenate([np.load(f, mmap_mode='r')[idx].astype(np.float32) for f in src['files']], axis=1))
    return train, val, out[0], out[1]


def run_probe(arm, src, budget, seed, b27, w, ev, out_root, repo_commit, subsets):
    import torch
    run_id = f'W4_D_{arm}_{budget}_sseed2024_seed{seed}_v01'
    run_dir = out_root / run_id
    if (run_dir / 'metrics.json').exists():
        cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        if cfg.get('feature_files') != [str(f) for f in src['files']]:
            C.fail(f'{run_id} exists with different features; use another arm name')
        return run_id, run_dir, json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    run_dir.mkdir(parents=True, exist_ok=True)
    chosen, meta = w.selected_subjects(subsets, budget)
    if {meta[s]['sampling_seed'] for s in chosen} != {'2024'}:
        C.fail('sampling_seed should be 2024')
    train, val, x_tr, x_va = load_xy(src, w, chosen)
    y_tr = np.array([int(r['y_true']) for r in train])
    subj = np.array([r['subject_id'] for r in train])
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    # ---- 与 27_b_w3_linear_probe.main 相同的步骤 ----
    folds = b27.subject_folds(chosen, min(5, len(chosen)))
    fold_of = np.empty(len(y_tr), dtype=int)
    oof = np.zeros((b27.MAX_EPOCHS, len(y_tr), 5))
    for k, held in enumerate(folds):
        ho = np.isin(subj, held)
        fold_of[ho] = k
        _, (xf, xh) = b27.standardize(x_tr[~ho], x_tr[ho])
        _, per_epoch, _ = b27.train_head(xf, y_tr[~ho], seed, b27.MAX_EPOCHS, xh)
        oof[:, ho] = np.stack(per_epoch)
    curve = []
    for e in range(b27.MAX_EPOCHS):
        m = ev.compute_metrics(y_tr, oof[e].argmax(1))
        curve.append(dict(epoch=e + 1, oof_macro_f1=m['macro_f1'], oof_n1_f1=m['n1_f1']))
    best = max(curve, key=lambda r: (r['oof_macro_f1'], -r['epoch']))
    e_star = best['epoch']
    (mean, std), (xa, xv) = b27.standardize(x_tr, x_va)
    model, _, updates = b27.train_head(xa, y_tr, seed, e_star)
    probs = w.predict_probs(model, lambda ids: xv[ids], len(xv))
    elapsed = time.perf_counter() - t0
    # ---- 输出（与 B 同格式） ----
    w.write_prediction_table(run_dir / 'predictions_val.csv', val, probs)
    met = ev.evaluate(run_dir / 'predictions_val.csv', reference=w.VAL_REFERENCE)
    met.update(run_id=run_id, selected_epoch=e_star, oof_macro_f1_at_selected=best['oof_macro_f1'])
    (run_dir / 'metrics.json').write_text(json.dumps(met, indent=2, ensure_ascii=False), encoding='utf-8')
    ev.per_subject_table(ev.load_predictions(run_dir / 'predictions_val.csv')).to_csv(run_dir / 'by_subject.csv',
                                                                                      index=False)
    pd.DataFrame(curve).to_csv(run_dir / 'inner_cv.csv', index=False)
    o = pd.DataFrame({k: [r[k] for r in train] for k in ('sample_id', 'subject_id', 'record_id', 'epoch_index')})
    o['y_true'], o['fold'], o['y_pred'] = y_tr, fold_of, oof[e_star - 1].argmax(1)
    for j, c in enumerate(w.PROB_COLS):
        o[c] = oof[e_star - 1][:, j]
    o.to_csv(run_dir / 'oof_train.csv', index=False, float_format='%.8g')
    torch.save(dict(state=model.state_dict(), epoch=e_star, mean=mean, std=std), run_dir / 'final.ckpt')
    C.write_json(run_dir / 'config.json', dict(
        experiment='W4 experiment D: linear probe on frozen features (B week-3 protocol)', arm=arm, run_id=run_id,
        features=src['desc'], feature_files=[str(f) for f in src['files']], feature_index=str(src['index']),
        checkpoint_sha256=src['checkpoint_sha256'],
        feature_dim=int(x_tr.shape[1]), nominal_budget=budget, labeled_subjects=len(chosen), sampling_seed=2024,
        downstream_seed=seed, train_samples=int(len(y_tr)), val_samples=len(val),
        head=f'Linear({x_tr.shape[1]},5)', optimizer='AdamW lr=1e-3 wd=1e-4', batch_size=128,
        standardization='fit on training-side subjects only (fold-train in inner CV, all budget subjects for final)',
        selection=f'B 27_b_w3_linear_probe: subject-wise CV K={len(folds)} fold_seed={b27.FOLD_SEED}, '
                  f'{b27.MAX_EPOCHS} epochs per fold, epoch=argmax pooled OOF macro-F1 (ties -> earliest)',
        folds=folds, selected_epoch=e_star, dev_used_for='final evaluation only', test_used=False,
        b_code_commit=repo_commit, evaluate_version=ev.EVALUATE_VERSION,
        val_reference_sha256=w.sha256_file(w.VAL_REFERENCE),
        predictions_sha256=w.sha256_file(run_dir / 'predictions_val.csv'), final_updates=updates,
        elapsed_seconds=round(elapsed, 2)))
    return run_id, run_dir, met


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True, help='analysis/<tag>/ output folder')
    p.add_argument('--arm', action='append', required=True, help='name=<features_dir>:eeg,eog:std (repeatable)')
    p.add_argument('--budgets', default='label25,label10')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--pair', action='append', default=[], help='comp:ref paired comparison (repeatable)')
    p.add_argument('--skip-missing', action='store_true', help='skip arms whose features are not ready')
    p.add_argument('--repo', default=str(C.HOME / 'repos/w3_main'), help='team repo clone with B scripts 27/28/29')
    args = p.parse_args()

    repo = Path(args.repo)
    for f in ('scripts/27_b_w3_linear_probe.py', 'scripts/28_b_n1_score_adjust.py', 'scripts/29_b_w3_evaluate.py',
              'src/evaluation/evaluate.py', 'manifests/val_reference_v1.csv'):
        if not (repo / f).exists():
            C.fail(f'Missing {repo / f}; clone/update the team repo first')
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    sys.path.insert(0, str(repo / 'scripts'))
    import w2_common as w
    ev = w.ev
    b27 = load_module(repo / 'scripts/27_b_w3_linear_probe.py', 'b27')
    b28 = load_module(repo / 'scripts/28_b_n1_score_adjust.py', 'b28')
    b29 = load_module(repo / 'scripts/29_b_w3_evaluate.py', 'b29')
    subsets = repo / 'manifests/label_subsets_v2.csv'
    if w.sha256_file(subsets) != C.LABEL_SUBSETS_SHA:
        C.fail('label_subsets_v2.csv in repo differs from the team manifest')
    print(f'B code commit {commit}; evaluate {ev.EVALUATE_VERSION}', flush=True)

    srcs = {}
    for a in args.arm:
        r = parse_arm(a, args.skip_missing)
        if r:
            srcs[r[0]] = r[1]
    if not srcs:
        C.fail('No arm has features ready')
    budgets = args.budgets.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    out_root = C.EXP_D / 'runs_bprotocol'
    ana = C.ANALYSIS / args.tag
    ana.mkdir(parents=True, exist_ok=True)

    reg = []
    t_all = time.perf_counter()
    n_total = len(srcs) * len(budgets) * len(seeds)
    for arm, src in srcs.items():
        for budget in budgets:
            for seed in seeds:
                t0 = time.perf_counter()
                run_id, run_dir, met = run_probe(arm, src, budget, seed, b27, w, ev, out_root, commit, subsets)
                reg.append(dict(run_id=run_id, group='D', arm=arm, budget=budget, seed=seed,
                                predictions=str(run_dir / 'predictions_val.csv')))
                print(f'[{len(reg):3d}/{n_total}] {run_id}: E*={met["selected_epoch"]} dev macroF1 '
                      f'{met["macro_f1"]:.4f} N1 P/R/F1 {met["n1_precision"]:.3f}/{met["n1_recall"]:.3f}/'
                      f'{met["n1_f1"]:.3f} ({time.perf_counter() - t0:.0f}s, '
                      f'total {(time.perf_counter() - t_all) / 60:.1f} min)', flush=True)
    reg = pd.DataFrame(reg, columns=b29.COLS)
    reg.to_csv(ana / 'd_eval_registry.csv', index=False)

    preds, rows = {}, []
    for r in reg.to_dict('records'):
        df, sha = b29.load(r['predictions'])
        preds[r['run_id']] = df
        rows.append(b29.run_metrics(r, df, sha))
    met = pd.DataFrame(rows)
    met.to_csv(ana / 'd_metrics.csv', index=False, float_format='%.6g')
    show = ['macro_f1', 'n1_f1', 'n1_precision', 'n1_recall', 'n1_ap', 'f1_REM', 'n1_true_to_W', 'n1_true_to_N2']
    summ = met.groupby(['arm', 'budget'])[show].agg(['mean', 'min', 'max'])
    summ.columns = [f'{a}_{b}' for a, b in summ.columns]
    summ.insert(0, 'n_seeds', met.groupby(['arm', 'budget']).size())
    summ.reset_index().to_csv(ana / 'd_summary.csv', index=False, float_format='%.6g')
    print(met.groupby(['arm', 'budget'])[show].mean().round(4).to_string(), flush=True)

    prow, psub = [], []
    for pr in args.pair:
        comp, ref = pr.split(':')
        if comp not in srcs or ref not in srcs:
            print(f'WARNING: skip pair {pr} (arm missing)', flush=True)
            continue
        r, s = b29.paired(reg, preds, comp, ref)
        prow += r
        psub += s
    if prow:
        pr = pd.DataFrame(prow)
        pr.to_csv(ana / 'd_paired.csv', index=False, float_format='%.6g')
        pd.DataFrame(psub).to_csv(ana / 'd_paired_by_subject.csv', index=False, float_format='%.6g')
        cols = ['comparison', 'reference', 'budget'] + [f'delta_{k}{x}' for k in b29.KEYS for x in ('', '_lo', '_hi')]
        print(pr[cols].round(4).to_string(index=False), flush=True)

    # ---- N1 分数调整（C 的规则：报原始 N1 F1、OOF 选 alpha 后的 F1、alpha、调整后 Macro-F1）：所有 run 都做 ----
    adj = []
    for r in reg.to_dict('records'):
        d = Path(r['predictions']).parent
        oof = pd.read_csv(d / 'oof_train.csv')
        val = ev.load_predictions(d / 'predictions_val.csv')
        ev.check_against_reference(val, pd.read_csv(w.VAL_REFERENCE))
        alpha, oof_f1, oof_f1_raw = b28.pick_alpha(oof)
        bm_, am_ = b28.describe(val), b28.describe(b28.adjusted(val, alpha))
        row = dict(arm=r['arm'], budget=r['budget'], seed=r['seed'], alpha=alpha,
                   oof_n1_f1_raw=oof_f1_raw, oof_n1_f1_adjusted=oof_f1)
        for k in bm_:
            row[f'before_{k}'], row[f'after_{k}'] = bm_[k], am_[k]
        adj.append(row)
    adj = pd.DataFrame(adj)
    adj.to_csv(ana / 'd_n1_adjust.csv', index=False, float_format='%.6g')
    print(adj.groupby(['arm', 'budget'])[['alpha', 'before_n1_f1', 'after_n1_f1', 'before_macro_f1',
                                          'after_macro_f1']].mean().round(4).to_string(), flush=True)
    C.write_json(ana / 'probe_run_info.json', dict(tag=args.tag, arms={k: v['desc'] for k, v in srcs.items()},
                                                   budgets=budgets, seeds=seeds, pairs=args.pair,
                                                   b_code_commit=commit, minutes=(time.perf_counter() - t_all) / 60))
    print(f'total {(time.perf_counter() - t_all) / 60:.1f} min', flush=True)
    print('PROBES DONE', flush=True)


if __name__ == '__main__':
    main()
