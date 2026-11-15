"""第 8 步：按 B 第三周统一协议重跑全部线性探针 + N1 分数调整 + 配对比较（CPU，约 40–60 分钟）。

直接调用 B 仓库代码（PR #16，main）里的函数，只替换"读特征"这一步：
  27_b_w3_linear_probe.py：subject_folds / standardize / train_head / MAX_EPOCHS（14 人或 5 人内按受试者 5 折、
      固定 50 轮、取 OOF Macro-F1 最高轮次 E*、全名单重训 E* 轮、开发集只预测一次）
  28_b_n1_score_adjust.py：pick_alpha / adjusted / describe（alpha 只在训练侧 OOF 上选）
  29_b_w3_evaluate.py：run_metrics / paired（seed 合并、按受试者 bootstrap 1000 次）
  src/evaluation/evaluate.py v1：官方评分与 val_reference_v1 核对

臂（全部在看本脚本结果前固定）：
  主比较（label25 与 label10，seed 0-2）：d_rand（step0）、d_pre240k（step240000），EEG+EOG std
  参照（label25 与 label10，seed 0-2）：eegdm_official（第二周 R1 缓存：官方 EEGDM 冻结 EEG+EOG std 2560+2560）
  探索性（label25，seed 0-2）：d_pre015k/030k/060k/120k；d_rand_eeg/d_pre240k_eeg；d_rand_eog/d_pre240k_eog；
                               d_rand_mean/d_pre240k_mean
  N1 分数调整（预先固定 label25、seed 0，同 B）：d_rand、d_pre240k、eegdm_official
输出：exp_d_pretrain/runs_bprotocol/<run_id>/（B 同格式）；analysis/bprotocol/*.csv
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import w3d_common as C

RUN = 'W3_D_pretrain_m64x12_bs108_seed0_v01'
R1_CACHE = C.HOME / 'week2/cache/R1_eegdm_cond14_pooled_v01'
SEEDS = [0, 1, 2]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def d_source(tag, mods, pooling):
    d = C.BIG / 'features' / RUN / tag
    if not (d / 'DONE').exists():
        C.fail(f'Missing features {d}')
    return dict(kind='d', dir=d, files=[d / f'{m}_{pooling}.npy' for m in mods], index=d / 'index.csv',
                desc=f'D small backbone {tag}, {"+".join(mods)}, {pooling} pooling')


def r1_source():
    if not (R1_CACHE / 'cache_manifest.json').exists():
        C.fail(f'Missing R1 cache {R1_CACHE}')
    return dict(kind='r1', dir=R1_CACHE, files=[R1_CACHE / 'eeg_pooled.npy', R1_CACHE / 'eog_pooled.npy'],
                index=R1_CACHE / 'sample_index.csv',
                desc='official EEGDM frozen (week-2 R1 cache: EEG Fpz-Cz cond14 + EOG cond14), std, 2560+2560')


def arms():
    eo = ['eeg', 'eog']
    main = {'d_rand': d_source('step0_ema_model', eo, 'std'),
            'd_pre240k': d_source('step240000_ema_model', eo, 'std'),
            'eegdm_official': r1_source()}
    expl = {f'd_pre{s // 1000:03d}k': d_source(f'step{s}_ema_model', eo, 'std') for s in (15000, 30000, 60000, 120000)}
    for m in ('eeg', 'eog'):
        expl[f'd_rand_{m}'] = d_source('step0_ema_model', [m], 'std')
        expl[f'd_pre240k_{m}'] = d_source('step240000_ema_model', [m], 'std')
    expl['d_rand_mean'] = d_source('step0_ema_model', eo, 'mean')
    expl['d_pre240k_mean'] = d_source('step240000_ema_model', eo, 'mean')
    plan = [(a, b) for a in main for b in ('label25', 'label10')] + [(a, 'label25') for a in expl]
    return {**main, **expl}, plan


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
    run_id = f'W3_D_{arm}_{budget}_sseed2024_seed{seed}_v01'
    run_dir = out_root / run_id
    if (run_dir / 'metrics.json').exists():
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
    ev.per_subject_table(ev.load_predictions(run_dir / 'predictions_val.csv')).to_csv(run_dir / 'by_subject.csv', index=False)
    pd.DataFrame(curve).to_csv(run_dir / 'inner_cv.csv', index=False)
    o = pd.DataFrame({k: [r[k] for r in train] for k in ('sample_id', 'subject_id', 'record_id', 'epoch_index')})
    o['y_true'], o['fold'], o['y_pred'] = y_tr, fold_of, oof[e_star - 1].argmax(1)
    for j, c in enumerate(w.PROB_COLS):
        o[c] = oof[e_star - 1][:, j]
    o.to_csv(run_dir / 'oof_train.csv', index=False, float_format='%.8g')
    torch.save(dict(state=model.state_dict(), epoch=e_star, mean=mean, std=std), run_dir / 'final.ckpt')
    C.write_json(run_dir / 'config.json', dict(
        experiment='W3 experiment D: linear probe on frozen features (B week-3 protocol)', arm=arm, run_id=run_id,
        features=src['desc'], feature_files=[str(f) for f in src['files']], feature_index=str(src['index']),
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
    p.add_argument('--repo', default=str(C.HOME / 'repos/w3_main'))
    args = p.parse_args()
    repo = Path(args.repo)
    for f in ('scripts/27_b_w3_linear_probe.py', 'scripts/28_b_n1_score_adjust.py', 'scripts/29_b_w3_evaluate.py',
              'src/evaluation/evaluate.py', 'manifests/val_reference_v1.csv'):
        if not (repo / f).exists():
            C.fail(f'Missing {repo / f}; clone/update the team repo first')
    import subprocess
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    sys.path.insert(0, str(repo / 'scripts'))
    import w2_common as w
    ev = w.ev
    b27 = load_module(repo / 'scripts/27_b_w3_linear_probe.py', 'b27')
    b28 = load_module(repo / 'scripts/28_b_n1_score_adjust.py', 'b28')
    b29 = load_module(repo / 'scripts/29_b_w3_evaluate.py', 'b29')
    subsets = repo / 'manifests/label_subsets_v2.csv'
    if w.sha256_file(subsets) != C.LABEL_SUBSETS_SHA:
        C.fail('label_subsets_v2.csv in repo differs from the one used for pretraining manifests')
    print(f'B code commit {commit}; evaluate {ev.EVALUATE_VERSION}', flush=True)

    out_root = C.EXP_D / 'runs_bprotocol'
    ana = C.ANALYSIS / 'bprotocol'
    ana.mkdir(parents=True, exist_ok=True)
    srcs, plan = arms()
    reg = []
    t_all = time.perf_counter()
    for i, (arm, budget) in enumerate(plan):
        for seed in SEEDS:
            t0 = time.perf_counter()
            run_id, run_dir, met = run_probe(arm, srcs[arm], budget, seed, b27, w, ev, out_root, commit, subsets)
            reg.append(dict(run_id=run_id, group='D', arm=arm, budget=budget, seed=seed,
                            predictions=str(run_dir / 'predictions_val.csv')))
            print(f'[{i + 1:2d}/{len(plan)}] {run_id}: E*={met["selected_epoch"]} dev macroF1 {met["macro_f1"]:.4f} '
                  f'N1 P/R/F1 {met["n1_precision"]:.3f}/{met["n1_recall"]:.3f}/{met["n1_f1"]:.3f} '
                  f'({time.perf_counter() - t0:.0f}s, total {(time.perf_counter() - t_all) / 60:.1f} min)', flush=True)
    reg = pd.DataFrame(reg, columns=b29.COLS)
    reg.to_csv(ana / 'd_eval_registry_w3.csv', index=False)

    # ---- 统一指标（B 29 的 run_metrics） ----
    preds, rows = {}, []
    for r in reg.to_dict('records'):
        df, sha = b29.load(r['predictions'])
        preds[r['run_id']] = df
        rows.append(b29.run_metrics(r, df, sha))
    met = pd.DataFrame(rows)
    met.to_csv(ana / 'd_w3_metrics.csv', index=False, float_format='%.6g')
    show = ['macro_f1', 'n1_f1', 'n1_precision', 'n1_recall', 'n1_ap', 'f1_REM', 'n1_true_to_W', 'n1_true_to_N2']
    summ = met.groupby(['arm', 'budget'])[show].agg(['mean', 'min', 'max'])
    summ.columns = [f'{a}_{b}' for a, b in summ.columns]
    summ.insert(0, 'n_seeds', met.groupby(['arm', 'budget']).size())
    summ.reset_index().to_csv(ana / 'd_w3_summary.csv', index=False, float_format='%.6g')
    print(met.groupby(['arm', 'budget'])[show].mean().round(4).to_string(), flush=True)

    # ---- 复现核对：eegdm_official vs B 发布的 W3_B_std_t1 指标 ----
    bm = pd.read_csv(repo / 'reports/week3/b_w3_metrics.csv')
    bm = bm[bm['arm'] == 'std_t1'].set_index(['budget', 'seed'])
    chk = []
    for r in met[met['arm'] == 'eegdm_official'].to_dict('records'):
        b = bm.loc[(r['budget'], r['seed'])]
        chk.append(dict(budget=r['budget'], seed=r['seed'], d_macro_f1=r['macro_f1'], b_macro_f1=b['macro_f1'],
                        diff_macro_f1=r['macro_f1'] - b['macro_f1'], d_n1_f1=r['n1_f1'], b_n1_f1=b['n1_f1'],
                        diff_n1_f1=r['n1_f1'] - b['n1_f1']))
    chk = pd.DataFrame(chk)
    chk.to_csv(ana / 'reproduce_B_std_t1_check.csv', index=False, float_format='%.6g')
    print('Reproduction of B std_t1 with week-2 R1 cache:\n' + chk.round(4).to_string(index=False), flush=True)

    # ---- 配对比较（B 29 的 paired） ----
    pairs = [('d_pre240k', 'd_rand'), ('d_pre240k', 'eegdm_official'), ('d_rand', 'eegdm_official'),
             ('d_pre015k', 'd_rand'), ('d_pre030k', 'd_rand'), ('d_pre060k', 'd_rand'), ('d_pre120k', 'd_rand'),
             ('d_pre240k_eeg', 'd_rand_eeg'), ('d_pre240k_eog', 'd_rand_eog'), ('d_pre240k_mean', 'd_rand_mean'),
             ('d_pre240k', 'd_pre240k_eeg'), ('d_rand', 'd_rand_eeg')]
    prow, psub = [], []
    for comp, ref in pairs:
        r, s = b29.paired(reg, preds, comp, ref)
        prow += r
        psub += s
    pr = pd.DataFrame(prow)
    pr.to_csv(ana / 'd_w3_paired.csv', index=False, float_format='%.6g')
    pd.DataFrame(psub).to_csv(ana / 'd_w3_paired_by_subject.csv', index=False, float_format='%.6g')
    cols = ['comparison', 'reference', 'budget'] + [f'delta_{k}{x}' for k in b29.KEYS for x in ('', '_lo', '_hi')]
    print(pr[cols].round(4).to_string(index=False), flush=True)

    # ---- N1 分数调整（B 28 的函数；预先固定 label25、seed 0） ----
    adj = []
    for arm in ('d_rand', 'd_pre240k', 'eegdm_official'):
        d = out_root / f'W3_D_{arm}_label25_sseed2024_seed0_v01'
        oof = pd.read_csv(d / 'oof_train.csv')
        val = ev.load_predictions(d / 'predictions_val.csv')
        ev.check_against_reference(val, pd.read_csv(w.VAL_REFERENCE))
        alpha, oof_f1, oof_f1_raw = b28.pick_alpha(oof)
        after = b28.adjusted(val, alpha)
        bm_, am_ = b28.describe(val), b28.describe(after)
        boot = ev.paired_bootstrap_delta(after, val, 1000, 0)
        row = dict(arm=arm, alpha=alpha, oof_n1_f1_raw=oof_f1_raw, oof_n1_f1_adjusted=oof_f1)
        for k in bm_:
            row[f'before_{k}'], row[f'after_{k}'] = bm_[k], am_[k]
        for k in ('n1_f1', 'macro_f1'):
            row[f'delta_{k}'], row[f'delta_{k}_lo'], row[f'delta_{k}_hi'] = boot[k]['point'], boot[k]['lo'], boot[k]['hi']
        adj.append(row)
        print(f'N1 adjust {arm}: alpha={alpha:.3f} dev N1 P/R/F1 {bm_["n1_precision"]:.3f}/{bm_["n1_recall"]:.3f}/'
              f'{bm_["n1_f1"]:.3f} -> {am_["n1_precision"]:.3f}/{am_["n1_recall"]:.3f}/{am_["n1_f1"]:.3f} '
              f'macroF1 {bm_["macro_f1"]:.4f} -> {am_["macro_f1"]:.4f}', flush=True)
    pd.DataFrame(adj).to_csv(ana / 'd_n1_adjust_summary.csv', index=False, float_format='%.6g')
    print(f'total {(time.perf_counter() - t_all) / 60:.1f} min', flush=True)
    print('BPROTOCOL DONE', flush=True)


if __name__ == '__main__':
    main()
