"""新一轮第 2 周 · B 组织预留 8 人统一评分 + 两张配对表 + 选题决定(第 9–10 天)。

表一(10 名单):加人 − 加深,在 W4 预留 8 人上复验第一周结论。
表二(9 名单):换夜 − 加深 与 加人 − 换夜(05 无另一夜)。
口径:
  - 特征 = D 共享缓存(384 维,role=heldout_val 的 8 人 18,897 段);每头用各自的
    normalization.npz 标准化;head = Linear(384,5),softmax 五类概率。
  - Macro-F1 五类等权(官方 evaluate v1);名单内同一样本三条件配对,跨名单等权汇总;
    同一受试者跨名单共用 Poisson(1) bootstrap 权重(1000 次,rng 0);
    逐人剔除(LOSO)= 从所有名单去掉该受试者后的平均差。
  - B 锁定的判定界值(第 6 天建议默认,本表冻结):实际意义 1 pp;等效带 ±1 pp;
    等效判定要求 95% 区间整体落入带内。

用法:
  python scripts/51_b_w2_heldout.py
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import BASE
import w2_common as w

ev = w.ev
CACHE = BASE / 'cache/w1_temporal/w6_d_main30k_l10_12_v01/W6_D_main30k_l10_12_v01'
OLD = {
    '01': BASE / 'cache/w2_budget/c/x_C_week1_heads_for_B_20261127/01/paired',
    '02': BASE / 'cache/w2_budget/c/x_C_week1_heads_for_B_20261127/02/paired',
    '03': BASE / 'cache/w2_budget/c/x_C_week1_heads_for_B_20261127/03/paired',
    '04': BASE / 'cache/w2_budget/c/x_C_week1_heads_for_B_20261127/04/paired',
    '05': BASE / 'cache/w2_budget/c/x_C_week1_heads_for_B_20261127/05/paired',
    '06': BASE / 'cache/w1_budget/d06_10/x/d_budget_week1_results_v01/06/paired',
    '07': BASE / 'cache/w1_budget/d06_10/x/d_budget_week1_results_v01/07/paired',
    '08': BASE / 'cache/w1_budget/d06_10/x/d_budget_week1_results_v01/08/paired',
    '09': BASE / 'cache/w1_budget/d06_10/x/d_budget_week1_results_v01/09/paired',
    '10': BASE / 'cache/w1_budget/d06_10/x/d_budget_week1_results_v01/10/paired',
}
NEW = {
    '01': BASE / 'runs/01/other_night',
    '07': BASE / 'runs/07/other_night',
    '09': BASE / 'runs/09/other_night',
    '02': BASE / 'cache/w2_budget/b/x_b_budget_week2_other_night_v01/b_budget_week2_other_night_v01/02/other_night',
    '08': BASE / 'cache/w2_budget/b/x_b_budget_week2_other_night_v01/b_budget_week2_other_night_v01/08/other_night',
    '03': BASE / 'cache/w2_budget/c/x_C_week2_matching_and_results_20261127/results/03/other_night',
    '10': BASE / 'cache/w2_budget/c/x_C_week2_matching_and_results_20261127/results/10/other_night',
    '04': BASE / 'cache/w2_budget/d/x_d_budget_week2_other_night_v01/d_budget_week2_other_night_v01/04/other_night',
    '06': BASE / 'cache/w2_budget/d/x_d_budget_week2_other_night_v01/d_budget_week2_other_night_v01/06/other_night',
}
N_BOOT = 1000
MEANINGFUL_PP = 1.0
EQUIV_BAND_PP = 1.0


def load_heldout():
    rows = list(csv.DictReader(open(CACHE / 'index.csv', encoding='utf-8')))
    eeg = np.load(CACHE / 'eeg_std.npy', mmap_mode='r')
    eog = np.load(CACHE / 'eog_std.npy', mmap_mode='r')
    idx = [i for i, r in enumerate(rows) if r['role'] == 'heldout_val']
    x = np.concatenate([np.asarray(eeg[idx], dtype=np.float32), np.asarray(eog[idx], dtype=np.float32)], axis=1)
    subj = np.array([rows[i]['subject_id'] for i in idx])
    return [rows[i] for i in idx], x, subj


def predict_head(head_dir, x):
    sd = torch.load(head_dir / 'head.pt', map_location='cpu', weights_only=True)
    model = torch.nn.Linear(sd['weight'].shape[1], sd['weight'].shape[0])
    model.load_state_dict(sd)
    model.eval()
    z = np.load(head_dir / 'normalization.npz')
    xs = torch.from_numpy((x - z['mean'].astype(np.float32)) / z['std'].astype(np.float32))
    probs = []
    with torch.inference_mode():
        for b in xs.split(2048):
            probs.append(model(b).softmax(1))
    return torch.cat(probs).numpy()


def macro_f1_w(y, pred, w):
    f1s = []
    for c in range(5):
        tp = w[(y == c) & (pred == c)].sum()
        fp = w[(y != c) & (pred == c)].sum()
        fn = w[(y == c) & (pred != c)].sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(BASE / 'reports' / 'w2_budget' / 'b_heldout'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, x, subj = load_heldout()
    y = np.array([int(r['y_true']) for r in rows])
    print(f'heldout: {len(rows)} 段, {len(set(subj))} 人', flush=True)

    # 回放自检:01 other_night 头在 dev8 上的重放 vs 训练器交付预测
    dev_pred = pd.read_csv(BASE / 'runs/01/other_night/predictions_dev8.csv')
    idx_rows = list(csv.DictReader(open(CACHE / 'index.csv', encoding='utf-8')))
    eeg = np.load(CACHE / 'eeg_std.npy', mmap_mode='r')
    eog = np.load(CACHE / 'eog_std.npy', mmap_mode='r')
    dev_ids = set(dev_pred['sample_id'])
    dv_idx = [i for i, r in enumerate(idx_rows) if r['sample_id'] in dev_ids]
    xdv = np.concatenate([np.asarray(eeg[dv_idx], dtype=np.float32), np.asarray(eog[dv_idx], dtype=np.float32)], axis=1)
    by_id = {s: i for i, s in enumerate(dev_pred['sample_id'])}
    order = [by_id[idx_rows[i]['sample_id']] for i in dv_idx]
    p = predict_head(NEW['01'], xdv)
    ref = dev_pred[w.PROB_COLS].to_numpy()[order]
    diff = np.abs(p - ref).max()
    print(f'回放自检(01 other_night dev8): 概率最大差 {diff:.3e}', flush=True)
    assert diff < 1e-5, '回放不一致,停止'

    # 统一评分:29 头 × heldout8;保存逐段预测
    records = []
    for r in [f'{i:02d}' for i in range(1, 11)]:
        conds = {}
        for c in ('add_people', 'deepen'):
            probs = predict_head(OLD[r] / c, x)
            conds[c] = probs
            pred = pd.DataFrame([dict(sample_id=rr['sample_id'], subject_id=rr['subject_id'],
                                      record_id=rr['record_id'], epoch_index=rr['epoch_index'],
                                      y_true=int(rr['y_true']), y_pred=int(probs[i].argmax()),
                                      **{k: float(probs[i][j]) for j, k in enumerate(w.PROB_COLS)})
                                 for i, rr in enumerate(rows)])
            pred.to_csv(out / f'heldout8_{r}_{c}.csv', index=False, float_format='%.8g')
        if r in NEW:
            probs = predict_head(NEW[r], x)
            conds['other_night'] = probs
            pred = pd.DataFrame([dict(sample_id=rr['sample_id'], subject_id=rr['subject_id'],
                                      record_id=rr['record_id'], epoch_index=rr['epoch_index'],
                                      y_true=int(rr['y_true']), y_pred=int(probs[i].argmax()),
                                      **{k: float(probs[i][j]) for j, k in enumerate(w.PROB_COLS)})
                                 for i, rr in enumerate(rows)])
            pred.to_csv(out / f'heldout8_{r}_other_night.csv', index=False, float_format='%.8g')
        records.append(dict(roster=r, y=y, subj=subj, **conds))

    subj_uniq = sorted(set(subj))
    rng = np.random.default_rng(0)

    def per_roster_delta(rec, a, b):
        return (macro_f1_w(rec['y'], rec[a].argmax(1), np.ones(len(rec['y'])))
                - macro_f1_w(rec['y'], rec[b].argmax(1), np.ones(len(rec['y'])))) * 100

    def analyze(contrasts, recs):
        # contrasts: [(name, a, b)]
        per = {name: [per_roster_delta(rec, a, b) for rec in recs] for name, a, b in contrasts}
        results = {}
        for name, a, b in contrasts:
            vals = np.array(per[name])
            mean = float(vals.mean())
            boot = []
            for _ in range(N_BOOT):
                wmap = dict(zip(subj_uniq, rng.poisson(1.0, size=len(subj_uniq))))
                draws = []
                for rec in recs:
                    sw = np.array([wmap[s] for s in rec['subj']])
                    if sw.sum() == 0:
                        continue
                    draws.append((macro_f1_w(rec['y'], rec[a].argmax(1), sw)
                                  - macro_f1_w(rec['y'], rec[b].argmax(1), sw)) * 100)
                boot.append(np.mean(draws) if draws else np.nan)
            boot = np.array(boot)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            loso = []
            for s in subj_uniq:
                vals_s = []
                for rec, v in zip(recs, per[name]):
                    keep = rec['subj'] != s
                    if keep.any():
                        vals_s.append((macro_f1_w(rec['y'][keep], rec[a][keep].argmax(1), np.ones(keep.sum()))
                                       - macro_f1_w(rec['y'][keep], rec[b][keep].argmax(1), np.ones(keep.sum()))) * 100)
                loso.append(np.mean(vals_s))
            results[name] = dict(mean_pp=mean, ci_pp=[lo, hi], n_positive=int((vals > 0).sum()),
                                 n_rosters=len(vals), per_roster=[round(v, 3) for v in vals],
                                 loso_min=float(np.min(loso)), loso_max=float(np.max(loso)),
                                 loso_sign_stable=bool(np.min(loso) > 0 or np.max(loso) < 0))
        return results

    t1 = analyze([('add-deepen', 'add_people', 'deepen')], records)
    recs9 = [r for r in records if 'other_night' in r]
    t2 = analyze([('other-deepen', 'other_night', 'deepen'),
                  ('add-other', 'add_people', 'other_night')], recs9)

    verdict = {}
    for name, s in {**t1, **t2}.items():
        first = name.split('-')[0]
        in_equiv = abs(s['ci_pp'][0]) <= EQUIV_BAND_PP and abs(s['ci_pp'][1]) <= EQUIV_BAND_PP
        above = s['ci_pp'][0] > 0 and s['mean_pp'] >= MEANINGFUL_PP
        below = s['ci_pp'][1] < 0 and s['mean_pp'] <= -MEANINGFUL_PP
        verdict[name] = f'{first} 占优(下限>0)' if above else f'{first} 占优(上限<0)' if below else \
                        ('等效(区间在 ±1pp 带内)' if in_equiv else '不确定')
    summary = dict(table1=t1, table2=t2, verdict=verdict,
                   meaningful_pp=MEANINGFUL_PP, equiv_band_pp=EQUIV_BAND_PP,
                   bootstrap='Poisson(1) 受试者权重,同一受试者跨名单共用,1000 次,rng 0',
                   heldout=dict(n_segments=len(rows), n_subjects=len(subj_uniq)),
                   notes='表一 10 名单;表二 9 名单(05 无另一夜)')
    (out / 'heldout_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
