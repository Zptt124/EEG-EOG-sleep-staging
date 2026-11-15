"""第 6 步：冻结特征上的 linear probe（CPU，每个 head 约 1–3 分钟）。

保底配方（B 的探针接口到之前使用；规则在看结果前固定，两种权重完全相同）：
  - 输入：EEG 与 EOG 的 pooled 特征拼接（默认 std pooling），每个模态 n_layer*d_model 维
  - 训练侧内部留出：14 个有标签受试者按固定 seed 抽 3 人做内部验证（早停用），其余 11 人训练
  - 标准化：只用 11 个内部训练者的样本拟合 mean/std（训练侧）
  - head：Linear(D,5)，AdamW lr 1e-3 wd 1e-4，batch 128，最多 50 轮，内部验证 Macro-F1 严格变好才更新，patience 10
  - 开发集 8 人只用于最终评估，不参与任何选择
输出：D: runs/<run_name>/probe/<tag>_<pooling>_head<k>/ 下的 config.json、history.csv、
     predictions_val.csv（11 列，全组格式）、metrics.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import w3d_common as C
from w3d_metrics import compute_metrics

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
INNER_SEED = 0
N_INNER_VAL = 3


def inner_split(labeled_subjects):
    rng = np.random.default_rng(INNER_SEED)
    perm = rng.permutation(sorted(labeled_subjects))
    return sorted(perm[N_INNER_VAL:]), sorted(perm[:N_INNER_VAL])


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=list(range(5)), average='macro', zero_division=0)


def train_head(xtr, ytr, xva, yva, seed, max_epochs=50, patience=10, lr=1e-3, wd=1e-4, bs=128):
    torch.manual_seed(seed)
    head = torch.nn.Linear(xtr.shape[1], 5)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr).long()
    xva_t = torch.from_numpy(xva)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, max_epochs + 1):
        head.train()
        perm = torch.randperm(len(xtr_t), generator=g)
        tot = 0.0
        for b in range(0, len(perm), bs):
            i = perm[b:b + bs]
            loss = torch.nn.functional.cross_entropy(head(xtr_t[i]), ytr_t[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(i)
        head.eval()
        with torch.no_grad():
            pv = head(xva_t).argmax(1).numpy()
        f = macro_f1(yva, pv)
        hist.append(dict(epoch=ep, train_loss=tot / len(perm), inner_val_macro_f1=f))
        if f > best:
            best, bad = f, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, hist, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=True)
    p.add_argument('--tags', required=True, help='e.g. step0_ema_model,step60000_ema_model')
    p.add_argument('--head-seeds', default='0,1,2')
    p.add_argument('--pooling', choices=['std', 'mean'], default='std')
    p.add_argument('--modalities', default='eeg,eog')
    args = p.parse_args()

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    labeled = C.label25_subjects()
    inner_tr, inner_va = inner_split(labeled)
    print(f'inner train ({len(inner_tr)}): {inner_tr}\ninner val ({len(inner_va)}): {inner_va}', flush=True)
    mods = args.modalities.split(',')

    for tag in args.tags.split(','):
        fdir = C.BIG / 'features' / args.run_name / tag
        if not (fdir / 'DONE').exists():
            C.fail(f'Features not found: {fdir}')
        index = C.read_csv(fdir / 'index.csv')
        X = np.concatenate([np.load(fdir / f'{m}_{args.pooling}.npy') for m in mods], axis=1).astype(np.float32)
        subj = np.array([r['subject_id'] for r in index])
        role = np.array([r['role'] for r in index])
        y = np.array([int(r['y_true']) for r in index])
        m_tr = (role == 'labeled_train') & np.isin(subj, inner_tr)
        m_iv = (role == 'labeled_train') & np.isin(subj, inner_va)
        m_val = role == 'val'
        if not (m_tr.any() and m_iv.any() and m_val.sum() > 0):
            C.fail('Empty partition')
        if set(subj[role == 'labeled_train']) != set(labeled):
            C.fail('Labeled subjects in features do not match label25 list')
        mu = X[m_tr].mean(0)
        sd = X[m_tr].std(0) + 1e-6
        Z = (X - mu) / sd
        for k in [int(s) for s in args.head_seeds.split(',')]:
            name = f'{tag}_{args.pooling}_{"".join(mods)}_head{k}'
            out = C.EXP_D / 'runs' / args.run_name / 'probe' / name
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            head, hist, best = train_head(Z[m_tr], y[m_tr], Z[m_iv], y[m_iv], seed=k)
            with torch.no_grad():
                prob = torch.softmax(head(torch.from_numpy(Z[m_val])), 1).numpy()
            pred = prob.argmax(1)
            rows = []
            for j, i in enumerate(np.where(m_val)[0]):
                r = index[i]
                rows.append(dict(sample_id=r['sample_id'], subject_id=r['subject_id'], record_id=r['record_id'],
                                 epoch_index=r['epoch_index'], y_true=int(y[i]), y_pred=int(pred[j]),
                                 **{f'p_{s}': f'{prob[j, c]:.6f}' for c, s in enumerate(STAGES)}))
            C.write_csv(out / 'predictions_val.csv', rows)
            C.write_csv(out / 'history.csv', hist)
            met = compute_metrics(y[m_val], pred, prob, subj[m_val])
            C.write_json(out / 'metrics.json', met)
            C.write_json(out / 'config.json', dict(
                run_name=args.run_name, feature_tag=tag, pooling=args.pooling, modalities=mods,
                feature_dim=int(X.shape[1]), head='Linear(D,5)', optimizer='AdamW lr 1e-3 wd 1e-4', batch=128,
                max_epochs=50, early_stopping='inner-val macro-F1 strictly better, patience 10',
                inner_train_subjects=inner_tr, inner_val_subjects=inner_va, inner_seed=INNER_SEED,
                standardization='mean/std on inner-train samples only', head_seed=k,
                best_inner_val_macro_f1=best, epochs_run=len(hist), seconds=time.perf_counter() - t0,
                n_inner_train=int(m_tr.sum()), n_inner_val=int(m_iv.sum()), n_val=int(m_val.sum()),
                label_list='label_subsets_v2.csv sampling_seed 2024 label25 (14 subjects)'))
            print(f'{name}: epochs={len(hist)} inner_val={best:.4f} | val macroF1={met["macro_f1"]:.4f} '
                  f'N1-F1={met["per_class_f1"]["N1"]:.4f} ({time.perf_counter() - t0:.0f}s)', flush=True)
    print('PROBE DONE', flush=True)


if __name__ == '__main__':
    main()
