"""Step W5-A1: Temporal context 对照(单 epoch / ±2 直接拼接 / 小 BiGRU)。

协议与 34 号(B 的 W4 探针)完全一致,只把输入从单 epoch 特征换成三种构造:
  1. 预算内训练者按受试者分 K=min(5,人数) 折(fold_seed=0),每折标准化只拟合折内训练者,固定 50 轮;
     OOF Macro-F1 选 E*(并列取最早)。
  2. 全部预算内训练者重新标准化、同 seed 训练 E* 轮,8 名开发集验证者只预测一次。
  3. N1 分数调整在 E* 的 OOF 上按 28 号网格选 α,固定后用于开发集。

特征:D 的 04 号 30k 普通(无 EEG 条件)双模态特征(eeg_std/eog_std [53496,768] float32
+ index.csv,行序与 npy 一致)。每个 epoch 一个 1536 维向量(eeg_std | eog_std)。

窗口:当前 epoch 与前后各 2 个 epoch(同受试者、同记录),只预测中心 epoch。
缺窗与记录边界统一规则:取该方向上最近可用 epoch(可用序列上的位置钳制,边界复制)。

三臂:
  single:Linear(1536,5) —— 与 34 号 --modalities eeg+eog 完全同构,可交叉验证
  cat:   5×1536=7680 维拼接 → Linear(7680,5)
  bgru:  [5,1536] 序列 → 小 BiGRU(hidden 64,双向,1 层)取中心时刻 → Linear(128,5)

用法(仓库根目录):
  python scripts/41_a_w5_temporal_compare.py --arm single --budget label25 --seed 0 \
      --feat-dir experiment3/week5/temporal/features/W4_D_main_m64x12_seed0_v01/step30000_ema_model_pool1s
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from common import RUNS, SPLIT
import w2_common as w

ev = w.ev
CODE_VERSION = 'v01'
D = 1536
WINDOW = 5
OFFSETS = (-2, -1, 0, 1, 2)


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


P27 = _load('probe27', '27_b_w3_linear_probe.py')
P28 = _load('adjust28', '28_b_n1_score_adjust.py')


class BiGRUCenter(torch.nn.Module):
    """小 BiGRU:双向单层,取中心时刻(位置 2)的输出接线性头。只预测中心 epoch。"""

    def __init__(self, d=D, hidden=64):
        super().__init__()
        self.gru = torch.nn.GRU(d, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = torch.nn.Linear(2 * hidden, 5)

    def forward(self, x):  # [B, 5, d]
        out, _ = self.gru(x)
        return self.head(out[:, 2, :])


def load_seq(feat_dir, rows):
    """构建全量窗口矩阵 [N, 5, 1536](rows 顺序即 npy 行序,已按 sample_id 核验)。

    窗口规则:同一受试者、同一记录内取 epoch±2;缺窗/边界统一用可用序列位置钳制
    (该方向上最近可用 epoch)。"""
    base = Path(feat_dir)
    eeg = np.load(base / 'eeg_std.npy', mmap_mode='r')
    eog = np.load(base / 'eog_std.npy', mmap_mode='r')
    if eeg.shape != (len(rows), 768) or eog.shape != (len(rows), 768):
        raise ValueError(f'特征形状异常: eeg {eeg.shape} eog {eog.shape}, 期望 ({len(rows)}, 768)')
    seq = np.empty((len(rows), WINDOW, D), dtype=np.float32)
    rec2pos = defaultdict(list)
    for i, r in enumerate(rows):
        rec2pos[r['record_id']].append(i)
    for rec, pos in rec2pos.items():
        L = len(pos)
        for p_i, i in enumerate(pos):
            idx = [pos[min(L - 1, max(0, p_i + k))] for k in OFFSETS]
            seq[i, :, :768] = eeg[idx]
            seq[i, :, 768:] = eog[idx]
    del eeg, eog
    return seq


def train_bgru(x_tr, y_tr, seed, epochs, x_eval=None):
    """与 P27.train_head 相同的种子顺序、优化器、批大小与逐轮评估。"""
    w.seed_everything(seed, deterministic=True)
    model = BiGRUCenter()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y = torch.as_tensor(y_tr, dtype=torch.long)
    per_epoch, updates = [], 0
    for _ in range(epochs):
        model.train()
        for ids in torch.randperm(len(y)).split(128):
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x_tr[ids]), y[ids]).backward()
            opt.step()
            updates += 1
        if x_eval is not None:
            per_epoch.append(w.predict_probs(model, lambda ids: x_eval[ids], len(x_eval)))
    return model, per_epoch, updates


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--arm', required=True, choices=['single', 'cat', 'bgru'])
    ap.add_argument('--feat-dir', required=True, help='D 04 号特征目录(eeg_std.npy/eog_std.npy/index.csv)')
    ap.add_argument('--budget', required=True, choices=['label10', 'label25'])
    ap.add_argument('--subsets', default=str(SPLIT / 'label_subsets_v2.csv'))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--runs-root', default=str(RUNS))
    ap.add_argument('--group', default='A')
    ap.add_argument('--hidden', type=int, default=64, help='BiGRU hidden(仅 bgru 臂)')
    ap.add_argument('--threads', type=int, default=4, help='torch 线程数(团队口径 4,与 34 号一致)')
    args = ap.parse_args(argv)

    chosen, meta = w.selected_subjects(args.subsets, args.budget)
    if {meta[s]['sampling_seed'] for s in chosen} != {'2024'}:
        sys.exit('错误: 名单 sampling_seed 应为 2024')
    index_rows = w.read_csv_rows(Path(args.feat_dir) / 'index.csv')
    if any(r['split'] == 'test' for r in index_rows):
        print('提示：索引含 test 行，已丢弃且不读取其特征')
    train, val = w.select_rows(index_rows, chosen)
    pos = {r['sample_id']: i for i, r in enumerate(index_rows)}
    if args.arm == 'single':
        # 单 epoch 臂不需要窗口矩阵:直接读中心时刻(即各模态的 epoch 向量)
        eeg = np.load(Path(args.feat_dir) / 'eeg_std.npy', mmap_mode='r')
        eog = np.load(Path(args.feat_dir) / 'eog_std.npy', mmap_mode='r')
        seq = np.concatenate([np.asarray(eeg, dtype=np.float32), np.asarray(eog, dtype=np.float32)], axis=1)
        del eeg, eog
    else:
        seq = load_seq(args.feat_dir, index_rows)

    def arm_matrix(rows):
        idx = np.array([pos[r['sample_id']] for r in rows])
        if args.arm == 'single':
            return seq[idx]
        if args.arm == 'cat':
            return seq[idx].reshape(len(idx), -1)
        return seq[idx]  # [N, 5, D]

    x_tr = arm_matrix(train)
    x_va = arm_matrix(val)
    y_tr = np.array([int(r['y_true']) for r in train])
    subj = np.array([r['subject_id'] for r in train])
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()

    def standardize_pair(xf, xh):
        """窗口臂在展开矩阵 [N*5, D] 上拟合标准化(每个位置共用同一 mean/std),再折回臂形状。"""
        if args.arm == 'single':
            return P27.standardize(xf, xh)
        (mean, std), (tf, th) = P27.standardize(xf.reshape(-1, D), xh.reshape(-1, D))
        shape = (-1, WINDOW, D) if args.arm == 'bgru' else (-1, WINDOW * D)
        return (mean, std), (tf.reshape(shape), th.reshape(shape))

    def head_train(xf, yf, xh):
        if args.arm == 'bgru':
            return train_bgru(xf, yf, args.seed, P27.MAX_EPOCHS, xh)
        return P27.train_head(xf, yf, args.seed, P27.MAX_EPOCHS, xh)

    folds = P27.subject_folds(chosen, min(5, len(chosen)))
    fold_of = np.empty(len(y_tr), dtype=int)
    oof = np.zeros((P27.MAX_EPOCHS, len(y_tr), 5))
    for k, held in enumerate(folds):
        ho = np.isin(subj, held)
        fold_of[ho] = k
        _, (xf, xh) = standardize_pair(x_tr[~ho], x_tr[ho])
        _, per_epoch, _ = head_train(xf, y_tr[~ho], xh)
        oof[:, ho] = np.stack(per_epoch)
    curve = []
    for e in range(P27.MAX_EPOCHS):
        m = ev.compute_metrics(y_tr, oof[e].argmax(1))
        curve.append(dict(epoch=e + 1, oof_macro_f1=m['macro_f1'], oof_n1_f1=m['n1_f1']))
    best = max(curve, key=lambda r: (r['oof_macro_f1'], -r['epoch']))
    e_star = best['epoch']

    (mean, std), (xa, xv) = standardize_pair(x_tr, x_va)
    if args.arm == 'bgru':
        model, _, updates = train_bgru(xa, y_tr, args.seed, e_star)
    else:
        model, _, updates = P27.train_head(xa, y_tr, args.seed, e_star)
    probs = w.predict_probs(model, lambda ids: xv[ids], len(xv))
    elapsed = time.perf_counter() - t0

    run_id = f'W5_{args.group}_tctx_{args.arm}_{args.budget}_sseed2024_seed{args.seed}_{CODE_VERSION}'
    run_dir = w.make_run_dir(run_id, args.runs_root)
    w.write_prediction_table(run_dir / 'predictions_val.csv', val, probs)
    met = ev.evaluate(run_dir / 'predictions_val.csv', reference=w.VAL_REFERENCE)
    pv = ev.load_predictions(run_dir / 'predictions_val.csv')

    o = pd.DataFrame({k: [r[k] for r in train] for k in ('sample_id', 'subject_id', 'record_id', 'epoch_index')})
    o['y_true'], o['fold'], o['y_pred'] = y_tr, fold_of, oof[e_star - 1].argmax(1)
    for j, c in enumerate(w.PROB_COLS):
        o[c] = oof[e_star - 1][:, j]
    o.to_csv(run_dir / 'oof_train.csv', index=False, float_format='%.8g')

    alpha, oof_n1_adj, oof_n1_raw = P28.pick_alpha(o)
    adj = P28.adjusted(pv, alpha)
    adj.to_csv(run_dir / 'predictions_val_n1adj.csv', index=False, float_format='%.8g')
    raw_m, adj_m = P28.describe(pv), P28.describe(adj)
    y1 = (pv['y_true'] == 1).to_numpy()
    met.update(run_id=run_id, selected_epoch=e_star, oof_macro_f1_at_selected=best['oof_macro_f1'],
               n1_ap=float(average_precision_score(y1, pv['p_N1'])), n1_auroc=float(roc_auc_score(y1, pv['p_N1'])),
               n1_adjust=dict(alpha=alpha, oof_n1_f1_raw=oof_n1_raw, oof_n1_f1_adjusted=oof_n1_adj,
                              n1_f1_raw=raw_m['n1_f1'], n1_f1_adjusted=adj_m['n1_f1'],
                              n1_precision_adjusted=adj_m['n1_precision'], n1_recall_adjusted=adj_m['n1_recall'],
                              macro_f1_raw=raw_m['macro_f1'], macro_f1_adjusted=adj_m['macro_f1'],
                              selection='alpha 只在训练侧 OOF(E* 轮)上选,开发集只应用一次'))
    (run_dir / 'metrics.json').write_text(json.dumps(met, indent=2, ensure_ascii=False), encoding='utf-8')
    ev.per_subject_table(pv).to_csv(run_dir / 'by_subject.csv', index=False)
    pd.DataFrame(curve).to_csv(run_dir / 'inner_cv.csv', index=False)
    torch.save(dict(state=model.state_dict(), epoch=e_star, mean=np.asarray(mean), std=np.asarray(std)),
               run_dir / 'final.ckpt')
    np.savez(run_dir / 'normalization.npz', mean=np.asarray(mean), std=np.asarray(std))
    pd.DataFrame({'subject_id': chosen}).to_csv(run_dir / 'labeled_subjects.csv', index=False)

    head_desc = dict(single='Linear(1536,5)', cat='Linear(7680,5)',
                     bgru=f'BiGRU({args.hidden},bidir,1层)取中心时刻 + Linear({2*args.hidden},5)')[args.arm]
    feat_dir = Path(args.feat_dir)
    info = json.loads((feat_dir / 'features_info.json').read_text(encoding='utf-8'))
    config = dict(experiment='W5 temporal context 读出(冻结表征)', group=args.group, arm=f'tctx_{args.arm}',
                  run_id=run_id, nominal_budget=args.budget, labeled_subjects=len(chosen), sampling_seed=2024,
                  downstream_seed=args.seed, subset_sha256=w.sha256_file(args.subsets),
                  train_samples=len(y_tr), val_samples=len(val),
                  feature_dims=dict(epoch=1536, window=WINDOW), head=head_desc,
                  optimizer='AdamW lr=1e-3 wd=1e-4', batch_size=128,
                  standardization='fit on training-side subjects only (fold-train in inner CV, all budget subjects for final);'
                                  ' windowed arms fit on unfolded [N*5,1536] (same mean/std per position)',
                  selection=f'inner subject-wise CV on budget subjects, K={len(folds)}, fold_seed={P27.FOLD_SEED}, '
                            f'fixed {P27.MAX_EPOCHS} epochs per fold, epoch = argmax pooled OOF macro-F1 (ties -> earliest)',
                  n1_adjust=f'alpha grid logspace(-1,1.5,801) on pooled OOF at E*; alpha={alpha:.6g}',
                  window=dict(offsets=list(OFFSETS), rule='same subject, same record; missing/boundary -> nearest '
                              'available epoch (position clamping on available sequence)', predict='center epoch only'),
                  folds=folds, selected_epoch=e_star, dev_used_for='final evaluation only',
                  sources={f: dict(path=str(feat_dir / f), sha256=w.sha256_file(feat_dir / f))
                           for f in ('eeg_std.npy', 'eog_std.npy', 'index.csv', 'features_info.json')},
                  backbone=dict(checkpoint=info.get('checkpoint'), checkpoint_sha256=info.get('checkpoint_sha256'),
                                step=info.get('step'), weights=info.get('weights'), switches=info.get('switches')),
                  torch=str(torch.__version__), device='cpu', test_used=False, code_commit=w.code_version(),
                  evaluate_version=ev.EVALUATE_VERSION, val_reference_sha256=w.sha256_file(w.VAL_REFERENCE),
                  predictions_sha256=w.sha256_file(run_dir / 'predictions_val.csv'),
                  final_updates=updates, elapsed_seconds=round(elapsed, 2))
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(f"{run_id}: E*={e_star}  dev macro-F1 {met['macro_f1']:.4f}  N1 AP {met['n1_ap']:.3f}  "
          f"N1 F1 {raw_m['n1_f1']:.3f} -> {adj_m['n1_f1']:.3f} (alpha {alpha:.2f}, macro-F1 {adj_m['macro_f1']:.4f})  {elapsed:.0f}s")


if __name__ == '__main__':
    main()
