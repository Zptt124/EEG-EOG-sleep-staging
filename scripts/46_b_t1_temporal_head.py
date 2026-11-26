"""新一轮第 1 周 · 时序线下游：A0（随机初始化 BiGRU）/ A1、A2（加载预训练 BiGRU）在 5 人名单上微调并评分。

设置（2026-11-26 已与 A 对齐）：
  - 训练 = 名单的 5 位标注者全部段；评估 = 固定 dev 8 人（主评）、该名单的补充 9 人，另报合并 17 人。
  - 输入 384 维特征按真实邻接组 ±2 窗口（不跨夜、不跨缺口，缺位置 0 + mask），见 t1_temporal_common。
  - 标准化：--norm-stats 给出 32 人预训练集的 mean/std（npz：mean, std）；三个条件共用。
    --norm-fit label 只用于试跑（在 5 位标注者上拟合），config 里会记为 dry-run 口径。
  - BiGRU(385→64，双向，1 层) + Linear(128,5)，整体训练；AdamW lr 1e-3 wd 1e-4，batch 128 保留尾批，固定 30 轮。
  - 种子：GRU 初始化、分类头初始化、batch 打乱各用独立随机流（t1_temporal_common.init_under / batch_order），
    同一 seed 下三个条件的分类头初始化与每轮打乱顺序逐位相同；A0 的 GRU 初始化也随 seed；
    --init 给出预训练权重时只覆盖 GRU 部分。
  - 名单 1 = 11,936 段 → 30 轮 2,820 步；名单 2 = 10,586 段 → 2,490 步。取最后一轮，不做五折选轮。
输出（--out/<run>/）：predictions_dev8.csv、predictions_eval9.csv（11 列 + roster、eval_set）、train_loss.csv、
  config.json、metrics.json（--dry-run 时不计算任何评估指标，只核对形状与覆盖）。

用法：
  python scripts/46_b_t1_temporal_head.py --cond A0 --roster 2 --seed 0 \
      --feat-dir <特征目录> --norm-stats <32 人标准化 npz> --out runs
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_temporal_common as T  # noqa: E402
import w2_common as w  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
PROB = w.PROB_COLS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cond', required=True, choices=['A0', 'A1', 'A2'])
    ap.add_argument('--roster', required=True, choices=['1', '2'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--feat-dir', required=True)
    ap.add_argument('--rosters', default=str(BASE / 'manifests' / 'temporal_rosters_t1_v01.csv'))
    ap.add_argument('--segments', default=str(BASE / 'manifests' / 'contiguous_segments_t1_v01.csv'))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--norm-stats', help='32 人预训练集的标准化参数 npz（mean, std）')
    g.add_argument('--norm-fit', choices=['label'], help='仅试跑：在 5 位标注者上拟合标准化')
    ap.add_argument('--init', help='A1/A2：预训练权重（state_dict，含 gru.*）')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--group', default='B')
    ap.add_argument('--tag', default='v01')
    ap.add_argument('--dry-run', action='store_true', help='不计算评估指标，只核对流程')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if (args.cond == 'A0') != (args.init is None):
        sys.exit('A0 不加载权重；A1/A2 必须给 --init')

    roster = [r for r in T.read_csv(args.rosters) if r['roster'] == args.roster]
    sets = {k: sorted(r['subject_id'] for r in roster if r['role'] == k) for k in ('label', 'eval9', 'dev8')}
    assert len(sets['label']) == 5 and len(sets['eval9']) == 9 and len(sets['dev8']) == 8

    rows, x, finfo = T.load_features(args.feat_dir)
    if any(r.get('split') == 'test' for r in rows):
        sys.exit('特征索引含测试集行')
    subj = np.array([r['subject_id'] for r in rows])
    segs = T.load_segments(args.segments)
    nbr = T.neighbor_table(rows, segs)
    idx = {k: np.flatnonzero(np.isin(subj, v)) for k, v in sets.items()}
    for k, v in sets.items():
        if set(subj[idx[k]]) != set(v):
            sys.exit(f'{k}: 特征里缺受试者 {sorted(set(v) - set(subj[idx[k]]))}')

    if args.norm_stats:
        z = np.load(args.norm_stats)
        mean, std = z['mean'].astype(np.float32), z['std'].astype(np.float32)
        norm_src = dict(mode='pretrain32', path=args.norm_stats, sha256=T.sha256(args.norm_stats))
    else:
        mean, std = T.fit_standardizer(x[idx['label']])
        norm_src = dict(mode='label5 (DRY-RUN ONLY)')
    xs = (x - mean) / std

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model = T.BiGRUCenter(args.seed)        # GRU / 分类头各用独立随机流（A0 的 GRU 初始化也随 head seed）
    if args.init:
        sd = torch.load(args.init, map_location='cpu')
        sd = {k: v for k, v in sd.items() if k.startswith('gru.')}
        missing = model.load_state_dict(sd, strict=False)
        if any(k.startswith('gru.') for k in missing.missing_keys):
            sys.exit(f'预训练权重缺 GRU 参数: {missing.missing_keys}')
    x_tr = torch.from_numpy(T.windows(xs, nbr, idx['label']))
    y_tr = torch.as_tensor([int(rows[i]['y_true']) for i in idx['label']], dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    t0, updates, hist = time.perf_counter(), 0, []
    gen = T.batch_order(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for b in torch.randperm(len(y_tr), generator=gen).split(128):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            updates += 1
        hist.append(dict(epoch=ep, train_loss=tot / len(y_tr)))

    run = f'T1_{args.group}_{args.cond}_r{args.roster}_seed{args.seed}_{args.tag}'
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out / 'train_loss.csv', index=False, float_format='%.6f', lineterminator='\n')
    preds = {}
    for k in ('dev8', 'eval9'):
        xe = T.windows(xs, nbr, idx[k])
        p = w.predict_probs(model, lambda ids: torch.from_numpy(xe[ids.numpy()]), len(xe))
        df = pd.DataFrame({c: [rows[i][c] for i in idx[k]] for c in
                           ('sample_id', 'subject_id', 'record_id', 'epoch_index', 'y_true')})
        df['y_pred'] = p.argmax(1)
        for j, c in enumerate(PROB):
            df[c] = p[:, j]
        df['roster'], df['eval_set'] = args.roster, k
        df.to_csv(out / f'predictions_{k}.csv', index=False, float_format='%.8g', lineterminator='\n')
        preds[k] = df
    nb_tr = nbr[idx['label']]
    cfg = dict(run_id=run, cond=args.cond, roster=args.roster, seed=args.seed, sets=sets,
               features=finfo, norm=norm_src, segments=dict(path=args.segments, sha256=T.sha256(args.segments)),
               rosters=dict(path=args.rosters, sha256=T.sha256(args.rosters)),
               init=(dict(path=args.init, sha256=T.sha256(args.init)) if args.init else None),
               model='BiGRU(385->64, bidir, 1 layer) center -> Linear(128,5); input = 384 feat + mask flag',
               window='center ±2; same record, epoch_index±k within contiguous segment; missing -> 0 + mask=1',
               optimizer='AdamW lr=1e-3 wd=1e-4', batch_size=128, epochs=args.epochs, updates=updates,
               train_epochs=len(y_tr), train_missing_positions=int((nb_tr < 0).sum()),
               threads=args.threads, torch=torch.__version__, code_commit=w.code_version(),
               elapsed_seconds=round(time.perf_counter() - t0, 1), dry_run=args.dry_run, test_used=False)
    T.dump_json(out / 'config.json', cfg)

    if args.dry_run:
        print(f'{run}: DRY-RUN  train {len(y_tr)} epochs, {updates} updates, final loss {hist[-1]["train_loss"]:.4f}; '
              f'dev8 {len(preds["dev8"])} / eval9 {len(preds["eval9"])} rows written; metrics not computed')
        return

    ev = w.ev
    met = {}
    sn1 = T.short_n1_runs(rows, segs)
    for k, df in (('dev8', preds['dev8']), ('eval9', preds['eval9']),
                  ('eval17', pd.concat([preds['dev8'], preds['eval9']]))):
        ii = np.concatenate([idx['dev8'], idx['eval9']]) if k == 'eval17' else idx[k]
        m = ev.compute_metrics(df['y_true'].to_numpy(int), df['y_pred'].to_numpy(int))
        from sklearn.metrics import average_precision_score
        m['n1_ap'] = float(average_precision_score(df['y_true'].to_numpy(int) == 1, df['p_N1'].to_numpy()))
        m.update(T.temporal_metrics(df['y_true'].to_numpy(int), df['y_pred'].to_numpy(int), sn1[ii]))
        met[k] = {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv) for kk, vv in m.items()}
    T.dump_json(out / 'metrics.json', met)
    d = met['dev8']
    print(f"{run}: dev8 macro-F1 {d['macro_f1']:.4f} N1 AP {d['n1_ap']:.4f} short-N1 recall {d['short_n1_recall']:.3f}")


if __name__ == '__main__':
    main()
