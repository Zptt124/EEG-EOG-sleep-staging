"""新一轮第 1 周 · 时序线预训练：遮挡中心段、用 ±2 邻居预测中心特征（A1 真实邻接 / A2 段内置换）。

默认设置（B 草案，已发给 A 对齐）：
  - 预训练集 = temporal_rosters_t1_v01 的 pretrain32（训练池 54 − 原 14 − W4 held-out 8），只用特征，不读标签。
  - 标准化只在 32 人的 81,731 个真实段上拟合一次并冻结，写到 norm_stats.npz（mean, std）；
    A1 目标、A0/A1/A2 微调、8/9/17 人评分全部复用，不重新拟合。
  - 窗口见 t1_temporal_common：A1 = 真实邻接；A2 = --permute-seed 0，每段连续记录内置换一次再取 ±2，
    置换后的邻居表保存为 neighbors.npy（邻居核查记录）。
  - 中心位置输入置 0、mask=1；目标 = 标准化后的 384 维中心特征；损失 MSE。
  - A2 只在预训练时置换；中心顺序与 A1 相同，只换邻居索引。微调和评分一律用真实邻接。
  - 模型 BiGRUCenter 的 GRU + decoder Linear(128, 384)（A1/A2 同结构同初始化）；下游只取 gru.*。
  - AdamW lr 1e-3 wd 1e-4，batch 128 保留尾批，固定 --epochs 20（639 步/轮，共 12,780 步），seed 0；
    GRU / decoder 初始化与 batch 打乱各用独立随机流，A1/A2 的 batch 组成完全相同。
  - A1、A2 各预训练一份，两份名单 × 3 个 head seed 复用。
输出（--out/<run>/）：gru_state.pt（去掉 body. 前缀的 state_dict，下游只取 gru.*）、norm_stats.npz、neighbors.npy、train_loss.csv、config.json

用法：
  python scripts/47_b_t1_temporal_pretrain.py --cond A2 --feat-dir <含 32 人的特征目录> --out runs
  （--subjects-role 只在冒烟测试时改，例如用 label 代替 pretrain32）
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


class Pretrainer(torch.nn.Module):
    """BiGRU + decoder Linear(128→384)；A1/A2 同结构、同初始化，微调时丢弃 decoder 和预训练用的分类头。"""

    def __init__(self, seed):
        super().__init__()
        self.body = T.BiGRUCenter(seed)
        self.pred = T.init_under(seed, 'decoder', lambda: torch.nn.Linear(128, T.D_FEAT))

    def forward(self, x):
        return self.pred(self.body.encode(x)[:, T.CENTER, :])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cond', required=True, choices=['A1', 'A2'])
    ap.add_argument('--feat-dir', required=True)
    ap.add_argument('--rosters', default=str(BASE / 'manifests' / 'temporal_rosters_t1_v01.csv'))
    ap.add_argument('--segments', default=str(BASE / 'manifests' / 'contiguous_segments_t1_v01.csv'))
    ap.add_argument('--subjects-role', default='pretrain32', help='冒烟测试可改为 label 等；正式只用 pretrain32')
    ap.add_argument('--permute-seed', type=int, default=0, help='A2 段内置换种子')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--group', default='B')
    ap.add_argument('--tag', default='v01')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    subs = sorted({r['subject_id'] for r in T.read_csv(args.rosters) if r['role'] == args.subjects_role})
    if not subs:
        sys.exit(f'名单里没有 role={args.subjects_role}')
    rows, x, finfo = T.load_features(args.feat_dir)
    subj = np.array([r['subject_id'] for r in rows])
    if missing := set(subs) - set(subj):
        sys.exit(f'特征里缺 {len(missing)} 名预训练受试者，例如 {sorted(missing)[:3]}')
    keep = np.flatnonzero(np.isin(subj, subs))
    rows = [rows[i] for i in keep]                 # 只保留预训练受试者，邻居也只在其中取
    x = x[keep]
    mean, std = T.fit_standardizer(x)
    xs = (x - mean) / std
    segs = T.load_segments(args.segments)
    nbr = T.neighbor_table(rows, segs, permute_seed=(args.permute_seed if args.cond == 'A2' else None))

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model = Pretrainer(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xw = T.windows(xs, nbr, np.arange(len(rows)))
    xw[:, T.CENTER, :T.D_FEAT] = 0.0               # 遮挡中心
    xw[:, T.CENTER, T.D_FEAT] = 1.0
    xw = torch.from_numpy(xw)
    target = torch.from_numpy(xs)
    t0, updates, hist = time.perf_counter(), 0, []
    gen = T.batch_order(args.seed)    # 中心顺序固定为规范行序，A1/A2 的 batch 组成完全相同；A2 只换邻居索引
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for b in torch.randperm(len(target), generator=gen).split(128):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xw[b]), target[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            updates += 1
        hist.append(dict(epoch=ep, train_mse=tot / len(target)))
        print(f'epoch {ep}: mse {tot / len(target):.4f}', flush=True)

    run = f'T1_{args.group}_{args.cond}_pretrain_seed{args.seed}_{args.tag}'
    if args.subjects_role != 'pretrain32':
        run += f'_SMOKE_{args.subjects_role}'
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k.removeprefix('body.'): v for k, v in model.state_dict().items()}, out / 'gru_state.pt')
    np.savez(out / 'norm_stats.npz', mean=mean, std=std)
    np.save(out / 'neighbors.npy', nbr)
    pd.DataFrame(hist).to_csv(out / 'train_loss.csv', index=False, float_format='%.6f', lineterminator='\n')
    pd.DataFrame({'row': np.arange(len(rows)), 'sample_id': [r['sample_id'] for r in rows]}).to_csv(
        out / 'neighbors_rows.csv', index=False, lineterminator='\n')
    nb = nbr[:, [0, 1, 3, 4]]
    ep_ = np.array([int(r['epoch_index']) for r in rows])
    true_adj = float(np.mean(ep_[nbr[nbr[:, 3] >= 0, 3]] == ep_[nbr[:, 3] >= 0] + 1))
    T.dump_json(out / 'config.json', dict(
        run_id=run, cond=args.cond, subjects_role=args.subjects_role, subjects=subs, n_epochs_data=len(rows),
        features=finfo, segments=dict(path=args.segments, sha256=T.sha256(args.segments)),
        rosters=dict(path=args.rosters, sha256=T.sha256(args.rosters)),
        permute_seed=(args.permute_seed if args.cond == 'A2' else None),
        neighbor_check=dict(missing_positions=int((nb < 0).sum()), plus1_truly_adjacent=true_adj),
        objective='mask center (0 + flag), MSE to standardized 384-d center feature',
        model='BiGRU(385->64, bidir, 1 layer) center -> Linear(128,384)', optimizer='AdamW lr=1e-3 wd=1e-4',
        batch_size=128, epochs=args.epochs, updates=updates, seed=args.seed, threads=args.threads,
        torch=torch.__version__, code_commit=w.code_version(), elapsed_seconds=round(time.perf_counter() - t0, 1),
        labels_read=False, test_used=False,
        sha256={f: T.sha256(out / f) for f in ('gru_state.pt', 'norm_stats.npz', 'neighbors.npy')}))
    print(f'{run}: {updates} updates, final mse {hist[-1]["train_mse"]:.4f}, '
          f'+1 neighbor truly adjacent {true_adj:.4f}')


if __name__ == '__main__':
    main()
