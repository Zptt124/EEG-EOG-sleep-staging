"""Step W1-A1: 时序自监督线三条件(A0 监督 / A1 真实邻接 / A2 同夜随机邻接)。

新一轮第 1 周(2026-11-26 定稿)时序线。用户执行 A、B 两条线(18 次拟合)。

- 特征:D 共享缓存(W4 30k EMA,第 10-12 层,1s std 30s 平均;eeg/eog 各 192 维拼 384)。
- BiGRU:单层、hidden 64、双向、取中心时刻输出(与 W5 41 号同构)。
- 窗口:中心 ±2(5 段),同受试者同记录;微调与评分一律真实邻接;缺窗/边界取最近可用
  epoch(位置钳制)。
- A0:随机初始化,直接用名单标签训练(CE)。
- A1:真实邻接掩码预训练——32 人(54−14−8)无标签窗口,中心特征置零,BiGRU 从真实
  邻居预测中心特征(MSE);一份权重共享两份名单;再在名单上监督微调(CE)。
- A2:同夜随机邻接——对无缺口连续段先做整段置换(固定种子,保存映射),再切 5 段窗口,
  中心由同夜随机邻居预测(MSE);微调同 A1(恢复真实邻接)。
- 拟合:固定 30 轮,AdamW lr1e-3 wd1e-4,batch 128 保留尾批,标准化只拟合名单训练侧;
  head seed 0/1/2。A0 按 seed 重建模型;A1/A2 共用同一份预训练权重,按 seed 重建
  head 后微调全部参数。
- 评分:每份名单评 dev 8 人(21,186 段)、补充 9 人(原 14 − 名单)、合并 17 人三套;
  报告 N1 AP、Macro-F1、原始 N1 F1、短 N1(连续 1–2 段)召回与 W/N2 错报。

用法(仓库根目录):
  预训练(A1/A2,一份权重共享两份名单):
    python scripts/44_a_w1_temporal.py --condition a1 --stage pretrain \
        --feat-dir cache/w1_temporal/w6_d_main30k_l10_12_v01/W6_D_main30k_l10_12_v01
  拟合与评分(18 次 = 3 条件 × 2 名单 × 3 seed):
    python scripts/44_a_w1_temporal.py --condition a1 --stage finetune --list 1 --seed 0 --feat-dir ...
    python scripts/44_a_w1_temporal.py --condition a0 --stage finetune --list 1 --seed 0 --feat-dir ...
  联调(小包,2 轮):
    python scripts/44_a_w1_temporal.py --condition a0 --stage finetune --list 1 --seed 0 \
        --feat-dir cache/w1_temporal/w6_d_main30k_l10_12_small_v01/W6_D_main30k_l10_12_small_v01 \
        --epochs 2 --small
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score

from common import BASE, RUNS
import w2_common as w

ev = w.ev
D = 384
WINDOW = 5
OFFSETS = (-2, -1, 0, 1, 2)
PERM_SEED = 20261126
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 30
LISTS_CSV = BASE / 'manifests' / 'w1_temporal_lists_v01.csv'
SUBSETS = BASE / 'manifests' / 'label_subsets_v2.csv'
CODE_VERSION = 'v01'


class BiGRUCenter(torch.nn.Module):
    """单层双向 GRU(hidden 64),取中心时刻输出。

    head:监督分类(5 类);pred:预训练特征重建投影(128 -> 384),仅预训练阶段使用。
    """

    def __init__(self, d=D, hidden=64):
        super().__init__()
        self.gru = torch.nn.GRU(d, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = torch.nn.Linear(2 * hidden, 5)
        self.pred = torch.nn.Linear(2 * hidden, d)

    def forward(self, x):  # [B, 5, d]
        out, _ = self.gru(x)
        return self.head(out[:, 2, :])


def load_cache(feat_dir):
    base = Path(feat_dir)
    rows = list(csv.DictReader(open(base / 'index.csv', encoding='utf-8')))
    eeg = np.load(base / 'eeg_std.npy', mmap_mode='r')
    eog = np.load(base / 'eog_std.npy', mmap_mode='r')
    if eeg.shape != (len(rows), 192) or eog.shape != (len(rows), 192):
        raise ValueError(f'特征形状异常 {eeg.shape} {eog.shape},期望 ({len(rows)}, 192)')
    x = np.concatenate([np.asarray(eeg, dtype=np.float32), np.asarray(eog, dtype=np.float32)], axis=1)
    info = json.loads((base / 'features_info.json').read_text(encoding='utf-8'))
    return rows, x, info


def list_subjects(list_id):
    return sorted({r['subject_id'] for r in csv.DictReader(open(LISTS_CSV, encoding='utf-8'))
                   if int(r['list_id']) == list_id})


def original14():
    return sorted({r['subject_id'] for r in csv.DictReader(open(SUBSETS, encoding='utf-8'))
                   if r['sampling_seed'] == '2024' and r['label25'] == '1'})


def contiguous_runs(record_entries):
    """同一记录内按 epoch 连续的段(不跨缺口)。entries = [(epoch_index, row_pos), ...] 已按 epoch 排序。"""
    runs, run = [], [record_entries[0]]
    for a, b in zip(record_entries, record_entries[1:]):
        if b[0] == a[0] + 1:
            run.append(b)
        else:
            runs.append(run)
            run = [b]
    runs.append(run)
    return runs


def build_windows(rows, x, subject_filter=None):
    """真实邻接窗口 [N,5,384]。同记录、同连续段(不跨缺口)内取 ±2;边界取段内最近可用 epoch。"""
    pos = defaultdict(list)
    for i, r in enumerate(rows):
        if subject_filter is None or r['subject_id'] in subject_filter:
            pos[r['record_id']].append((int(r['epoch_index']), i))
    idx_all = [i for ps in pos.values() for _, i in sorted(ps)]
    seq = np.empty((len(idx_all), WINDOW, D), dtype=np.float32)
    row_of = {i: k for k, i in enumerate(idx_all)}
    for rec in sorted(pos):
        for run in contiguous_runs(sorted(pos[rec])):
            L = len(run)
            for k, (_, i) in enumerate(run):
                idx = [run[min(L - 1, max(0, k + off))][1] for off in OFFSETS]
                seq[row_of[i]] = x[idx]
    return seq, np.array(idx_all), row_of


def build_permuted_windows(rows, x, subject_filter, out_dir):
    """A2:先对无缺口连续段整段置换(固定种子),再切 ±2 窗口;保存置换映射。

    无缺口连续段 = 同一记录内 epoch_index 连续的最大段。先置换后切窗(不是只调换
    窗口内邻居)。"""
    perm_map, pos = {}, defaultdict(list)
    for i, r in enumerate(rows):
        if subject_filter is not None and r['subject_id'] not in subject_filter:
            continue
        pos[r['record_id']].append((int(r['epoch_index']), i))
    seq_rows = []
    rng = np.random.default_rng(PERM_SEED)
    for rec in sorted(pos):
        runs = contiguous_runs(sorted(pos[rec]))
        for ri, run in enumerate(runs):
            order = rng.permutation(len(run))
            perm_map[f'{rec}#run{ri}'] = dict(original=[e[0] for e in run],
                                             permuted=[run[o][0] for o in order])
            permuted = [run[o][1] for o in order]  # 该连续段内的置换行序列
            L = len(permuted)
            for k, i in enumerate(permuted):
                idx = [permuted[min(L - 1, max(0, k + off))] for off in OFFSETS]  # 段内钳制,不跨缺口
                seq_rows.append((i, idx))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'permutation_map.json').write_text(json.dumps(perm_map, indent=1, ensure_ascii=False) + '\n',
                                              encoding='utf-8')
    seq = np.empty((len(seq_rows), WINDOW, D), dtype=np.float32)
    centers = []
    for k, (center, idx) in enumerate(seq_rows):
        seq[k] = x[idx]
        centers.append(center)
    return seq, np.array(centers), perm_map


def masked_input(seq):
    """遮挡中心:中心位置置零。"""
    s = seq.copy()
    s[:, 2, :] = 0.0
    return s


def train_pretrain(seq, seed, epochs, out_dir):
    """MSE 预训练:从遮挡中心的 5 段窗口预测中心特征。"""
    w.seed_everything(seed, deterministic=True)
    model = BiGRUCenter()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target = torch.from_numpy(seq[:, 2, :])
    inp = torch.from_numpy(masked_input(seq))
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        losses = []
        for ids in torch.randperm(len(target)).split(128):
            opt.zero_grad(set_to_none=True)
            out, _ = model.gru(inp[ids])
            loss = torch.nn.functional.mse_loss(model.pred(out[:, 2, :]), target[ids])
            loss.backward()
            opt.step()
            losses.append(loss.item() * len(ids))
        history.append(dict(epoch=ep, mse=sum(losses) / len(target),
                            seconds=round(time.perf_counter() - t0, 2)))
        if ep % 5 == 0:
            print(f'  pretrain epoch {ep}/{epochs} mse {history[-1]["mse"]:.5f}', flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), epoch=epochs, seed=seed), out / 'pretrain.ckpt')
    (out / 'pretrain_history.json').write_text(json.dumps(history, indent=2) + '\n', encoding='utf-8')
    return model, history


def standardize_fit(x_fit):
    mean, std = x_fit.mean(axis=0), x_fit.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_std(x, mean, std):
    return torch.from_numpy((x - mean) / std)


def train_finetune(model, x_tr, y_tr, seed, epochs):
    """监督微调:CE,固定轮数;模型权重从当前状态出发,按 seed 重置数据顺序与 head。

    head seed 语义:A0 整个模型按 seed 重建;A1/A2 预训练主干固定,head 按 seed 重建。
    """
    w.seed_everything(seed, deterministic=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y = torch.as_tensor(y_tr, dtype=torch.long)
    updates = 0
    for _ in range(epochs):
        model.train()
        for ids in torch.randperm(len(y)).split(128):
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x_tr[ids]), y[ids]).backward()
            opt.step()
            updates += 1
    return model, updates


def predict_group(model, x_eval):
    probs = w.predict_probs(model, lambda ids: x_eval[ids], len(x_eval))
    return probs


def short_n1_stats(rows, probs):
    """短 N1(连续 1–2 段,不跨缺口)召回与 W/N2 错报。rows = 该评估组的 index 行。"""
    y_true = np.array([int(r['y_true']) for r in rows])
    y_pred = probs.argmax(1)
    short_idx = set()
    by_rec = defaultdict(list)
    for i, r in enumerate(rows):
        by_rec[r['record_id']].append((int(r['epoch_index']), i))
    for rec, entries in by_rec.items():
        for run in contiguous_runs(sorted(entries)):
            n1_pos = [i for e, i in run if y_true[i] == 1]
            # 在段内按位置连续切 N1 run
            k = 0
            while k < len(n1_pos):
                m = k
                while m + 1 < len(n1_pos) and n1_pos[m + 1] == n1_pos[m] + 1:
                    m += 1
                if m - k + 1 <= 2:
                    short_idx.update(n1_pos[k:m + 1])
                k = m + 1
    short = sorted(short_idx)
    if not short:
        return dict(n_short=0, recall=None, to_W=None, to_N2=None)
    yt, yp = y_true[short], y_pred[short]
    return dict(n_short=len(short),
                recall=float((yp == 1).mean()),
                to_W=float((yp == 0).mean()),
                to_N2=float((yp == 2).mean()))


def eval_group(name, rows, probs, out_dir):
    df_rows = [dict(sample_id=r['sample_id'], subject_id=r['subject_id'], record_id=r['record_id'],
                    epoch_index=r['epoch_index'], y_true=int(r['y_true']),
                    **{c: p for c, p in zip(w.PROB_COLS, probs[i])}) for i, r in enumerate(rows)]
    pred = pd.DataFrame(df_rows)
    pred['y_pred'] = pred[w.PROB_COLS].to_numpy().argmax(1)
    pred = pred[['sample_id', 'subject_id', 'record_id', 'epoch_index', 'y_true', 'y_pred'] + list(w.PROB_COLS)]
    out = Path(out_dir)
    pred.to_csv(out / f'predictions_{name}.csv', index=False, float_format='%.8g')
    y1 = (pred['y_true'] == 1).to_numpy()
    m = ev.compute_metrics(pred['y_true'].to_numpy(), pred['y_pred'].to_numpy())
    metrics = dict(group=name, n_epochs=len(pred), n_subjects=pred['subject_id'].nunique(),
                   macro_f1=m['macro_f1'], n1_f1=m['n1_f1'],
                   n1_ap=float(average_precision_score(y1, pred['p_N1'])),
                   n1_precision=m['n1_precision'], n1_recall=m['n1_recall'],
                   short_n1=short_n1_stats(rows, probs))
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--condition', required=True, choices=['a0', 'a1', 'a2'])
    ap.add_argument('--stage', required=True, choices=['pretrain', 'finetune'])
    ap.add_argument('--list', type=int, choices=[1, 2])
    ap.add_argument('--seed', type=int, choices=[0, 1, 2])
    ap.add_argument('--feat-dir', required=True)
    ap.add_argument('--epochs', type=int, default=None, help='覆盖默认 30 轮(仅联调用)')
    ap.add_argument('--small', action='store_true', help='小包联调:只跑前几条记录')
    ap.add_argument('--runs-root', default=str(RUNS))
    ap.add_argument('--group', default='A')
    args = ap.parse_args(argv)

    if args.stage == 'pretrain':
        if args.condition not in ('a1', 'a2') or args.list is not None or args.seed is not None:
            sys.exit('pretrain 只用于 a1/a2,不需要 --list/--seed')
    else:
        if args.list is None or args.seed is None:
            sys.exit('finetune 需要 --list 与 --seed')

    epochs = args.epochs or (PRETRAIN_EPOCHS if args.stage == 'pretrain' else FINETUNE_EPOCHS)
    rows, x, info = load_cache(args.feat_dir)
    torch.set_num_threads(4)

    l14 = original14()
    lists = {1: list_subjects(1), 2: list_subjects(2)}
    pretrain_subjects = {r['subject_id'] for r in rows
                         if r['role'] == 'pretrain_train' and r['subject_id'] not in l14}
    dev_rows = [r for r in rows if r['role'] == 'dev']

    if args.small:
        # 联调子集:名单 1 的前两条记录 + 第一条 dev 记录(端到端、含真实名单与评分对象)
        if args.stage == 'finetune' and args.list == 1:
            recs = sorted({r['record_id'] for r in rows if r['subject_id'] in lists[1]})[:2]
            recs += sorted({r['record_id'] for r in dev_rows})[:1]
        else:
            recs = sorted({r['record_id'] for r in rows})[:3]
        keep = set(recs)
        keep_pos = [i for i, r in enumerate(rows) if r['record_id'] in keep]
        rows = [rows[i] for i in keep_pos]
        x = x[keep_pos]
        dev_rows = [r for r in rows if r['role'] == 'dev']

    cond_name = dict(a0='A0', a1='A1', a2='A2')[args.condition]

    if args.stage == 'pretrain':
        tag = f'W1_{args.group}_{args.condition}_pretrain_v01'
        out_dir = Path(args.runs_root) / tag
        if args.condition == 'a1':
            seq, centers, _ = build_windows(rows, x, pretrain_subjects)
        else:
            seq, centers, _ = build_permuted_windows(rows, x, pretrain_subjects, out_dir)
        print(f'{tag}: {len(seq)} 窗口,{len(pretrain_subjects)} 人,{epochs} 轮', flush=True)
        model, history = train_pretrain(seq, 0, epochs, out_dir)
        (out_dir / 'config.yaml').write_text(yaml.safe_dump(dict(
            experiment='W1 时序线预训练', condition=cond_name, stage='pretrain',
            n_subjects=len(pretrain_subjects), n_windows=len(seq), epochs=epochs,
            batch_size=128, loss='MSE on masked center (zeroed)', seed=0,
            permutation_seed=PERM_SEED if args.condition == 'a2' else None,
            feature_info_sha256=info.get('checkpoint_sha256'), window=list(OFFSETS),
            model='BiGRU(64,bidir,1层)取中心时刻', test_used=False), allow_unicode=True, sort_keys=False))
        print(f'{tag} DONE mse {history[-1]["mse"]:.5f}', flush=True)
        return

    # ---- finetune ----
    list_subj = lists[args.list]
    extra_subj = [s for s in l14 if s not in list_subj]
    tr_rows = [r for r in rows if r['role'] == 'pretrain_train' and r['subject_id'] in list_subj]
    extra_rows = [r for r in rows if r['role'] == 'pretrain_train' and r['subject_id'] in extra_subj]
    if not tr_rows or (not args.small and len({r['subject_id'] for r in tr_rows}) != 5):
        sys.exit(f'名单 {args.list} 训练行异常: {len(tr_rows)} 段,'
                 f' {len({r["subject_id"] for r in tr_rows})} 人')
    tr_pos = {r['sample_id']: i for i, r in enumerate(rows)}
    tr_idx = np.array([tr_pos[r['sample_id']] for r in tr_rows])
    ex_idx = np.array([tr_pos[r['sample_id']] for r in extra_rows])
    dv_idx = np.array([tr_pos[r['sample_id']] for r in dev_rows])

    def windows_for(idx):
        """真实邻接、不跨缺口(连续段内钳制)的窗口。"""
        sub = defaultdict(list)
        for j, i in enumerate(idx):
            sub[rows[i]['record_id']].append((int(rows[i]['epoch_index']), i, j))
        seq = np.empty((len(idx), WINDOW, D), dtype=np.float32)
        for rec, entries in sub.items():
            for run in contiguous_runs(sorted(entries)):
                L = len(run)
                for k, (_, _, j) in enumerate(run):
                    seq[j] = x[[run[min(L - 1, max(0, k + off))][1] for off in OFFSETS]]
        return seq

    seq_tr = windows_for(tr_idx)
    seq_ex = windows_for(ex_idx)
    seq_dv = windows_for(dv_idx)
    mean, std = standardize_fit(seq_tr.reshape(-1, D))
    x_tr = apply_std(seq_tr.reshape(-1, D), mean, std).reshape(-1, WINDOW, D)
    x_ex = apply_std(seq_ex.reshape(-1, D), mean, std).reshape(-1, WINDOW, D)
    x_dv = apply_std(seq_dv.reshape(-1, D), mean, std).reshape(-1, WINDOW, D)
    y_tr = np.array([int(r['y_true']) for r in tr_rows])

    if args.condition == 'a0':
        w.seed_everything(args.seed, deterministic=True)
        model = BiGRUCenter()
    else:
        ckpt = Path(args.runs_root) / f'W1_{args.group}_{args.condition}_pretrain_v01/pretrain.ckpt'
        if not ckpt.exists():
            sys.exit(f'缺预训练权重 {ckpt},先跑 --stage pretrain')
        model = BiGRUCenter()
        state = torch.load(ckpt, map_location='cpu', weights_only=True)['state']
        model.load_state_dict(state)
        w.seed_everything(args.seed, deterministic=True)
        model.head = torch.nn.Linear(128, 5)  # head 按 seed 重建
    model, updates = train_finetune(model, x_tr, y_tr, args.seed, epochs)

    tag = f'W1_{args.group}_{args.condition}_list{args.list}_seed{args.seed}_v01'
    out_dir = Path(args.runs_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    metrics = []
    for name, idx, rws in (('dev8', dv_idx, dev_rows), ('extra9', ex_idx, extra_rows)):
        if len(rws) == 0:
            metrics.append(dict(group=name, n_epochs=0, note='empty group (small-pack integration)'))
            continue
        probs = predict_group(model, locals()[f'x_{"dv" if name == "dev8" else "ex"}'])
        metrics.append(eval_group(name, rws, probs, out_dir))
    merged_rows = dev_rows + extra_rows
    if merged_rows:
        merged_probs = np.concatenate([predict_group(model, x_dv), predict_group(model, x_ex)])
        metrics.append(eval_group('merged17', merged_rows, merged_probs, out_dir))
    else:
        metrics.append(dict(group='merged17', n_epochs=0, note='empty group (small-pack integration)'))
    elapsed = time.perf_counter() - t0

    config = dict(experiment='W1 时序线拟合与评分', condition=cond_name, list_id=args.list,
                  list_subjects=list_subj, extra9=extra_subj, head_seed=args.seed,
                  epochs=epochs, updates=updates, batch_size=128,
                  optimizer='AdamW lr=1e-3 wd=1e-4 (保留尾批)',
                  standardization='fit on list train rows only (unfolded [N*5,384])',
                  window=dict(offsets=list(OFFSETS), rule='real adjacency for finetune/eval; '
                              'missing/boundary -> nearest available epoch'),
                  pretrain=dict(condition=args.condition,
                                subjects='32 = 54-14-8 (role=pretrain_train, not in label25 sseed2024)',
                                weight='W1_%s_%s_pretrain_v01/pretrain.ckpt' % (args.group, args.condition)
                                if args.condition != 'a0' else None,
                                permutation_seed=PERM_SEED if args.condition == 'a2' else None),
                  metrics=metrics, elapsed_seconds=round(elapsed, 2),
                  feature_info_sha256=info.get('checkpoint_sha256'), test_used=False,
                  code_version=CODE_VERSION, torch=str(torch.__version__))
    (out_dir / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                                         encoding='utf-8')
    (out_dir / 'metrics.json').write_text(json.dumps(dict(metrics=metrics), indent=2, ensure_ascii=False) + '\n',
                                          encoding='utf-8')
    torch.save(dict(state=model.state_dict(), mean=mean, std=std, seed=args.seed),
               out_dir / 'final.ckpt')
    print(f'{tag}: ' + ' | '.join(f"{m['group']} N1AP {m['n1_ap']:.3f} Macro {m['macro_f1']:.4f}"
                                  for m in metrics if 'n1_ap' in m) + f'  {elapsed:.0f}s', flush=True)


if __name__ == '__main__':
    main()
