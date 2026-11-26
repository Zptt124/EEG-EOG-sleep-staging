"""新一轮第 1 周 · 时序线公共部分（A0 / A1 / A2 共用；设置 2026-11-26 已与 A 对齐）。

- 特征：每段 EEG、EOG 各 192 维（W4 30k EMA，t=1，第 10–12 层，每秒 std 后 30 秒平均），拼成 384 维。
  若输入是 W4 全 12 层的 768 维特征，取第 576–768 列（第 10–12 层），与 D 的新版缓存口径相同。
- 窗口：中心 ±2 共 5 个位置。邻居必须是同一记录（夜）内 epoch_index 恰好相差 k 的段，不跨夜、不跨缺口；
  缺的位置特征置 0（标准化之后），并在第 385 维写 mask=1。A1 预训练遮挡中心时用同一个标记。
- A2：在每段无缺口连续记录内，用固定 seed 把段的顺序置换一次，再按置换后的位置取 ±2 窗口（不出段）。
- 模型：BiGRU(385→64，双向，1 层)，取中心位置输出，接 Linear(128, 5)。
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

OFFSETS = (-2, -1, 0, 1, 2)
CENTER = 2
D_FEAT = 384
D_IN = D_FEAT + 1          # 最后一维是 mask 标记
LAYER_COLS = slice(576, 768)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_features(feat_dir):
    """-> (index_rows, x [N,384] float32, info)。768 维输入取第 10–12 层。"""
    feat_dir = Path(feat_dir)
    rows = read_csv(feat_dir / 'index.csv')
    parts, info = [], dict(feat_dir=str(feat_dir), index_sha256=sha256(feat_dir / 'index.csv'), files={})
    for m in ('eeg', 'eog'):
        p = feat_dir / f'{m}_std.npy'
        a = np.load(p, mmap_mode='r')
        if a.shape[0] != len(rows):
            raise SystemExit(f'{p.name} 行数 {a.shape[0]} != index {len(rows)}')
        if a.shape[1] == 768:
            a = a[:, LAYER_COLS]
            info['layer_slice'] = '576:768 of 768 (layers 10-12)'
        elif a.shape[1] != 192:
            raise SystemExit(f'{p.name} 维度 {a.shape[1]}，期望 192 或 768')
        parts.append(np.asarray(a, dtype=np.float32))
        info['files'][p.name] = sha256(p)
    fi = feat_dir / 'features_info.json'
    if fi.exists():
        info['features_info_sha256'] = sha256(fi)
    return rows, np.concatenate(parts, axis=1), info


def fit_standardizer(x):
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x.std(0, dtype=np.float64).astype(np.float32), 1e-6)
    return mean, std


def load_segments(path):
    """-> {record_id: [(start, end), ...]}，start/end 为 epoch_index（含两端）。"""
    seg = defaultdict(list)
    for r in read_csv(path):
        seg[r['record_id']].append((int(r['start_epoch']), int(r['end_epoch'])))
    return seg


def neighbor_table(rows, segments, permute_seed=None):
    """-> nbr [N,5] int64，值为 rows 中的行号，-1 表示该位置缺失。

    permute_seed=None：真实邻接（同记录、epoch_index±k、同一连续段）。
    permute_seed=s：A2，每个连续段内用 np.random.default_rng(s) 置换一次段的顺序，再按置换后位置取 ±2（不出段）。
    同时核对每条记录的段落覆盖与 rows 的 epoch 完全一致。
    """
    by_rec = defaultdict(dict)
    for i, r in enumerate(rows):
        by_rec[r['record_id']][int(r['epoch_index'])] = i
    nbr = np.full((len(rows), len(OFFSETS)), -1, dtype=np.int64)
    rng = np.random.default_rng(permute_seed) if permute_seed is not None else None
    for rec in sorted(by_rec):
        pos = by_rec[rec]
        segs = segments.get(rec)
        if segs is None:
            raise SystemExit(f'连续段索引里没有记录 {rec}')
        covered = set()
        for a, b in sorted(segs):
            order = [pos[e] for e in range(a, b + 1)]          # 段内行号，按 epoch 顺序
            covered.update(range(a, b + 1))
            if rng is not None:
                order = [order[j] for j in rng.permutation(len(order))]
            L = len(order)
            for p, i in enumerate(order):
                for c, k in enumerate(OFFSETS):
                    if 0 <= p + k < L:
                        nbr[i, c] = order[p + k]
        if covered != set(pos):
            raise SystemExit(f'{rec}: 连续段覆盖与特征 index 的 epoch 不一致')
    return nbr


def windows(x_std, nbr, idx):
    """x_std [N,384]（已标准化），nbr [N,5]，idx 行号 -> [len(idx),5,385] float32；缺位置 0 + mask=1。"""
    nb = nbr[idx]
    out = np.zeros((len(idx), len(OFFSETS), D_IN), dtype=np.float32)
    ok = nb >= 0
    out[..., :D_FEAT][ok] = x_std[nb[ok]]
    out[..., D_FEAT] = (~ok).astype(np.float32)
    return out


SEED_OFFSET = {'gru': 0, 'head': 10_000, 'decoder': 20_000}


def init_under(seed, part, ctor):
    """在独立随机流里构造一个子模块：不受、也不影响全局 RNG 和其他子模块。
    这样 A0 是否初始化 BiGRU、A1/A2 是否覆盖 GRU，都不会改变分类头初始化。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + SEED_OFFSET[part])
        return ctor()


def batch_order(seed):
    """batch 打乱专用的 generator（与模型初始化、A2 置换分开）。"""
    return torch.Generator().manual_seed(seed)


class BiGRUCenter(torch.nn.Module):
    """与 W5 A 的 41 号相同结构：GRU(d→64，双向，1 层) 直接接 Linear(128,5)，无投影层；d = 384 + 1 位 mask。"""

    def __init__(self, seed, d=D_IN, hidden=64):
        super().__init__()
        self.gru = init_under(seed, 'gru', lambda: torch.nn.GRU(d, hidden, num_layers=1, batch_first=True,
                                                                 bidirectional=True))
        self.head = init_under(seed, 'head', lambda: torch.nn.Linear(2 * hidden, 5))

    def encode(self, x):                 # [B,5,d] -> [B,5,128]
        return self.gru(x)[0]

    def forward(self, x):
        return self.head(self.encode(x)[:, CENTER, :])


def short_n1_runs(rows, segments, max_len=2):
    """真值 N1（y_true=1）在连续段内的连续长度 ≤ max_len 的段 -> 布尔数组 [N]。"""
    by_rec = defaultdict(dict)
    for i, r in enumerate(rows):
        by_rec[r['record_id']][int(r['epoch_index'])] = i
    flag = np.zeros(len(rows), dtype=bool)
    for rec, pos in by_rec.items():
        for a, b in segments[rec]:
            run = []
            for e in range(a, b + 2):
                i = pos.get(e) if e <= b else None
                if i is not None and int(rows[i]['y_true']) == 1:
                    run.append(i)
                    continue
                if 0 < len(run) <= max_len:
                    flag[run] = True
                run = []
    return flag


def temporal_metrics(y_true, y_pred, short_n1):
    """短 N1 召回（按段）与 W/N2 被判成 N1 的个数和比例。"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    wn2 = np.isin(y_true, [0, 2])
    return dict(short_n1_n=int(short_n1.sum()),
                short_n1_recall=float((y_pred[short_n1] == 1).mean()) if short_n1.any() else float('nan'),
                wn2_to_n1_n=int((wn2 & (y_pred == 1)).sum()),
                wn2_to_n1_rate=float((y_pred[wn2] == 1).mean()))


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
