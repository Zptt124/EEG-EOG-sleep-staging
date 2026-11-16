"""Step W4-B1: 锁定第四周预训练 held-out 名单与固定验证窗（B → A、D，第 1 天）。

规则（README 第四周、experiment3/week4/cross_modal_protocol.md）：
  - 从 54 人训练池中留出 8 名不在任何标签预算名单（label_subsets_v2, sampling_seed 2024 的 label10/25/50，
    即 5 ⊂ 14 ⊂ 27 人）里的受试者。协议最低要求是排除 14 人；同时排除 27 人档，使三档预算者全部留在预训练池内，
    以后在第四周 backbone 上画 5/14/27 预算曲线时条件一致。
  - 选法：27 名候选按受试者编号排序，np.array_split 分 8 层（4,4,4,3,3,3,3,3 人），每层用固定种子抽 1 人，
    使留出者覆盖整个编号范围（Sleep-EDF SC 的编号大体随年龄增加）。不读标签、不看任何模型结果。
  - 其余 46 人为第四周预训练池；开发集 8 人与测试集 16 人照旧不进入预训练。

固定验证窗（供 32_b_w4_pretrain_mse.py 计算分段 MSE）：
  - 每名留出者抽 PER_SUBJECT 个 5 秒窗位置 (record, epoch_index, k)，k 为 30 秒段内第 k 个 5 秒窗（0–5）。
  - 同一位置 EEG、EOG 各评估一次（t 相同、噪声独立），便于模态间比较，也便于以后做同窗条件 MSE。
  - 每名留出者内 t=0..49 各出现 PER_SUBJECT/50 次；噪声由 noise_seed 在 CPU 上按窗表行序生成，SHA256 写入元数据。
  - 只用 NPZ 的 epoch_index 与 signal 长度，不读 labels。

输出：
  manifests/pretrain_split_w4_v01.csv           54 名训练池受试者的预训练角色
  manifests/pretrain_val_windows_w4_v01.csv     固定验证窗（每行一个 窗×模态）
  manifests/pretrain_val_w4_v01.json            规则、种子、计数、清单与噪声 SHA256

用法：
  python scripts/31_b_w4_pretrain_heldout.py            # 生成
  python scripts/31_b_w4_pretrain_heldout.py --check    # 重新生成并与已入库文件逐字节比对
"""
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
MANIFESTS = BASE / 'manifests'
PROCESSED = BASE / 'data' / 'processed'
SPLIT_CSV = MANIFESTS / 'split_subjects.csv'
LABEL_CSV = MANIFESTS / 'label_subsets_v2.csv'
# 与 D 第三周 w3d_common.py 登记的 SHA 相同，名单文件一旦变化即报错
SPLIT_SHA = '1420db02a89907c0bf4ca66316718b711b0fcb682d61d96965ffb9a96423c10f'
LABEL_SHA = 'b343f5026b7f291ad2e1f89a78ccf2876e87473bc53b0a994329f49b8cb22ed6'

VERSION = 'w4_v01'
OUT_SPLIT = MANIFESTS / f'pretrain_split_{VERSION}.csv'
OUT_WINDOWS = MANIFESTS / f'pretrain_val_windows_{VERSION}.csv'
OUT_META = MANIFESTS / f'pretrain_val_{VERSION}.json'

SAMPLING_SEED = 2024        # 14 人 / 5 人预算所属抽样
SELECT_SEED = 2026          # 分层抽取留出者
WINDOW_SEED = 20261116      # 抽窗位置与分配 t
NOISE_SEED = 20261117       # 扩散噪声（由 32 号脚本按此重建）
N_HELDOUT = 8
PER_SUBJECT = 250           # 每名留出者的窗位置数；×2 模态 × 8 人 = 4000 行
T_STEPS = 50                # 官方 DDPMScheduler num_train_timesteps
T_SEGMENT = 5               # 分段 MSE：每 5 个扩散步一段，共 10 段
N_WIN = 6                   # 每 30 秒段切 6 个 5 秒窗（与 D 第三周一致）
WIN_200HZ = 1000
MODALITIES = (('eeg', 0), ('eog', 1))   # NPZ signal 通道顺序 = D 的模态编号


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_text(rows, fields):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def select_heldout():
    for path, sha in ((SPLIT_CSV, SPLIT_SHA), (LABEL_CSV, LABEL_SHA)):
        if sha256_file(path) != sha:
            sys.exit(f'SHA256 mismatch: {path}')
    split = [r for r in read_csv(SPLIT_CSV) if r['split'] == 'train']
    lab = {r['subject_id']: r for r in read_csv(LABEL_CSV) if int(r['sampling_seed']) == SAMPLING_SEED}
    if len(split) != 54 or set(lab) != {r['subject_id'] for r in split}:
        sys.exit('Training pool / label list mismatch')
    l25 = {s for s, r in lab.items() if r['label25'] == '1'}
    l10 = {s for s, r in lab.items() if r['label10'] == '1'}
    l50 = {s for s, r in lab.items() if r['label50'] == '1'}
    if len(l50) != 27 or len(l25) != 14 or len(l10) != 5 or not l10 <= l25 <= l50:
        sys.exit('Unexpected budget lists')

    cand = sorted(r['subject_id'] for r in split if r['subject_id'] not in l50)
    rng = np.random.default_rng(SELECT_SEED)
    stratum_of, heldout = {}, []
    for i, layer in enumerate(np.array_split(np.array(cand), N_HELDOUT)):
        layer = [str(s) for s in layer]
        for s in layer:
            stratum_of[s] = i
        heldout.append(layer[int(rng.integers(len(layer)))])

    rows = []
    for r in sorted(split, key=lambda r: r['subject_id']):
        s = r['subject_id']
        rows.append(dict(subject_id=s, record_ids=r['record_ids'], n_records=len(r['record_ids'].split(';')),
                         n_epochs=lab[s]['n_epochs'],
                         pretrain_role='heldout_val' if s in heldout else 'pretrain_train',
                         stratum=stratum_of.get(s, ''),
                         label10=lab[s]['label10'], label25=lab[s]['label25'], label50=lab[s]['label50']))
    return rows, heldout, cand


def build_windows(split_rows):
    """固定验证窗：每名留出者 PER_SUBJECT 个位置，t 在人内均衡，EEG/EOG 共用位置与 t。"""
    if PER_SUBJECT % T_STEPS:
        sys.exit('PER_SUBJECT must be a multiple of T_STEPS')
    rng = np.random.default_rng(WINDOW_SEED)
    positions = []
    for r in split_rows:
        if r['pretrain_role'] != 'heldout_val':
            continue
        cells = []   # (record_id, epoch_index, row_in_npz)
        for rec in r['record_ids'].split(';'):
            path = PROCESSED / f'{rec}.npz'
            if not path.exists():
                sys.exit(f'Missing {path}; 需要 A 的 153 个预处理 NPZ')
            with np.load(path) as z:
                ei = z['epoch_index']
                if z['signal'].shape != (len(ei), 2, 3000) or not np.isfinite(z['signal']).all():
                    sys.exit(f'Bad signal in {path}')
            cells += [(rec, int(e), i) for i, e in enumerate(ei)]
        n_units = len(cells) * N_WIN
        pick = np.sort(rng.choice(n_units, size=PER_SUBJECT, replace=False))
        t = rng.permutation(np.repeat(np.arange(T_STEPS), PER_SUBJECT // T_STEPS))
        for u, tt in zip(pick, t):
            rec, e, row = cells[u // N_WIN]
            positions.append(dict(subject_id=r['subject_id'], record_id=rec, epoch_index=e, npz_row=row,
                                  window_k=int(u % N_WIN), t=int(tt)))
    rows = []
    for m_name, m_id in MODALITIES:
        for p in positions:
            rows.append(dict(row=len(rows), unit_id=f"{p['record_id']}_{p['epoch_index']:04d}_k{p['window_k']}_{m_name}",
                             sample_id=f"{p['record_id']}_{p['epoch_index']:04d}", subject_id=p['subject_id'],
                             record_id=p['record_id'], epoch_index=p['epoch_index'], npz_row=p['npz_row'],
                             window_k=p['window_k'], start_sec=30 * p['epoch_index'] + 5 * p['window_k'],
                             modality=m_name, modality_id=m_id, t=p['t'], t_segment=p['t'] // T_SEGMENT))
    return rows


def make_noise(n_rows):
    """与 32 号脚本共用：CPU 上按行序生成 [n_rows, 1, 1000] 标准正态噪声。"""
    import torch
    g = torch.Generator().manual_seed(NOISE_SEED)
    return torch.randn((n_rows, 1, WIN_200HZ), generator=g, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='重新生成并与已入库文件比对，不写文件')
    args = ap.parse_args()

    split_rows, heldout, cand = select_heldout()
    win_rows = build_windows(split_rows)
    split_txt = csv_text(split_rows, list(split_rows[0]))
    win_txt = csv_text(win_rows, list(win_rows[0]))
    noise = make_noise(len(win_rows))

    pool = [r for r in split_rows if r['pretrain_role'] == 'pretrain_train']
    meta = dict(
        version=VERSION, owner='B', purpose='第四周小 EEGDM 预训练验证名单与固定分段 MSE 窗',
        inputs={'split_subjects.csv': SPLIT_SHA, 'label_subsets_v2.csv': LABEL_SHA},
        rule=dict(pool='split=train 的 54 人',
                  exclude='label_subsets_v2 sampling_seed=2024 label50=1 的 27 人（含 label25 的 14 人、label10 的 5 人）',
                  selection=f'剩余 {len(cand)} 人按编号排序 np.array_split 分 {N_HELDOUT} 层，'
                            f'每层 np.random.default_rng({SELECT_SEED}) 依次抽 1 人',
                  labels_used=False),
        heldout_subjects=heldout,
        heldout_records=[x for r in split_rows if r['pretrain_role'] == 'heldout_val' for x in r['record_ids'].split(';')],
        pretrain_pool=dict(n_subjects=len(pool), n_records=sum(r['n_records'] for r in pool),
                           n_epochs=sum(int(r['n_epochs']) for r in pool),
                           n_heldout_epochs=sum(int(r['n_epochs']) for r in split_rows if r['pretrain_role'] == 'heldout_val')),
        windows=dict(per_subject=PER_SUBJECT, modalities=[m for m, _ in MODALITIES], n_rows=len(win_rows),
                     window_seed=WINDOW_SEED, t_steps=T_STEPS, t_segment=T_SEGMENT, n_segments=T_STEPS // T_SEGMENT,
                     window_samples_200hz=WIN_200HZ, n_win_per_epoch=N_WIN,
                     window_slice='200 Hz 变换后 [1000k, 1000(k+1))，k=window_k',
                     preprocessing='100→200 Hz resample_poly(2,1)，µV/100，官方 staged_mu_law（与 D 第三周 transform_uv 一致）',
                     t_balance='每名留出者内 t=0..49 各 PER_SUBJECT/50 次；EEG 与 EOG 同位置同 t、噪声独立'),
        noise=dict(seed=NOISE_SEED, generator='torch.Generator(cpu).manual_seed(seed); torch.randn((n_rows,1,1000), float32)',
                   sha256=sha256_bytes(noise.numpy().tobytes())),
        files={OUT_SPLIT.name: sha256_bytes(split_txt.encode()), OUT_WINDOWS.name: sha256_bytes(win_txt.encode())},
    )
    meta_txt = json.dumps(meta, indent=2, ensure_ascii=False) + '\n'

    if args.check:
        ok = True
        for path, txt in ((OUT_SPLIT, split_txt), (OUT_WINDOWS, win_txt), (OUT_META, meta_txt)):
            same = path.exists() and path.read_text(encoding='utf-8') == txt
            ok &= same
            print(f'{"OK  " if same else "DIFF"} {path.relative_to(BASE)}')
        sys.exit(0 if ok else 1)

    OUT_SPLIT.write_text(split_txt, encoding='utf-8')
    OUT_WINDOWS.write_text(win_txt, encoding='utf-8')
    OUT_META.write_text(meta_txt, encoding='utf-8')
    print('held-out:', ', '.join(heldout))
    print(f"pretrain pool: {meta['pretrain_pool']}")
    print(f'windows: {len(win_rows)} rows; noise sha256 {meta["noise"]["sha256"][:16]}…')


if __name__ == '__main__':
    main()
