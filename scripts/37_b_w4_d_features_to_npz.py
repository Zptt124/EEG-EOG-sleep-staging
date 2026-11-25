"""Step W4-B7: 把 D 的 04_extract_features.py 输出转成 B 探针（34 号）读的 npz + 样本索引。

D 的输出目录（SLEEP_W4_BIG/features/<run>/step<N>_<weights>_pool<k>s/）：
  {eeg,eog}_{std,mean}.npy  [N, 768] float32，行序 = index.csv
  index.csv                 sample_id,subject_id,record_id,epoch_index,split,role,y_true
  features_info.json        checkpoint SHA、读出配方、各文件 SHA、EOG-only 验收结果
  DONE

核对：DONE 存在；各 npy 的 SHA256 与 features_info.json 一致；index.csv SHA 一致；EOG-only 验收全部通过；
     sample_id 唯一；人员 = label25 的 14 人 + 8 名开发集验证者，不含测试者；y_true 与 A 的 NPZ 标签逐段一致。
输出（--out 目录）：
  <modality>_<stat>.npz     sample_id[str]、x[N,768] float32（未标准化）
  sample_index.csv          sample_id,subject_id,record_id,epoch_index,split,y_true
  conversion.json           来源目录、checkpoint SHA、读出配方、源文件与输出文件 SHA

用法：
  python scripts/37_b_w4_d_features_to_npz.py --src <D 的特征目录> --out <输出目录> [--stats std]
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / 'data' / 'processed'
SPLIT_CSV = BASE / 'manifests' / 'split_subjects.csv'
SUBSETS_CSV = BASE / 'manifests' / 'label_subsets_v2.csv'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stats', default='std', help='逗号分隔：std,mean')
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    if not (src / 'DONE').exists():
        sys.exit(f'{src} 没有 DONE，提取未完成')
    info = json.loads((src / 'features_info.json').read_text(encoding='utf-8'))
    if info.get('limit_records'):
        sys.exit('这是 --limit-records 的测试输出，不能用于评分')
    if sha256_file(src / 'index.csv') != info['index_sha256']:
        sys.exit('index.csv SHA256 与 features_info.json 不一致')
    ver = info.get('eog_only_verification') or []
    if 'eog' in info['modalities'] and (not ver or not all(v['eog_identical_without_eeg'] for v in ver)):
        sys.exit(f'EOG-only 验收缺失或未通过：{ver}')

    idx = pd.read_csv(src / 'index.csv', dtype={'epoch_index': int, 'y_true': int})
    if idx['sample_id'].duplicated().any():
        sys.exit('sample_id 重复')
    split = list(csv.DictReader(open(SPLIT_CSV, encoding='utf-8')))
    sub_split = {r['subject_id']: r['split'] for r in split}
    lab25 = {r['subject_id'] for r in csv.DictReader(open(SUBSETS_CSV, encoding='utf-8')) if r['label25'] == '1'}
    val = {s for s, sp in sub_split.items() if sp == 'val'}
    got = set(idx['subject_id'])
    if got != lab25 | val:
        sys.exit(f'人员不符：多 {sorted(got - lab25 - val)}，少 {sorted((lab25 | val) - got)}')
    if any(sub_split[s] == 'test' for s in got):
        sys.exit('含测试集受试者')
    # 真值与 A 的 NPZ 逐段核对
    n_bad = 0
    for rec, g in idx.groupby('record_id'):
        with np.load(PROCESSED / f'{rec}.npz') as z:
            lab = dict(zip(z['epoch_index'].tolist(), z['labels'].tolist()))
        n_bad += int(sum(lab.get(e) != y for e, y in zip(g['epoch_index'], g['y_true'])))
    if n_bad:
        sys.exit(f'{n_bad} 段 y_true 与 NPZ 标签不一致')

    out.mkdir(parents=True, exist_ok=True)
    ids = idx['sample_id'].to_numpy().astype(str)
    files = {}
    for m in info['modalities']:
        for st in args.stats.split(','):
            k = f'{m}_{st}'
            p = src / f'{k}.npy'
            if sha256_file(p) != info['files'][k]['sha256']:
                sys.exit(f'{k}.npy SHA256 与 features_info.json 不一致')
            x = np.load(p).astype(np.float32)
            if x.shape != (len(idx), info['dim_per_modality']) or not np.isfinite(x).all():
                sys.exit(f'{k}.npy 形状或数值异常 {x.shape}')
            np.savez(out / f'{k}.npz', sample_id=ids, x=x)
            files[f'{k}.npz'] = dict(source_sha256=info['files'][k]['sha256'], sha256=sha256_file(out / f'{k}.npz'),
                                     shape=list(x.shape))
    idx[['sample_id', 'subject_id', 'record_id', 'epoch_index', 'split', 'y_true']].to_csv(
        out / 'sample_index.csv', index=False, lineterminator='\n')
    conv = dict(source_dir=src.name, source_parent=src.parent.name, checkpoint=info['checkpoint'],
                checkpoint_sha256=info['checkpoint_sha256'], step=info['step'], weights=info['weights'],
                switches=info['switches'], readout=info['readout'], pool_sec=info['pool_sec'],
                device=info.get('device', 'cuda'), eog_only_verification=ver,
                n_samples=len(idx), n_subjects=len(got), n_labeled_train_subjects=len(got & lab25),
                n_val_subjects=len(got & val), y_true_checked_against_npz=True,
                sample_index_sha256=sha256_file(out / 'sample_index.csv'), files=files)
    (out / 'conversion.json').write_text(json.dumps(conv, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: conv[k] for k in ('checkpoint', 'step', 'pool_sec', 'device', 'n_samples', 'n_subjects')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
