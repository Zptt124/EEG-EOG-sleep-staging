"""新一轮第 1 周 · 时序线：导出两份 5 人名单、各自的 9 人补充评估集，以及 A1/A2 预训练的 32 人。

规则（草案，待与 A 确认；不引入新的随机数，只用已锁定的 label_subsets_v2 抽样顺序）：
  - 原 14 人 = label_subsets_v2 中 label25=1 的受试者（sampling_rank 1–14）。
  - 名单 1 = sampling_rank 1–5（即现有 label10，可与 W3–W5 的 5 人档结果对接）。
  - 名单 2 = sampling_rank 6–10（与名单 1 不重叠）。
  - 每份名单的补充评估集 = 原 14 人去掉该名单的 5 人（9 人）；与固定 dev 8 人合并为 17 人。
  - 预训练 32 人 = 训练池 54 人 − 原 14 人 − W4 held-out 8 人（pretrain_split_w4_v01 的 heldout_val）。
核对：dev 8 人 = split_subjects 的 val；任何名单、评估集、预训练集都不含 dev（主评除外）、测试集或 held-out。

输出（--out 目录）：
  temporal_rosters_t1_v01.csv   roster,subject_id,record_ids,role,sampling_rank
                                role ∈ label / eval9 / dev8；预训练集写作 roster=pretrain, role=pretrain32
  temporal_rosters_t1_v01.json  规则、来源文件 SHA256、各集合人数与名单、输出 SHA256

用法：
  python scripts/44_b_t1_temporal_rosters.py --out manifests
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = {
    'label_subsets': BASE / 'manifests' / 'label_subsets_v2.csv',
    'split_subjects': BASE / 'manifests' / 'split_subjects.csv',
    'pretrain_split': BASE / 'manifests' / 'pretrain_split_w4_v01.csv',
}
ROSTER_RANKS = {'1': range(1, 6), '2': range(6, 11)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split = {r['subject_id']: r for r in read(SRC['split_subjects'])}
    train = {s for s, r in split.items() if r['split'] == 'train'}
    dev = sorted(s for s, r in split.items() if r['split'] == 'val')
    test = {s for s, r in split.items() if r['split'] == 'test'}
    pre = read(SRC['pretrain_split'])
    heldout = {r['subject_id'] for r in pre if r['pretrain_role'] == 'heldout_val'}
    lab = sorted(read(SRC['label_subsets']), key=lambda r: int(r['sampling_rank']))
    rank = {r['subject_id']: int(r['sampling_rank']) for r in lab}
    orig14 = [r['subject_id'] for r in lab if r['label25'] == '1']

    assert len(train) == 54 and len(dev) == 8 and len(test) == 16 and len(heldout) == 8
    assert [rank[s] for s in orig14] == list(range(1, 15)), '原 14 人应为 sampling_rank 1–14'
    assert set(orig14) <= train and not (set(orig14) & heldout)
    assert heldout <= train

    rows = []
    rosters = {}
    for rid, ranks in ROSTER_RANKS.items():
        label = [s for s in orig14 if rank[s] in ranks]
        eval9 = [s for s in orig14 if s not in label]
        assert len(label) == 5 and len(eval9) == 9
        rosters[rid] = dict(label=label, eval9=eval9)
        for role, subs in (('label', label), ('eval9', eval9), ('dev8', dev)):
            for s in subs:
                rows.append(dict(roster=rid, subject_id=s, record_ids=split[s]['record_ids'], role=role,
                                 sampling_rank=rank.get(s, '')))
    assert not set(rosters['1']['label']) & set(rosters['2']['label']), '两份名单应不重叠'

    pretrain32 = sorted(train - set(orig14) - heldout)
    assert len(pretrain32) == 32 and not (set(pretrain32) & (set(dev) | test | heldout | set(orig14)))
    for s in pretrain32:
        rows.append(dict(roster='pretrain', subject_id=s, record_ids=split[s]['record_ids'], role='pretrain32',
                         sampling_rank=rank.get(s, '')))

    csv_path = out / 'temporal_rosters_t1_v01.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['roster', 'subject_id', 'record_ids', 'role', 'sampling_rank'],
                           lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    info = dict(
        script='scripts/44_b_t1_temporal_rosters.py',
        status='draft: rule pending alignment with A',
        rule={'orig14': 'label_subsets_v2 label25=1 (sampling_rank 1-14)',
              'roster_1': 'sampling_rank 1-5 (= existing label10)',
              'roster_2': 'sampling_rank 6-10 (disjoint from roster 1)',
              'eval9': 'orig14 minus the roster',
              'eval17': 'eval9 + dev8',
              'pretrain32': 'train54 - orig14 - W4 heldout8'},
        sources={k: dict(path=str(v.relative_to(BASE)), sha256=sha256(v)) for k, v in SRC.items()},
        rosters=rosters, dev8=dev, pretrain32=pretrain32,
        output_sha256=sha256(csv_path))
    (out / 'temporal_rosters_t1_v01.json').write_text(json.dumps(info, indent=1, ensure_ascii=False) + '\n',
                                                      encoding='utf-8')
    for rid, r in rosters.items():
        print(f"roster {rid}: label {' '.join(r['label'])} | eval9 {' '.join(r['eval9'])}")
    print(f"dev8 {' '.join(dev)}")
    print(f"pretrain32 ({len(pretrain32)}) sha {info['output_sha256'][:12]}")


if __name__ == '__main__':
    main()
