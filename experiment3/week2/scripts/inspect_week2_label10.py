"""Feasibility counts for proposed 10% nested lists; does not publish/replace B lists."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'vendor/experiment2-review-6e7df99/manifests'
OUT=ROOT/'reports/week2_plan_20261026'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def main():
    index=read(ROOT/'features/eeg_cond14_v01/feature_index.csv')
    train=[r for r in index if r['split']=='train']
    assert len(train)==132938
    counts=defaultdict(Counter)
    records=defaultdict(set)
    for r in train:
        counts[r['subject_id']][int(r['y_true'])]+=1
        records[r['subject_id']].add(r['record_id'])
    result=[]
    for seed in (2024,2025,2026):
        filename='label_subsets_v1.csv' if seed==2024 else f'label_subsets_v1-s{seed}.csv'
        path=B/filename
        rows=sorted(read(path),key=lambda r:int(r['sampling_rank']))
        assert len(rows)==len({r['subject_id'] for r in rows})==54
        assert {r['subject_id'] for r in rows}==set(counts)
        k=round(.1*len(rows))
        selected=rows[:k]
        assert k==5 and all(r['label25']=='1' for r in selected)
        total=Counter()
        for r in selected:
            assert sum(counts[r['subject_id']].values())==int(r['n_epochs'])
            total.update(counts[r['subject_id']])
        result.append(dict(sampling_seed=seed,labeled_subjects=k,unlabeled_subjects=54-k,
            actual_subject_percentage=100*k/54,subjects=[r['subject_id'] for r in selected],
            labeled_records=sum(len(records[r['subject_id']]) for r in selected),
            labeled_epochs=sum(total.values()),actual_epoch_percentage=100*sum(total.values())/len(train),
            class_counts=dict(zip(['W','N1','N2','N3','REM'],[total[i] for i in range(5)])),
            all_classes_present=all(total[i]>0 for i in range(5)),nested_with_existing_label25=True,
            original_B_manifest=filename,original_B_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(status='proposal_feasibility_only',rule='round(54*0.10)=5; first 5 by existing sampling_rank',
        official_manifest_published=False,validation_or_test_used=False,
        no_subject_reselection_based_on_class_counts=True,proposals=result)
    (OUT/'label10_feasibility_DRAFT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    main()
