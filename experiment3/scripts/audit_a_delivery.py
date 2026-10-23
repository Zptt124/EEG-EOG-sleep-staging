import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np

root = Path(__file__).resolve().parents[1]
repo = root.parent
package = repo / '交付包/实验①交付包_20261018'
data = package / '2_预处理数据与脚本_给B和C/data/processed'
def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
split = rows(repo / 'manifests/split_subjects.csv')
assert len({r['subject_id'] for r in split}) == len(split) == 78
records = {}
for r in split:
    for rec in r['record_ids'].split(';'):
        assert rec not in records and rec[:5] == r['subject_id']
        records[rec] = r['split']
assert set(records) == {p.stem for p in data.glob('*.npz')}
counts = Counter()
val_truth = {}
for rec, partition in records.items():
    if partition == 'test':
        continue  # Do not inspect test arrays or labels.
    with np.load(data / f'{rec}.npz', allow_pickle=False) as z:
        assert set(z.files) == {'signal', 'labels', 'epoch_index'}
        labels, indices = z['labels'], z['epoch_index']
        assert labels.shape == indices.shape
        assert np.all(np.isin(labels, range(5)))
        assert np.all(np.diff(indices) > 0)
        counts[partition] += len(labels)
        if partition == 'val':
            val_truth.update({f'{rec}_{int(i):04d}': int(y) for i,y in zip(indices,labels)})
with np.load(data / 'SC4001.npz', allow_pickle=False) as z:
    signal = z['signal']
    assert signal.shape == (len(z['labels']), 2, 3000)
    assert np.isfinite(signal).all()
    sample = {'record':'SC4001', 'split':records['SC4001'], 'shape':list(signal.shape),
              'dtype':str(signal.dtype), 'finite':True, 'eeg_std_uv':float(signal[:,0].std())}
pred_checks = {}
prob_cols = ['p_W','p_N1','p_N2','p_N3','p_REM']
for modality in ['EEG','EOG']:
    path = package / f'3_验证集预测_给D/predictions_val_{modality}.csv'
    pred = rows(path)
    ids = [r['sample_id'] for r in pred]
    assert len(set(ids)) == len(ids) and set(ids) == set(val_truth)
    assert all(int(r['y_true']) == val_truth[r['sample_id']] for r in pred)
    assert all(records[r['record_id']] == 'val' and r['subject_id'] == r['record_id'][:5] for r in pred)
    p = np.array([[float(r[c]) for c in prob_cols] for r in pred])
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    assert np.allclose(p.sum(1),1,atol=1e-5)
    assert np.array_equal(p.argmax(1),np.array([int(r['y_pred']) for r in pred]))
    pred_checks[modality] = {'rows':len(pred),'sample_ids_and_labels_match':True,
                            'probabilities_valid':True,'columns':list(pred[0])}
result={'status':'passed','commit':'02f120ee80c1c7a7822c33746ce92017632f1faa',
        'subject_counts':dict(Counter(r['split'] for r in split)),
        'record_count':len(records),'epoch_counts':dict(counts),'training_sample':sample,
        'prediction_checks':pred_checks,'test_arrays_opened':False,
        'scope':'All train/val label/index arrays; signal values checked for SC4001 only. No baseline retraining.'}
(root/'reports/a_delivery_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
