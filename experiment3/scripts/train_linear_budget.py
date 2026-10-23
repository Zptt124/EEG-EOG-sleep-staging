"""25%-label mean-pooling probe using B's pinned subject lists and official metrics."""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'vendor/experiment2-review-6e7df99'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, default=ROOT/'features/eeg_cond14_v01')
    p.add_argument('--output', type=Path)
    p.add_argument('--sampling-seed', type=int, choices=[2024,2025,2026], required=True)
    p.add_argument('--wait', action='store_true', help='Wait up to 3 hours for full extraction')
    p.add_argument('--epochs', type=int, default=50)
    args = p.parse_args()
    if args.output is None:
        args.output = ROOT/'runs'/f'E3_EEG_label25_sseed{args.sampling_seed}_seed0_v01'
    subset_path = B/'manifests'/('label_subsets_v1.csv' if args.sampling_seed==2024 else f'label_subsets_v1-s{args.sampling_seed}.csv')
    with subset_path.open(encoding='utf-8',newline='') as f:
        subset = list(csv.DictReader(f))
    assert len(subset)==54 and len({r['subject_id'] for r in subset})==54
    assert {int(r['sampling_seed']) for r in subset}=={args.sampling_seed}
    selected = {r['subject_id'] for r in subset if r['label25']=='1'}
    assert len(selected)==14
    spec = importlib.util.spec_from_file_location('b_official',B/'src/evaluation/evaluate.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    deadline = time.monotonic() + 10800
    while not (args.features/'summary.json').exists():
        if not args.wait or time.monotonic() > deadline:
            raise RuntimeError('Full feature extraction has not completed')
        time.sleep(30)
    summary = json.loads((args.features/'summary.json').read_text())
    if summary['status'] != 'complete' or summary['split_counts'] != {'train':132938,'val':21186}:
        raise ValueError('Expected complete A train/val extraction; partial data rejected')
    with (args.features/'feature_index.csv').open(encoding='utf-8',newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 154124 or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid index count or duplicate sample IDs')
    if {r['split'] for r in rows} != {'train','val'}:
        raise ValueError('Unexpected split')
    subjects = {s:{r['subject_id'] for r in rows if r['split']==s} for s in ('train','val')}
    assert not subjects['train'] & subjects['val']
    assert len(subjects['train']) == 54 and len(subjects['val']) == 8
    assert subjects['train']=={r['subject_id'] for r in subset if r['label100']=='1'}
    assert selected <= subjects['train'] and not selected & subjects['val']
    rows = [r for r in rows if r['split']=='val' or r['subject_id'] in selected]
    selected_rows = [r for r in rows if r['split']=='train']
    assert len(selected_rows)=={2024:32310,2025:33775,2026:35203}[args.sampling_seed]
    config_b = B/'reports/runs_E2'/f'config_E2_EEG_label25_sseed{args.sampling_seed}_seed0_v01.yaml'
    import yaml
    bcfg = yaml.safe_load(config_b.read_text())
    counts = {i:sum(int(r['y_true'])==i for r in selected_rows) for i in range(5)}
    assert set(bcfg['label_budget']['labeled_subjects'])==selected
    assert len({r['record_id'] for r in selected_rows})==bcfg['label_budget']['n_labeled_records']
    assert list(counts.values())==list(bcfg['train_class_counts'].values())
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():
        raise RuntimeError('Run already complete; choose a new output directory')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    # CPU head avoids competing with an extraction process that is exiting.
    # Fixed time averaging is linear; all 20 layers and 128 hidden units remain.
    x = np.empty((len(rows),2560),dtype=np.float32)
    grouped = {}
    for i,r in enumerate(rows):
        grouped.setdefault(r['feature_path'],[]).append((i,int(r['row_index'])))
    for name, entries in grouped.items():
        f = np.load(args.features/name,mmap_mode='r',allow_pickle=False)
        if f.shape[1:] != (20,30,128) or f.dtype != np.float16:
            raise ValueError(f'Invalid features: {name}')
        for start in range(0,len(entries),64):
            block = entries[start:start+64]
            data = np.asarray(f[[j for _,j in block]],dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError(f'Nonfinite features: {name}')
            x[[i for i,_ in block]] = data.mean(axis=2).reshape(len(block),2560)
        del f
        print(json.dumps(dict(stage='load_features',record=name)),flush=True)
    mask = np.array([r['split']=='train' for r in rows])
    y = np.array([int(r['y_true']) for r in rows],dtype=np.int64)
    assert set(y.tolist()) == set(range(5))
    mean = x[mask].mean(axis=0,dtype=np.float64).astype(np.float32)
    std = x[mask].std(axis=0,dtype=np.float64).astype(np.float32)
    std = np.maximum(std,1e-6)
    x -= mean
    x /= std
    train_x, val_x = torch.from_numpy(x[mask].copy()), torch.from_numpy(x[~mask].copy())
    train_y = torch.from_numpy(y[mask].copy())
    val_y = y[~mask].copy()
    del x
    cfg = dict(seed=0, sampling_seed=args.sampling_seed, label_fraction=0.25, training_subjects=sorted(selected),
        validation_subjects=sorted(subjects['val']), pooling='mean over 30 time windows, flatten layer/hidden',
        input_dim=2560, classifier='single affine Linear(2560,5), no hidden layer',
        normalization='per-feature mean/std fitted on training only; floor std 1e-6',
        optimizer='AdamW', lr=0.001, weight_decay=0.0001, batch_size=128,
        max_epochs=args.epochs, patience=10, selection='highest validation macro-F1',
        evaluation='B src/evaluation/evaluate.py v1',
        b_commit='6e7df9978c48c84bbef4d2f7e9d66e9ccd93b923',
        label_manifest=str(subset_path.relative_to(ROOT)),
        train_samples=len(train_x),train_class_counts=counts,
        unlabeled_subjects_used=False,
        test_used=False, backbone_frozen=True, device='cpu',
        feature_config=json.loads((args.features/'feature_config.json').read_text()))
    save_json(args.output/'run_config.json',cfg)
    (args.output/'training_subjects_25_from_B.txt').write_text('\n'.join(sorted(selected))+'\n')
    model = torch.nn.Linear(2560,5)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    history, best, stale = [], -1., 0
    start = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train()
        order = torch.randperm(len(train_x))
        total = 0.
        for ids in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(train_x[ids]),train_y[ids])
            loss.backward()
            optimizer.step()
            total += loss.item()*len(ids)
        model.eval()
        with torch.inference_mode():
            pred = torch.cat([model(b).argmax(1) for b in val_x.split(1024)]).numpy()
        score = float(f1_score(val_y,pred,labels=list(range(5)),average='macro',zero_division=0))
        item = dict(epoch=epoch,train_loss=total/len(train_x),val_macro_f1=score)
        history.append(item)
        save_json(args.output/'history.json',history)
        print(json.dumps(item),flush=True)
        if score > best:
            best, stale = score, 0
            torch.save(dict(model=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),
                            epoch=epoch,config=cfg),args.output/'best_head.pt')
        else:
            stale += 1
        if stale >= 10:
            break
    ckpt = torch.load(args.output/'best_head.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['model'])
    with torch.inference_mode():
        probs = torch.cat([model(b).softmax(1) for b in val_x.split(1024)]).numpy()
    pred = probs.argmax(1)
    metrics = official.compute_metrics(val_y,pred)
    val_rows = [r for r in rows if r['split']=='val']
    with (args.output/'predictions_val.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['sample_id','subject_id','record_id','epoch_index','y_true','y_pred','p_W','p_N1','p_N2','p_N3','p_REM'])
        writer.writeheader()
        for i,r in enumerate(val_rows):
            row = {k:r[k] for k in ('sample_id','subject_id','record_id','epoch_index','y_true')}
            row.update(y_pred=int(pred[i]))
            row.update(dict(zip(['p_W','p_N1','p_N2','p_N3','p_REM'],map(float,probs[i]))))
            writer.writerow(row)
    frame = official.load_predictions(args.output/'predictions_val.csv')
    assert np.isfinite(frame[official.PROB_COLS].to_numpy()).all()
    assert frame['sample_id'].nunique()==21186 and frame['subject_id'].nunique()==8
    metrics = official.evaluate(args.output/'predictions_val.csv',n_boot=1000,seed=0)
    metrics['run_id'] = args.output.name
    metrics['b_commit'] = cfg['b_commit']
    save_json(args.output/'metrics_official.json',metrics)
    save_json(args.output/'summary.json',dict(status='complete',evaluation='B official v1',test_used=False,
        sampling_seed=args.sampling_seed,training_subjects=len(selected),
        train_samples=len(train_x),val_samples=len(val_x),best_epoch=ckpt['epoch'],
        val_macro_f1=metrics['macro_f1'],training_seconds=time.perf_counter()-start))


if __name__ == '__main__':
    main()
