"""W2 locked EEG diagnostic matrix: audit/pool S0,S1, then 2x2x5 heads.

No NPZ or backbone is opened. The parent preregistration is read-only.
Full caches must pass identity, split, finite-value and S1 SHA checks first.
"""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml

import validate_adapt_lock as contract

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/week2_eeg_comparison_20261028'
SOURCES = {'S0': ROOT/'features/eeg_cond14_v01',
           'S1': ROOT/'features/eeg_cond14_S1_aseed0_steps10000_v01'}
PREPARATION = ROOT/'reports/week2_next_stage_20261028/extraction_binding.json'
PLAN = ROOT/'reports/week2_adaptation_lock_20261027/downstream_run_plan.csv'
EVALUATOR = ROOT/'vendor/experiment2-review-6e7df99/src/evaluation/evaluate.py'
SUBSETS = ROOT/'manifests/week2_c_adaptation_v01/label_subsets_v2.csv'
SPLIT = ROOT/'vendor/experiment1-review/manifests/split_subjects.csv'
EXPECTED = {'train': 132938, 'val': 21186}
sha = contract.sha
require = contract.require
read_csv = contract.csv_rows


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def official_module():
    spec = importlib.util.spec_from_file_location('official_b', EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_index(rows, splits, strict=True):
    require(bool(rows), 'Empty feature index')
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate sample_id')
    partitions = {}
    for row in splits:
        for record in row['record_ids'].split(';'):
            partitions[record] = (row['subject_id'], row['split'])
    grouped = defaultdict(list)
    subjects = defaultdict(set)
    for position, row in enumerate(rows):
        rec = row['record_id']
        require(row['split'] in ('train', 'val'), 'Test/unknown split forbidden')
        require(partitions.get(rec) == (row['subject_id'], row['split']), 'Split identity mismatch')
        require(row['sample_id'] == f'{rec}_{int(row["epoch_index"]):04d}', 'Sample identity mismatch')
        require(row['feature_path'] == f'{rec}.npy', 'Unsafe feature path')
        require(int(row['y_true']) in range(5), 'Invalid stage label')
        grouped[rec].append((position, row))
        subjects[row['split']].add(row['subject_id'])
    require(not subjects['train'] & subjects['val'], 'Overlapping train/val subjects')
    for items in grouped.values():
        require([int(r['row_index']) for _, r in items] == list(range(len(items))), 'Noncanonical feature row order')
        require(len({r['epoch_index'] for _, r in items}) == len(items), 'Duplicate epoch')
    if strict:
        require(Counter(r['split'] for r in rows) == EXPECTED, 'Incomplete feature index')
        require(len(grouped) == 122 and len(subjects['train']) == 54 and len(subjects['val']) == 8,
                'Record/subject coverage mismatch')
        require(set(grouped) == {r for r, (_, s) in partitions.items() if s in ('train', 'val')},
                'Missing train/val record')
    return dict(grouped)


def align_indices(s0, s1):
    require(len(s0) == len(s1), 'S0/S1 index lengths differ')
    keys = ['sample_id', 'subject_id', 'record_id', 'epoch_index', 'split', 'y_true', 'feature_path', 'row_index']
    require(all(all(a[k] == b[k] for k in keys) for a, b in zip(s0, s1)), 'S0/S1 index alignment differs')


def selected_positions(rows, subset_rows, budget, strict=True):
    require(budget in ('label10', 'label25'), 'Unregistered label budget')
    selected = {r['subject_id'] for r in subset_rows if r[budget] == '1'}
    require({int(r['sampling_seed']) for r in subset_rows} == {2024}, 'Sampling seed changed')
    train_subjects = {r['subject_id'] for r in rows if r['split'] == 'train'}
    require(selected <= train_subjects, 'Budget includes a non-training subject')
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train' and r['subject_id'] in selected])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    require(len(train) > 0 and len(val) > 0, 'Empty train or val selection')
    if strict:
        nsubjects, nsamples, nrecords = {'label10': (5, 11936, 10), 'label25': (14, 32310, 27)}[budget]
        require(len(selected) == nsubjects and len(train) == nsamples, 'Budget coverage mismatch')
        require(len({rows[i]['record_id'] for i in train}) == nrecords, 'Budget record count differs')
    return selected, train, val


def fit_normalization(train):
    require(np.isfinite(train).all(), 'Nonfinite training features')
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(train.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    return mean, std


def pool_block(raw):
    require(raw.ndim == 4 and raw.shape[1:] == (20, 30, 128) and raw.dtype == np.float16,
            'Wrong raw feature shape/dtype')
    values = np.asarray(raw, dtype=np.float32)
    require(np.isfinite(values).all(), 'Nonfinite raw feature')
    return values.mean(axis=2, dtype=np.float32).reshape(len(raw), 2560)


def new_head(seed, dimension=2560):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Linear(dimension, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    return model, optimizer


def selection_update(score, best, stale):
    require(np.isfinite(score), 'Nonfinite selection metric')
    improved = score > best
    return improved, max(score, best), 0 if improved else stale+1


def prepare_binding(path):
    contract.validate_locked_files()
    files = [Path(__file__), ROOT/'tests/test_s0_s1_eeg_comparison.py', Path(contract.__file__),
             contract.DEFAULT_CONFIG, contract.DEFAULT_CONFIG.with_suffix('.lock.json'),
             PREPARATION, PLAN, SUBSETS, SPLIT, EVALUATOR]
    files += [folder/'feature_config.json' for folder in SOURCES.values()]
    files += [folder/'feature_index.csv' for folder in SOURCES.values()]
    for folder in SOURCES.values():
        summary = json.loads((folder/'summary.json').read_text())
        require(summary['status'] == 'complete' and summary['split_counts'] == EXPECTED,
                'Full S0/S1 extraction must complete before downstream binding')
    require(not path.exists(), 'Do not overwrite a runtime binding')
    value = dict(status='bound_before_head_training', scope='20_EEG_diagnostic_heads_only',
                 input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
                 test_used=False, EOG_cache_received=False)
    write_json(path, value)
    return value


def validate_binding(path):
    bound = json.loads(path.read_text())
    require(bound['status'] == 'bound_before_head_training', 'Invalid runtime binding')
    for relative, expected in bound['input_sha256'].items():
        require(sha(contract.resolve_path(relative)) == expected, f'Downstream bound input changed: {relative}')
    contract.validate_locked_files()
    return bound


def audit_and_pool(representation, rows, grouped, output):
    folder = SOURCES[representation]
    summary = json.loads((folder/'summary.json').read_text())
    require(summary['status'] == 'complete' and summary['split_counts'] == EXPECTED and not summary['test_used'],
            'Incomplete/unsafe source cache')
    preparation = json.loads(PREPARATION.read_text())
    cfg = json.loads((folder/'feature_config.json').read_text())
    if representation == 'S0':
        require(sha(folder/'feature_index.csv') == preparation['source_index_sha256'], 'S0 reference index changed')
        require(sha(folder/'feature_config.json') == preparation['source_S0_config_sha256'], 'S0 config changed')
    else:
        require(cfg['checkpoint_sha256'] == summary['selected_weight_sha256'] == preparation['selected_weight_sha256'],
                'S1 selected weight differs')
        require(cfg['representation'] == 'S1' and cfg['adaptation_steps'] == 10000
                and cfg['weight_role'] == 'selected_new_ema', 'S1 weight role differs')
        require(json.loads((folder/'extraction_binding.json').read_text()) == preparation, 'S1 provenance binding differs')
    target = output/f'{representation}_pooled.npy'
    partial = target.with_suffix('.partial.npy')
    require(not target.exists() and not partial.exists(), 'Pooled output already exists')
    destination = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(len(rows), 2560))
    profiles = []
    for rec, entries in grouped.items():
        source = folder/f'{rec}.npy'
        digest = sha(source)
        if representation == 'S1':
            receipt = json.loads((folder/f'{rec}.done.json').read_text())
            require(receipt['record_id'] == rec and receipt['n_samples'] == len(entries)
                    and receipt['sha256'] == digest and receipt['weight_sha256'] == preparation['selected_weight_sha256'],
                    'S1 per-record checksum/identity mismatch')
        array = np.load(source, mmap_mode='r', allow_pickle=False)
        require(array.shape == (len(entries), 20, 30, 128) and array.dtype == np.float16, 'Raw shape/dtype mismatch')
        for start in range(0, len(entries), 64):
            block = entries[start:start+64]
            destination[[i for i, _ in block]] = pool_block(array[start:start+len(block)])
        del array
        profiles.append(dict(record_id=rec, samples=len(entries), sha256=digest))
        print(json.dumps(dict(stage='audit_pool', representation=representation, record=rec,
                              records_done=len(profiles), records_total=len(grouped))), flush=True)
    destination.flush()
    del destination
    partial.replace(target)
    report = dict(status='passed', representation=representation, records=len(grouped), samples=len(rows),
                  split_counts=dict(Counter(r['split'] for r in rows)), raw_files=profiles,
                  raw_all_finite=True, pooled_shape=[len(rows), 2560], pooled_dtype='float32',
                  pooling='float32 mean over 30 time windows then flatten', pooled_sha256=sha(target),
                  source_index_sha256=sha(folder/'feature_index.csv'), source_config_sha256=sha(folder/'feature_config.json'),
                  test_used=False, NPZ_opened=False)
    write_json(output/f'{representation}_feature_audit.json', report)
    return target


def predict(model, x, probabilities=False):
    with torch.inference_mode():
        if probabilities:
            return torch.cat([model(b).softmax(1) for b in x.split(1024)]).numpy()
        return torch.cat([model(b).argmax(1) for b in x.split(1024)]).numpy()


def train_head(plan, rows, train_x, train_y, val_x, val_y, mean, std, context, official):
    output = ROOT/'runs'/plan['run_id']
    output.mkdir(parents=True, exist_ok=False)
    seed = int(plan['downstream_seed'])
    model, optimizer = new_head(seed)
    cfg = dict(context, run_id=plan['run_id'], representation=plan['representation'],
               adaptation_seed=0 if plan['representation'] == 'S1' else None,
               downstream_seed=seed, sampling_seed=2024, budget=plan['budget'], modality='EEG',
               role='diagnostic', input_dim=2560, classifier='Linear(2560,5)',
               optimizer='AdamW', learning_rate=0.001, weight_decay=0.0001,
               batch_size=128, max_epochs=50, patience=10, loss='unweighted_cross_entropy',
               selection='strictly highest development validation macro-F1',
               normalization='fit only current budget labeled train subjects, independently for S0/S1',
               std_floor=1e-6, backbone_frozen=True, test_used=False, device='cpu',
               cpu_threads=4, deterministic=True, N1_REM_from_same_selected_head=True)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    history, best, stale = [], -1., 0
    start_time = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        total = 0.
        for ids in torch.randperm(len(train_x)).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(train_x[ids]), train_y[ids])
            require(bool(torch.isfinite(loss)), 'Nonfinite classifier loss')
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                    'Nonfinite/missing classifier gradient')
            optimizer.step()
            total += loss.item()*len(ids)
        model.eval()
        pred = predict(model, val_x)
        score = official.compute_metrics(val_y, pred)['macro_f1']
        history.append(dict(epoch=epoch, train_loss=total/len(train_x), val_macro_f1=score))
        write_json(output/'history.json', history)
        improved, best, stale = selection_update(score, best, stale)
        if improved:
            torch.save(dict(model=model.state_dict(), mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                            epoch=epoch, config=cfg), output/'best_head.pt')
        print(json.dumps(dict(stage='train_head', run_id=plan['run_id'], **history[-1])), flush=True)
        if stale >= 10:
            break
    checkpoint = torch.load(output/'best_head.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    probs = predict(model, val_x, True)
    require(np.isfinite(probs).all(), 'Nonfinite validation probabilities')
    predictions = []
    for row, probability in zip(rows, probs):
        result = {k: row[k] for k in ('sample_id', 'subject_id', 'record_id', 'epoch_index', 'y_true')}
        result['y_pred'] = int(probability.argmax())
        result.update(zip(official.PROB_COLS, map(float, probability)))
        predictions.append(result)
    write_csv(output/'predictions_val.csv', predictions)
    frame = official.load_predictions(output/'predictions_val.csv')
    require(len(frame) == 21186 and frame['subject_id'].nunique() == 8, 'Validation prediction coverage differs')
    metrics = official.evaluate(output/'predictions_val.csv', n_boot=1000, seed=0)
    write_json(output/'metrics.json', metrics)
    official.per_subject_table(frame).to_csv(output/'per_subject_metrics.csv', index=False)
    write_csv(output/'confusion_matrix.csv', [dict(true_stage=stage, **dict(zip(official.STAGE_NAMES, counts)))
              for stage, counts in zip(official.STAGE_NAMES, metrics['confusion_matrix'])])
    elapsed = time.perf_counter()-start_time
    write_csv(output/'resource_profile.csv', [dict(device='cpu', threads=4, seconds=elapsed,
              epochs=len(history), train_samples=len(train_x), val_samples=len(val_x), test_used=False)])
    result = dict(status='complete', run_id=plan['run_id'], representation=plan['representation'], budget=plan['budget'],
                  seed=seed, epochs=len(history), best_epoch=checkpoint['epoch'], val_macro_f1=metrics['macro_f1'],
                  n1_f1=metrics['n1_f1'], rem_f1=metrics['per_class_f1']['REM'], seconds=elapsed, test_used=False)
    (output/'run_summary.md').write_text(
        f"# {plan['run_id']}\n\nEEG-only diagnostic; fixed adaptation seed 0 for S1. "
        f"Best epoch {checkpoint['epoch']}; validation Macro-F1 {metrics['macro_f1']:.8f}, "
        f"N1 F1 {metrics['n1_f1']:.8f}, REM F1 {metrics['per_class_f1']['REM']:.8f}. "
        "All scores use the same selected head. Test unused.\n", encoding='utf-8')
    write_json(output/'artifact_sha256.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'summary.json', result)
    return result


def paired_report(results, output, official):
    paired = []
    for budget in ('label10', 'label25'):
        for seed in range(5):
            same = [r for r in results if r['budget'] == budget and r['seed'] == seed]
            by_rep = {r['representation']: r for r in same}
            if set(by_rep) != {'S0', 'S1'}:
                continue
            paths = {rep: ROOT/'runs'/r['run_id']/'predictions_val.csv' for rep, r in by_rep.items()}
            delta = official.paired_bootstrap_delta(official.load_predictions(paths['S1']),
                                                   official.load_predictions(paths['S0']), n_boot=1000, seed=0)
            write_json(output/f'paired_{budget}_seed{seed}.json', delta)
            paired.append(dict(budget=budget, seed=seed, delta_macro_f1=delta['macro_f1']['point'],
                               delta_n1_f1=delta['n1_f1']['point'],
                               delta_rem_f1=by_rep['S1']['rem_f1']-by_rep['S0']['rem_f1']))
    if paired:
        write_csv(output/'paired_deltas.csv', paired)
    aggregates = []
    for budget in ('label10', 'label25'):
        for representation in ('S0', 'S1'):
            group = [r for r in results if r['budget'] == budget and r['representation'] == representation]
            if not group:
                continue
            item = dict(budget=budget, representation=representation, completed_seeds=len(group))
            for metric in ('val_macro_f1', 'n1_f1', 'rem_f1'):
                values = [r[metric] for r in group]
                item[metric+'_mean'] = float(np.mean(values))
                item[metric+'_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
            aggregates.append(item)
    if aggregates:
        write_csv(output/'aggregate_metrics.csv', aggregates)


def run(binding_path, output):
    bound = validate_binding(binding_path)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'.run.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = dict(status='running', stage='audit_and_pool', completed_heads=0, expected_heads=20,
                        test_used=False, EOG_status='waiting_for_D_original_cache', pid=os.getpid())
        write_json(output/'summary.json', progress)
        try:
            indices = {rep: read_csv(folder/'feature_index.csv') for rep, folder in SOURCES.items()}
            splits = read_csv(SPLIT)
            grouped = {rep: audit_index(rows, splits) for rep, rows in indices.items()}
            align_indices(indices['S0'], indices['S1'])
            pooled = {rep: audit_and_pool(rep, indices[rep], grouped[rep], output) for rep in ('S0', 'S1')}
            subset_rows = read_csv(SUBSETS)
            plans = [r for r in read_csv(PLAN) if r['modality'] == 'EEG']
            require(len(plans) == 20 and {(r['representation'], r['budget'], int(r['downstream_seed'])) for r in plans}
                    == {(rep, b, s) for rep in ('S0', 'S1') for b in ('label10', 'label25') for s in range(5)},
                    'EEG matrix changed')
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            official = official_module()
            results = []
            rows = indices['S0']
            # Pair S0/S1 consecutively for every seed; all 20 positions are unconditional.
            for budget in ('label10', 'label25'):
                selected, train_ids, val_ids = selected_positions(rows, subset_rows, budget)
                val_rows = [rows[i] for i in val_ids]
                for seed in range(5):
                    for rep in ('S0', 'S1'):
                        raw = np.load(pooled[rep], mmap_mode='r', allow_pickle=False)
                        train = np.array(raw[train_ids], copy=True)
                        val = np.array(raw[val_ids], copy=True)
                        mean, std = fit_normalization(train)
                        train -= mean; train /= std
                        val -= mean; val /= std
                        require(np.isfinite(train).all() and np.isfinite(val).all(), 'Nonfinite normalized features')
                        train_x, val_x = torch.from_numpy(train), torch.from_numpy(val)
                        train_y = torch.tensor([int(rows[i]['y_true']) for i in train_ids], dtype=torch.int64)
                        val_y = np.array([int(r['y_true']) for r in val_rows])
                        plan = next(p for p in plans if p['representation'] == rep and p['budget'] == budget
                                    and int(p['downstream_seed']) == seed)
                        progress.update(stage='head_training', current_run=plan['run_id'])
                        write_json(output/'summary.json', progress)
                        context = dict(training_subjects=sorted(selected), training_samples=len(train_ids),
                                       validation_subjects=sorted({r['subject_id'] for r in val_rows}),
                                       validation_samples=len(val_ids), train_class_counts=dict(Counter(map(int, train_y.tolist()))),
                                       feature_audit_sha256=sha(output/f'{rep}_feature_audit.json'),
                                       runtime_binding_sha256=sha(binding_path), trainer_sha256=sha(__file__),
                                       evaluator_sha256=sha(EVALUATOR), label_manifest_sha256=sha(SUBSETS),
                                       local_unlabeled_adaptation_subjects=54 if rep == 'S1' else 0)
                        result = train_head(plan, val_rows, train_x, train_y, val_x, val_y, mean, std, context, official)
                        results.append(result)
                        write_csv(output/'run_registry.csv', results)
                        progress.update(completed_heads=len(results))
                        write_json(output/'summary.json', progress)
                        paired_report(results, output, official)
                        del raw, train, val, train_x, val_x, train_y
            validate_binding(binding_path)
            progress.update(status='complete', stage='EEG_diagnostic_matrix_complete', current_run=None,
                            independent_adaptation_repeats=1, downstream_seeds_per_cell=5)
        except BaseException:
            import traceback
            progress.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            write_json(output/'summary.json', progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run'], required=True)
    parser.add_argument('--binding', type=Path, default=REPORT/'runtime_binding.json')
    parser.add_argument('--output', type=Path, default=REPORT/'execution')
    args = parser.parse_args()
    if args.mode == 'prepare':
        args.binding.parent.mkdir(parents=True, exist_ok=True)
        result = prepare_binding(args.binding)
    else:
        result = run(args.binding, args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
