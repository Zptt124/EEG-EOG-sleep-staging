"""Versioned repeat runtime. Numerical extraction/head functions match the pilot.

Only adaptation-seed metadata and the head seed argument differ. Source-parity
unit tests protect the unchanged arithmetic and selection rules.
"""
import json
import time
import numpy as np
import torch
import yaml
import extract_adapted_features as extraction
import run_s0_s1_eeg_comparison as heads

ROOT = heads.ROOT
sha = heads.sha
require = heads.require
write_json = heads.write_json
write_csv = heads.write_csv
new_head = heads.new_head
predict = heads.predict
selection_update = heads.selection_update
EXPECTED = heads.EXPECTED
pool_block = heads.pool_block
legacy = extraction.legacy
reference = extraction.reference
contract = extraction.contract
AdaptedExtractor = extraction.AdaptedExtractor
load_signal_record = extraction.load_signal_record
atomic_json = extraction.atomic_json
Counter = extraction.Counter


def extract_locked(config,preparation,model,index,grouped,partitions,out,resume):
    if resume:
        require(out.is_dir(),'No extraction to resume')
        old=json.loads((out/'extraction_binding.json').read_text(encoding='utf-8'))
        require(old==preparation,'Existing extraction weight/config/code differs')
        old_summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        require(old_summary['status']!='complete','Extraction is already complete')
    else:
        atomic_json(out/'extraction_binding.json',preparation)
    cfg=legacy.config('eeg',14)
    cfg.update(version=config['config_id'],checkpoint_sha256=preparation['selected_weight_sha256'],
        representation='S1',adaptation_seed=config['adaptation_seed'],adaptation_steps=10000,weight_role='selected_new_ema',
        source_index_sha256=preparation['source_index_sha256'],extractor_script_sha256=sha(__file__),
        cache_policy='fresh_frozen_kernels_after_selected_weights_load')
    atomic_json(out/'feature_config.json',cfg)
    summary=dict(status='running',records=0,samples=0,test_used=False,full_extraction_started=True,
        expected_records=122,expected_samples=154124,selected_weight_sha256=preparation['selected_weight_sha256'])
    atomic_json(out/'summary.json',summary)
    torch.manual_seed(0)
    extractor=AdaptedExtractor(model,14)
    extractor.cache_frozen_kernels()
    profiles=[]; completed_rows=[]
    try:
        for rec,expected in sorted(grouped.items()):
            target=out/f'{rec}.npy'; marker=out/f'{rec}.done.json'
            if marker.exists():
                profile=json.loads(marker.read_text(encoding='utf-8'))
                require(profile['weight_sha256']==preparation['selected_weight_sha256'] and sha(target)==profile['sha256'],
                        'Completed record hash/weight mismatch')
                array=np.load(target,mmap_mode='r',allow_pickle=False)
                require(array.shape==(len(expected),20,30,128) and array.dtype==np.float16,'Completed record shape/dtype mismatch')
                del array
            else:
                signals=load_signal_record(contract.resolve_path(config['source_data_root']),rec,partitions,expected)
                partial=out/f'{rec}.partial.npy'
                destination=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float16,shape=(len(expected),20,30,128))
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                start=time.perf_counter()
                for first in range(0,len(expected),config['batch_size']):
                    features=extractor(signals[first:first+config['batch_size']]).astype(np.float16)
                    require(bool(np.isfinite(features).all()),'Nonfinite float16 features')
                    destination[first:first+len(features)]=features
                destination.flush(); del destination,signals
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                partial.replace(target)
                profile=dict(record_id=rec,split=expected[0]['split'],n_samples=len(expected),seconds=elapsed,
                    samples_per_second=len(expected)/elapsed,peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20,
                    bytes=target.stat().st_size,batch_size=config['batch_size'],cache_frozen_kernels=True,
                    weight_sha256=preparation['selected_weight_sha256'],sha256=sha(target))
                atomic_json(marker,profile)
            profiles.append(profile)
            for old in expected:
                new=dict(old); new['config_id']=config['config_id']; completed_rows.append(new)
            reference.write_csv(out/'feature_index.csv',completed_rows)
            reference.write_csv(out/'resource_profile.csv',profiles)
            summary.update(records=len(profiles),samples=len(completed_rows))
            atomic_json(out/'summary.json',summary)
            print(json.dumps(profile),flush=True)
        require(len(completed_rows)==154124,'Incomplete extracted sample coverage')
        require({r['sample_id'] for r in completed_rows}=={r['sample_id'] for r in index},'S1 sample set differs from S0')
        summary.update(status='complete',records=122,samples=154124,split_counts=dict(Counter(r['split'] for r in completed_rows)),
            total_feature_bytes=sum(p['bytes'] for p in profiles),S0_overwritten=False,NPZ_labels_read=False)
    except BaseException:
        import traceback
        summary.update(status='incomplete',error=traceback.format_exc())
        raise
    finally:
        atomic_json(out/'summary.json',summary)
    return summary



def train_head(plan, rows, train_x, train_y, val_x, val_y, mean, std, context, official, adaptation_seed):
    output = ROOT/'runs'/plan['run_id']
    output.mkdir(parents=True, exist_ok=False)
    seed = int(plan['downstream_seed'])
    model, optimizer = new_head(seed)
    cfg = dict(context, run_id=plan['run_id'], representation=plan['representation'],
               adaptation_seed=adaptation_seed if plan['representation'] == 'S1' else None,
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
        f"# {plan['run_id']}\n\nEEG-only diagnostic; fixed adaptation seed {adaptation_seed} for S1. "
        f"Best epoch {checkpoint['epoch']}; validation Macro-F1 {metrics['macro_f1']:.8f}, "
        f"N1 F1 {metrics['n1_f1']:.8f}, REM F1 {metrics['per_class_f1']['REM']:.8f}. "
        "All scores use the same selected head. Test unused.\n", encoding='utf-8')
    write_json(output/'artifact_sha256.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'summary.json', result)
    return result



def audit_and_pool(representation, rows, grouped, output, folder, preparation_path):
    summary = json.loads((folder/'summary.json').read_text())
    require(summary['status'] == 'complete' and summary['split_counts'] == EXPECTED and not summary['test_used'],
            'Incomplete/unsafe source cache')
    preparation = json.loads(preparation_path.read_text())
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

