"""Primary EEG+EOG comparison, four EEG states with one original frozen EOG.

The EOG cache is explicitly a local reconstruction of D's reference protocol.
No assertion is made that unavailable D raw feature bytes were received.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

import run_s0_s1_eeg_comparison as heads
import replication_feature_head_runtime as runtime
import train_primary_fusion_head as fusion_head

ROOT = heads.ROOT
REPORT = ROOT/'reports/week2_fusion_20261029'
require, sha, read_csv, write_json = heads.require, heads.sha, heads.read_csv, heads.write_json


def validate_config(cfg):
    require(cfg['budgets'] == ['label10', 'label25'] and cfg['downstream_seeds'] == list(range(5)), 'Budget/seed matrix changed')
    require(cfg['adaptation_seeds'] == [0, 1, 2] and cfg['planned_new_heads'] == 40, 'Backbone matrix changed')
    require(cfg['input_dim'] == 5120 and cfg['feature_order'] == ['EEG', 'EOG'], 'Fusion dimension/order changed')
    require(not cfg['test_used'] and not cfg['D_full_cache_received'], 'Test/provenance policy differs')
    require(cfg['EOG_source'] == 'local_regeneration_using_identical_D_reference_configuration'
            and cfg['D_raw_file_byte_equivalence_unverified'], 'Local EOG provenance must remain explicit')
    expected = dict(name='Linear(5120,5)', optimizer='AdamW', learning_rate=0.001, weight_decay=0.0001,
                    batch_size=128, max_epochs=50, patience=10, loss='unweighted_cross_entropy',
                    selection='strictly highest development validation macro-F1',
                    normalization='per-feature fitted only current labeled training subjects', std_floor=1e-6)
    require(cfg['classifier'] == expected, 'Locked classifier method changed')
    require([(s['key'], s['adaptation_seed'], s['representation']) for s in cfg['EEG_sources']]
            == [('S0', None, 'S0'), ('S1_aseed0', 0, 'S1'), ('S1_aseed1', 1, 'S1'), ('S1_aseed2', 2, 'S1')],
            'EEG source matrix changed')
    for source in cfg['EEG_sources']:
        for key in ['feature_root', 'pooled', 'audit']:
            heads.contract.resolve_path(source[key])
    return True


def plan(cfg):
    result = []
    for budget in cfg['budgets']:
        for seed in cfg['downstream_seeds']:
            for source in cfg['EEG_sources']:
                prefix = 'W2_C_S0_EEG_EOG' if source['representation'] == 'S0' else f"W2_C_S1_EEG_EOG_aseed{source['adaptation_seed']}"
                result.append(dict(run_id=f'{prefix}_{budget}_sseed2024_seed{seed}_v01', representation=source['representation'],
                                   adaptation_seed=source['adaptation_seed'], source_key=source['key'], budget=budget,
                                   downstream_seed=seed, modality='EEG_EOG', role='primary'))
    require(len(result) == len({r['run_id'] for r in result}) == 40, 'Duplicate/missing fusion positions')
    return result


def concatenate(eeg, eog):
    require(eeg.ndim == eog.ndim == 2 and eeg.shape == eog.shape and eeg.shape[1] == 2560, 'Invalid paired pooled shape')
    require(eeg.dtype == eog.dtype == np.float32, 'Pooled computation must use float32')
    return np.concatenate([eeg, eog], axis=1)


def binding_files(cfg):
    files = [Path(__file__), Path(fusion_head.__file__), Path(runtime.__file__), Path(heads.__file__),
             ROOT/'tests/test_primary_fusion.py', REPORT/'runtime_tests.log', REPORT/'fusion_config.json',
             REPORT/'EOG_source_binding.json', REPORT/'EOG_launch.json', REPORT/'EOG_protocol_preflight.json',
             REPORT/'EOG_probe/probe_audit.json', heads.SUBSETS, heads.SPLIT, heads.EVALUATOR,
             ROOT/cfg['parent_config'], (ROOT/cfg['parent_config']).with_suffix('.lock.json')]
    for source in cfg['EEG_sources']:
        folder = ROOT/source['feature_root']
        files += [ROOT/source['audit'], folder/'feature_index.csv', folder/'feature_config.json']
    return files


def bind():
    cfg = json.loads((REPORT/'fusion_config.json').read_text())
    validate_config(cfg)
    heads.contract.validate_locked_files()
    tests = (REPORT/'runtime_tests.log').read_text()
    require('Ran 8 tests' in tests and '\nOK\n' in tests, 'Fusion tests did not pass')
    for row in plan(cfg):
        require(not (ROOT/'runs'/row['run_id']).exists(), 'Existing fusion head would be overwritten')
    path = REPORT/'runtime_binding.json'
    require(not path.exists(), 'Do not replace a fusion runtime binding')
    files = binding_files(cfg)
    value = dict(status='bound_before_primary_classification', expected_heads=40, test_used=False,
                 input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files})
    write_json(path, value)
    heads.write_csv(REPORT/'primary_run_plan.csv', plan(cfg))
    return value


def check_binding():
    value = json.loads((REPORT/'runtime_binding.json').read_text())
    require(value['status'] == 'bound_before_primary_classification', 'Invalid fusion binding')
    for relative, expected in value['input_sha256'].items():
        require(sha(ROOT/relative) == expected, f'Fusion input changed: {relative}')
    original = json.loads((REPORT/'EOG_source_binding.json').read_text())
    for relative, expected in original['input_sha256'].items():
        require(sha(ROOT/relative) == expected, f'Original EOG extraction input changed: {relative}')


def wait_for_eog(cfg, status):
    folder = ROOT/cfg['EOG_root']
    launch = json.loads((REPORT/'EOG_launch.json').read_text())
    previous = -1
    while True:
        summary = folder/'summary.json'
        if summary.exists():
            value = json.loads(summary.read_text())
            require(value['status'] == 'complete' and value['records'] == 122 and value['samples'] == 154124
                    and value['split_counts'] == heads.EXPECTED and not value['test_used'], 'Incomplete EOG cache')
            status.update(EOG_records=122, EOG_samples=154124)
            write_json(REPORT/'summary.json', status)
            return folder
        os.kill(launch['pid'], 0)
        receipts = []
        for path in folder.glob('*.done.json'):
            try:
                receipts.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                pass
        if len(receipts) != previous:
            status.update(EOG_records=len(receipts), EOG_samples=sum(r['n_samples'] for r in receipts))
            write_json(REPORT/'summary.json', status)
            previous = len(receipts)
        time.sleep(20)


def verify_eeg_pools(cfg, canonical, execution):
    paths, checks = {}, []
    for source in cfg['EEG_sources']:
        folder, audit_path, pooled = ROOT/source['feature_root'], ROOT/source['audit'], ROOT/source['pooled']
        audit = json.loads(audit_path.read_text())
        require(audit['status'] == 'passed' and audit['samples'] == 154124 and audit['raw_all_finite'], 'Prior EEG audit failed')
        require(sha(folder/'feature_index.csv') == audit['source_index_sha256']
                and sha(folder/'feature_config.json') == audit['source_config_sha256'], 'EEG source metadata changed')
        rows = read_csv(folder/'feature_index.csv')
        heads.audit_index(rows, read_csv(heads.SPLIT))
        heads.align_indices(canonical, rows)
        feature_cfg = json.loads((folder/'feature_config.json').read_text())
        if source['representation'] == 'S1':
            require(feature_cfg['adaptation_seed'] == source['adaptation_seed'] and feature_cfg['adaptation_steps'] == 10000,
                    'Wrong adapted EEG source')
        require(sha(pooled) == audit['pooled_sha256'], 'EEG pooled cache checksum mismatch')
        values = np.load(pooled, mmap_mode='r', allow_pickle=False)
        require(values.shape == (154124, 2560) and values.dtype == np.float32, 'Invalid EEG pooled cache')
        del values
        paths[source['key']] = pooled
        checks.append(dict(source_key=source['key'], pooled_sha256=audit['pooled_sha256'], samples=154124,
                           source_metadata_match=True, previous_full_finite_audit_verified_by_cache_SHA=True))
        print(json.dumps(dict(stage='verify_existing_EEG_pool', source=source['key'])), flush=True)
    write_json(execution/'EEG_pooled_reuse_audit.json', dict(status='passed', sources=checks))
    return paths


def run():
    check_binding()
    cfg = json.loads((REPORT/'fusion_config.json').read_text())
    validate_config(cfg)
    execution = REPORT/'execution'
    execution.mkdir(exist_ok=False)
    with (REPORT/'.primary.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(status='running', stage='EOG_full_feature_extraction', coordinator_pid=os.getpid(), EOG_records=0,
                      EOG_samples=0, expected_records=122, expected_samples=154124,
                      completed_heads=0, expected_heads=40, test_used=False, D_full_cache_received=False)
        write_json(REPORT/'summary.json', status)
        try:
            eog_folder = wait_for_eog(cfg, status)
            check_binding()
            status.update(stage='EOG_cache_audit_and_pool')
            write_json(REPORT/'summary.json', status)
            d_folder = ROOT/cfg['D_metadata']
            require(json.loads((eog_folder/'feature_config.json').read_text()) == json.loads((d_folder/'feature_config.json').read_text()),
                    'Reconstructed EOG settings differ from D reference')
            canonical = read_csv(heads.SOURCES['S0']/'feature_index.csv')
            eog_rows = read_csv(eog_folder/'feature_index.csv')
            grouped = heads.audit_index(eog_rows, read_csv(heads.SPLIT))
            heads.align_indices(canonical, eog_rows)
            heads.align_indices(read_csv(d_folder/'feature_index.csv'), eog_rows)
            eog_audit_folder = execution/'EOG_audit'
            eog_audit_folder.mkdir()
            preparation = eog_audit_folder/'pool_source.json'
            write_json(preparation, dict(source_index_sha256=sha(eog_folder/'feature_index.csv'),
                       source_S0_config_sha256=sha(eog_folder/'feature_config.json'),
                       modality='EOG', weight_role='original_frozen_EMA', D_full_cache_received=False))
            # Here S0 denotes the always-original EOG state, in its own EOG folder.
            eog_pooled = runtime.audit_and_pool('S0', eog_rows, grouped, eog_audit_folder, eog_folder, preparation)
            status.update(stage='verify_existing_EEG_pools')
            write_json(REPORT/'summary.json', status)
            eeg_paths = verify_eeg_pools(cfg, canonical, execution)
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            official = heads.official_module()
            subsets, results = read_csv(heads.SUBSETS), []
            eog = np.load(eog_pooled, mmap_mode='r', allow_pickle=False)
            for entry in plan(cfg):
                selected, train_ids, val_ids = heads.selected_positions(canonical, subsets, entry['budget'])
                eeg = np.load(eeg_paths[entry['source_key']], mmap_mode='r', allow_pickle=False)
                train = concatenate(eeg[train_ids], eog[train_ids])
                val = concatenate(eeg[val_ids], eog[val_ids])
                mean, std = heads.fit_normalization(train)
                train -= mean; train /= std
                val -= mean; val /= std
                require(np.isfinite(train).all() and np.isfinite(val).all(), 'Nonfinite fusion input after normalization')
                train_x, val_x = torch.from_numpy(train), torch.from_numpy(val)
                train_y = torch.tensor([int(canonical[i]['y_true']) for i in train_ids], dtype=torch.int64)
                val_rows = [canonical[i] for i in val_ids]
                val_y = np.array([int(r['y_true']) for r in val_rows])
                source = next(s for s in cfg['EEG_sources'] if s['key'] == entry['source_key'])
                context = dict(training_subjects=sorted(selected), training_samples=len(train_ids),
                               validation_subjects=sorted({r['subject_id'] for r in val_rows}), validation_samples=len(val_ids),
                               train_class_counts=dict(Counter(map(int, train_y.tolist()))),
                               EEG_feature_audit_sha256=sha(ROOT/source['audit']),
                               EOG_feature_audit_sha256=sha(eog_audit_folder/'S0_feature_audit.json'),
                               runtime_binding_sha256=sha(REPORT/'runtime_binding.json'), trainer_sha256=sha(fusion_head.__file__),
                               evaluator_sha256=sha(heads.EVALUATOR), label_manifest_sha256=sha(heads.SUBSETS),
                               local_unlabeled_adaptation_subjects=0 if entry['representation']=='S0' else 54,
                               EOG_source=cfg['EOG_source'], D_full_cache_received=False, feature_order=['EEG','EOG'])
                status.update(stage='primary_head_training', current_run=entry['run_id'])
                write_json(REPORT/'summary.json', status)
                result = fusion_head.train_head(entry, val_rows, train_x, train_y, val_x, val_y, mean, std, context,
                                                official, adaptation_seed=entry['adaptation_seed'])
                result.update(adaptation_seed=entry['adaptation_seed'], source_key=entry['source_key'], modality='EEG_EOG')
                results.append(result)
                heads.write_csv(execution/'run_registry.csv', results)
                status.update(completed_heads=len(results))
                write_json(REPORT/'summary.json', status)
                del eeg, train, val, train_x, val_x, train_y
            status.update(stage='paired_primary_analysis', current_run=None)
            write_json(REPORT/'summary.json', status)
            for seed in cfg['adaptation_seeds']:
                out = execution/f'comparison_aseed{seed}'
                out.mkdir()
                subset = [r for r in results if r['representation'] == 'S0' or r['adaptation_seed'] == seed]
                heads.paired_report(subset, out, official)
            check_binding()
            status.update(status='complete', stage='primary_fusion_matrix_complete', completed_heads=40,
                          independent_adaptation_repeats=3, downstream_seeds_per_cell=5)
        except BaseException:
            status.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            write_json(REPORT/'summary.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['bind','run'], required=True)
    args = parser.parse_args()
    print(json.dumps(bind() if args.mode=='bind' else run(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
