"""One detached job: selected repeat EMA -> full EEG cache -> ten heads.

S0 is reused only after full configuration/artifact/prediction identity checks.
The immutable pilot numerical routines are versioned with seed metadata changes.
"""
import argparse
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch
import yaml

import extract_adapted_features as extraction
import run_s0_s1_eeg_comparison as heads
import replication_feature_head_runtime as runtime

ROOT = heads.ROOT
require = heads.require
sha = heads.sha
read_csv = heads.read_csv
write_json = heads.write_json
PILOT_FEATURE_CONFIG = extraction.DEFAULT_CONFIG
PILOT_REPORT = heads.REPORT/'execution'


def validate_feature_derivation(config):
    seed = config['adaptation_seed']
    require(type(seed) is int and seed in (1, 2), 'Unregistered adaptation seed')
    revision = config.get('feature_revision', 1)
    require(type(revision) is int and 1 <= revision <= 99, 'Invalid feature artifact revision')
    expected = yaml.safe_load(PILOT_FEATURE_CONFIG.read_text(encoding='utf-8'))
    adaptation = ROOT/f'configs/adapt_W2_C_S1_aseed{seed}_v01.yaml'
    changes = dict(config_id=f'W2_C_S1_aseed{seed}_features_v{revision:02d}', adaptation_seed=seed,
                   adaptation_config=adaptation.relative_to(ROOT).as_posix(), adaptation_config_sha256=sha(adaptation),
                   adaptation_run=f'runs/W2_C_EEG_S1_aseed{seed}_steps10000_v01',
                   S1_feature_root=f'features/eeg_cond14_S1_aseed{seed}_steps10000_v{revision:02d}')
    if 'feature_revision' in config:
        changes['feature_revision'] = revision
    expected.update(changes)
    require(config == expected, 'Repeat feature settings differ beyond seed/paths and exact parent SHA')
    extraction.validate_feature_contract(config)
    return seed


def check_checkpoint_seed(checkpoint, seed):
    require(checkpoint.get('adaptation_seed') == seed, 'Selected checkpoint adaptation seed mismatch')


def load_selected(config_path):
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    seed = validate_feature_derivation(cfg)
    cfg, preparation, model, rows, grouped, partitions = extraction.load_selected(config_path)
    selected = extraction.contract.resolve_path(preparation['selected_weight_path'])
    checkpoint = torch.load(selected, map_location='cpu', weights_only=True)
    check_checkpoint_seed(checkpoint, seed)
    preparation.update(adaptation_seed=seed, extractor_script_sha256=sha(runtime.__file__),
                       pilot_extractor_script_sha256=sha(extraction.__file__),
                       pipeline_script_sha256=sha(__file__))
    return cfg, preparation, model, rows, grouped, partitions


def check_baseline_config(cfg, budget, seed, selected, ntrain, val_subjects):
    expected = dict(representation='S0', adaptation_seed=None, downstream_seed=seed, sampling_seed=2024,
                    budget=budget, modality='EEG', role='diagnostic', input_dim=2560,
                    classifier='Linear(2560,5)', optimizer='AdamW', learning_rate=0.001, weight_decay=0.0001,
                    batch_size=128, max_epochs=50, patience=10, loss='unweighted_cross_entropy',
                    selection='strictly highest development validation macro-F1',
                    normalization='fit only current budget labeled train subjects, independently for S0/S1',
                    std_floor=1e-6, backbone_frozen=True, test_used=False, device='cpu', cpu_threads=4,
                    deterministic=True, N1_REM_from_same_selected_head=True,
                    training_subjects=sorted(selected), training_samples=ntrain,
                    validation_subjects=sorted(val_subjects), validation_samples=21186,
                    local_unlabeled_adaptation_subjects=0,
                    evaluator_sha256=sha(heads.EVALUATOR), label_manifest_sha256=sha(heads.SUBSETS),
                    trainer_sha256=sha(heads.__file__),
                    feature_audit_sha256=sha(PILOT_REPORT/'S0_feature_audit.json'))
    require(all(cfg.get(k) == v for k, v in expected.items()), 'S0 full configuration match failed')


def baseline_reuse(rows, output):
    official = heads.official_module()
    old_audit = json.loads((PILOT_REPORT/'completion_audit.json').read_text())
    require(old_audit['status'] == 'passed' and old_audit['completed_heads'] == 20, 'Pilot baseline audit missing')
    source_audit = json.loads((PILOT_REPORT/'S0_feature_audit.json').read_text())
    require(source_audit['status'] == 'passed' and source_audit['samples'] == 154124
            and source_audit['source_index_sha256'] == sha(heads.SOURCES['S0']/'feature_index.csv'), 'S0 source changed')
    subsets = read_csv(heads.SUBSETS)
    registry = read_csv(PILOT_REPORT/'run_registry.csv')
    results, audits = [], []
    for budget in ('label10', 'label25'):
        selected, train, val = heads.selected_positions(rows, subsets, budget)
        val_rows = [rows[i] for i in val]
        expected_metadata = [(r['sample_id'], r['subject_id'], r['record_id'], int(r['epoch_index']), int(r['y_true']))
                             for r in val_rows]
        for seed in range(5):
            run_id = f'W2_C_S0_EEG_{budget}_sseed2024_seed{seed}_v01'
            folder = ROOT/'runs'/run_id
            artifacts = json.loads((folder/'artifact_sha256.json').read_text())
            require(all(sha(folder/name) == digest for name, digest in artifacts.items()), 'S0 archived artifact checksum failed')
            cfg = yaml.safe_load((folder/'config.yaml').read_text())
            check_baseline_config(cfg, budget, seed, selected, len(train), {r['subject_id'] for r in val_rows})
            frame = official.load_predictions(folder/'predictions_val.csv')
            actual_metadata = list(frame[['sample_id','subject_id','record_id','epoch_index','y_true']].itertuples(index=False, name=None))
            require(actual_metadata == expected_metadata, 'S0 validation sample identity mismatch')
            metrics = official.compute_metrics(frame['y_true'], frame['y_pred'])
            stored = json.loads((folder/'metrics.json').read_text())
            require(metrics['confusion_matrix'] == stored['confusion_matrix'], 'S0 metric replay mismatch')
            result = json.loads((folder/'summary.json').read_text())
            require(result['status'] == 'complete' and result['val_macro_f1'] == metrics['macro_f1'], 'S0 incomplete result')
            require(any(r['run_id'] == run_id and float(r['val_macro_f1']) == metrics['macro_f1'] for r in registry),
                    'S0 run registry mismatch')
            results.append(result)
            audits.append(dict(run_id=run_id, configuration_match=True, archived_artifacts_verified=len(artifacts),
                               validation_sample_metadata_match=True, metrics_recomputed=True,
                               predictions_sha256=sha(folder/'predictions_val.csv')))
    write_json(output, dict(status='passed', reused_heads=10, test_used=False, checks=audits))
    return results


def bound_files(config_path, report):
    cfg = yaml.safe_load(config_path.read_text())
    adaptation = extraction.contract.resolve_path(cfg['adaptation_config'])
    files = [Path(__file__), Path(runtime.__file__), Path(extraction.__file__), Path(heads.__file__),
             ROOT/'scripts/extract_features.py', ROOT/'scripts/validate_adapt_lock.py',
             ROOT/'tests/test_replication_features_heads.py', config_path, PILOT_FEATURE_CONFIG,
             adaptation, adaptation.with_suffix('.lock.json'), heads.SUBSETS, heads.SPLIT, heads.EVALUATOR,
             heads.SOURCES['S0']/'feature_index.csv', heads.SOURCES['S0']/'feature_config.json',
             report/'extraction_binding.json', report/'probe/probe_audit.json', report/'runtime_tests.log',
             PILOT_REPORT/'S0_feature_audit.json', PILOT_REPORT/'completion_audit.json']
    for budget in ('label10', 'label25'):
        for seed in range(5):
            folder = ROOT/'runs'/f'W2_C_S0_EEG_{budget}_sseed2024_seed{seed}_v01'
            files += [folder/name for name in ['artifact_sha256.json', 'config.yaml', 'summary.json',
                                               'metrics.json', 'predictions_val.csv']]
    return files


def bind(config_path, report):
    tests = (report/'runtime_tests.log').read_text()
    require('Ran 10 tests' in tests and '\nOK\n' in tests, 'Repeat pipeline tests have not passed')
    cfg, preparation, model, rows, grouped, partitions = load_selected(config_path)
    require(preparation == json.loads((report/'extraction_binding.json').read_text()), 'Prepared extractor provenance differs')
    probe = json.loads((report/'probe/probe_audit.json').read_text())
    require(probe['status'] == 'passed' and probe['selected_weight_sha256'] == preparation['selected_weight_sha256'],
            'New selected-model numerical probe missing')
    results = baseline_reuse(rows, report/'S0_reuse_audit.json')
    path = report/'pipeline_binding.json'
    require(not path.exists(), 'Do not overwrite pipeline binding')
    files = bound_files(config_path, report) + [report/'S0_reuse_audit.json']
    value = dict(status='bound_before_full_extraction', adaptation_seed=cfg['adaptation_seed'], new_heads=10,
                 reused_S0_heads=10, input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
                 feature_config=config_path.relative_to(ROOT).as_posix(),
                 selected_weight_sha256=preparation['selected_weight_sha256'], test_used=False)
    write_json(path, value)
    return value


def validate_binding(path):
    value = json.loads(path.read_text())
    require(value['status'] == 'bound_before_full_extraction', 'Invalid pipeline binding')
    for name, digest in value['input_sha256'].items():
        require(sha(extraction.contract.resolve_path(name)) == digest, f'Pipeline input changed: {name}')
    return value


def run(config_path, report, resume=False):
    binding_path = report/'pipeline_binding.json'
    binding = validate_binding(binding_path)
    cfg, preparation, model, rows, grouped, partitions = load_selected(config_path)
    require(preparation == json.loads((report/'extraction_binding.json').read_text()), 'Extraction preparation changed')
    seed = cfg['adaptation_seed']
    execution = report/'execution'
    execution.mkdir(parents=True, exist_ok=resume)
    feature_root = extraction.safe_output_root(cfg)
    with (execution/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(status='running', stage='full_feature_extraction', adaptation_seed=seed, pid=os.getpid(),
                      completed_new_heads=0, expected_new_heads=10, reused_S0_heads=10,
                      feature_summary=(feature_root/'summary.json').relative_to(ROOT).as_posix(), test_used=False)
        write_json(execution/'summary.json', status)
        try:
            if not resume:
                feature_root.mkdir(parents=True, exist_ok=False)
            with (feature_root/'.extract.lock').open('a') as file_lock:
                fcntl.flock(file_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                completed = (feature_root/'summary.json').exists() and json.loads((feature_root/'summary.json').read_text())['status'] == 'complete'
                if not completed:
                    runtime.extract_locked(cfg, preparation, model, rows, grouped, partitions, feature_root, resume)
                else:
                    require(resume and json.loads((feature_root/'extraction_binding.json').read_text()) == preparation,
                            'Existing complete extraction has different provenance')
            del model
            torch.cuda.empty_cache()
            validate_binding(binding_path)
            status.update(stage='full_cache_audit_and_pool')
            write_json(execution/'summary.json', status)
            s1_rows = read_csv(feature_root/'feature_index.csv')
            grouped_index = heads.audit_index(s1_rows, read_csv(heads.SPLIT))
            heads.align_indices(rows, s1_rows)
            require(json.loads((feature_root/'feature_config.json').read_text())['adaptation_seed'] == seed,
                    'Full feature cache is labeled with the wrong adaptation seed')
            pooled = execution/'S1_pooled.npy'
            if not pooled.exists():
                runtime.audit_and_pool('S1', s1_rows, grouped_index, execution, feature_root, report/'extraction_binding.json')
            else:
                audit = json.loads((execution/'S1_feature_audit.json').read_text())
                require(resume and audit['status'] == 'passed' and sha(pooled) == audit['pooled_sha256'], 'Invalid existing pooled cache')
            results = baseline_reuse(rows, execution/'S0_reuse_audit.json')
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            official = heads.official_module()
            subsets = read_csv(heads.SUBSETS)
            for budget in ('label10', 'label25'):
                selected, train_ids, val_ids = heads.selected_positions(rows, subsets, budget)
                val_rows = [rows[i] for i in val_ids]
                for downstream_seed in range(5):
                    run_id = f'W2_C_S1_EEG_aseed{seed}_{budget}_sseed2024_seed{downstream_seed}_v01'
                    status.update(stage='head_training', current_run=run_id)
                    write_json(execution/'summary.json', status)
                    require(not (ROOT/'runs'/run_id).exists(), 'Head output exists; classifier state is never silently restarted')
                    raw = np.load(pooled, mmap_mode='r', allow_pickle=False)
                    train = np.array(raw[train_ids], copy=True)
                    val = np.array(raw[val_ids], copy=True)
                    mean, std = heads.fit_normalization(train)
                    train -= mean; train /= std
                    val -= mean; val /= std
                    require(np.isfinite(train).all() and np.isfinite(val).all(), 'Nonfinite normalized features')
                    train_x, val_x = torch.from_numpy(train), torch.from_numpy(val)
                    train_y = torch.tensor([int(rows[i]['y_true']) for i in train_ids], dtype=torch.int64)
                    val_y = np.array([int(r['y_true']) for r in val_rows])
                    plan = dict(run_id=run_id, representation='S1', budget=budget, downstream_seed=downstream_seed)
                    context = dict(training_subjects=sorted(selected), training_samples=len(train_ids),
                                   validation_subjects=sorted({r['subject_id'] for r in val_rows}), validation_samples=len(val_ids),
                                   train_class_counts=dict(Counter(map(int, train_y.tolist()))),
                                   feature_audit_sha256=sha(execution/'S1_feature_audit.json'),
                                   runtime_binding_sha256=sha(binding_path), trainer_sha256=sha(runtime.__file__),
                                   evaluator_sha256=sha(heads.EVALUATOR), label_manifest_sha256=sha(heads.SUBSETS),
                                   local_unlabeled_adaptation_subjects=54)
                    result = runtime.train_head(plan, val_rows, train_x, train_y, val_x, val_y, mean, std, context,
                                                official, adaptation_seed=seed)
                    results.append(result)
                    heads.write_csv(execution/'run_registry.csv', results)
                    status.update(completed_new_heads=len(results)-10)
                    write_json(execution/'summary.json', status)
                    heads.paired_report(results, execution, official)
                    del raw, train, val, train_x, val_x, train_y
            validate_binding(binding_path)
            status.update(status='complete', stage='seed_repeat_EEG_comparison_complete', current_run=None,
                          independently_adapted_backbones_in_this_job=1, adaptation_seeds_available=[0, seed])
        except BaseException:
            status.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            write_json(execution/'summary.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'probe', 'bind', 'run'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        args.report.mkdir(parents=True, exist_ok=False)
        cfg, preparation, model, rows, grouped, partitions = load_selected(args.config)
        write_json(args.report/'extraction_binding.json', preparation)
        result = preparation
    elif args.mode == 'probe':
        cfg, preparation, model, rows, grouped, partitions = load_selected(args.config)
        result = extraction.probe(cfg, preparation, model, rows, grouped, partitions, args.report/'probe')
    elif args.mode == 'bind':
        result = bind(args.config, args.report)
    else:
        result = run(args.config, args.report, args.resume)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
