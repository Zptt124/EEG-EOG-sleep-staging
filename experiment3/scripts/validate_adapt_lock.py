"""Read-only validation of the W2 C configuration lock; never trains or reads NPZ."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/adapt_W2_C_S1_pilot_v01.yaml'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def resolve_path(value):
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT.resolve()), f'Path escapes experiment workspace: {value}')
    return path


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def validate_contract(config):
    c = config
    require(c['status'] == 'locked_training_not_started', 'Unexpected lifecycle state')
    data = c['data']
    require((data['n_subjects'], data['n_records'], data['allowed_split']) == (54, 106, 'train'), 'Training pool changed')
    require(set(data['allowed_npz_members']) == {'signal.npy', 'epoch_index.npy'}, 'Forbidden signal member policy')
    require(not any(data[k] for k in ['sleep_stage_labels_visible', 'read_validation_signals', 'read_test_signals', 'class_balanced_sampling']), 'Information isolation violated')
    require(data['y_placeholder'] == -1, 'Unlabeled y must remain hidden')
    init = c['initialization']
    require(init['reload_original_before_update_1'] and not init['inherit_diagnostic_weights'] and not init['inherit_optimizer_or_ema_counters'], 'Original reset policy violated')
    train = c['training']
    require((train['batch_size'], train['max_successful_optimizer_updates'], train['accumulation_steps']) == (2, 10000, 1), 'Training budget changed')
    require(train['total_crop_occurrences'] == train['batch_size'] * train['max_successful_optimizer_updates'], 'Crop budget inconsistent')
    require(train['learning_rate'] == 1e-5 and train['lr_schedule'] == 'constant', 'Optimizer schedule changed')
    require(train['precision'] == 'float32' and not train['autocast'] and not train['tf32'], 'Precision changed')
    require(train['extra_warmup_updates'] == train['diagnostic_updates_inherited'] == 0, 'Hidden updates introduced')
    model = c['model']
    require(model['trainable_prefixes'] == ['layers.18.', 'layers.19.', 'out_layer.'], 'Trainable blocks changed')
    require((model['trainable_parameter_count'], model['trainable_tensor_count']) == (527233, 34), 'Trainable count changed')
    require(model['freeze_other_parameters'] and not model['frozen_inference_kernel_memoization'], 'Frozen/cache policy violated')
    ema = c['ema']
    require(ema['enabled'] and ema['beta'] == 0.999 and ema['update_every'] == ema['first_update_step'] == 1, 'EMA schedule changed')
    require(not ema['adaptive_decay_or_warmup'] and ema['averaged_parameters'] == 'exact_trainable_parameter_names_only', 'EMA averaging policy changed')
    selection = c['stopping_and_selection']
    require(selection['selected_step'] == train['max_successful_optimizer_updates'] and selection['selected_state'] == 'new_EMA_after_step_10000_update', 'Final weight selection changed')
    require(not any(selection[k] for k in ['early_stopping', 'external_validation_used_for_backbone_selection', 'choose_by_adaptation_loss', 'choose_by_N1_or_macro_f1']), 'Data-dependent backbone selection introduced')
    require(c['diffusion']['prediction_type'] == 'v_prediction' and c['diffusion']['num_train_timesteps'] == 50, 'Diffusion target or noise schedule changed')
    heads = c['downstream_comparison']
    require(heads['budgets'] == ['label10', 'label25'] and heads['downstream_seeds'] == [0, 1, 2, 3, 4], 'Downstream matrix changed')
    require(heads['labeled_subject_counts'] == [5, 14] and heads['formal_heads_planned_before_reuse_audit'] == 40, 'Downstream budget count mismatch')
    require(not c['feature_extraction_after_adaptation']['overwrite_S0'], 'S0 overwrite forbidden')
    require(c['runtime_binding']['this_lock_operation_starts_training'] is False, 'Config check must not train')
    return True


def validate_locked_files(config_path=DEFAULT_CONFIG, lock_path=None):
    config_path = Path(config_path).resolve()
    lock_path = Path(lock_path) if lock_path else config_path.with_suffix('.lock.json')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    validate_contract(config)
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    require(sha(config_path) == lock['config_sha256'], 'Configuration changed after lock; create a new version')
    for relative, expected in lock['input_sha256'].items():
        require(sha(resolve_path(relative)) == expected, f'Locked input changed: {relative}')
    require(sha(resolve_path(config['initialization']['checkpoint'])) == config['initialization']['checkpoint_sha256'], 'Original checkpoint changed')
    split = csv_rows(resolve_path(config['data']['split']))
    expected = {r['subject_id']: r['record_ids'] for r in split if r['split'] == 'train'}
    subjects = csv_rows(resolve_path(config['data']['adaptation_subjects']))
    require(len(subjects) == 54 and len({r['subject_id'] for r in subjects}) == 54, 'Missing or repeated adaptation subjects')
    require({r['subject_id']: r['record_ids'] for r in subjects} == expected, 'Adaptation subjects do not exactly match train split')
    require(all(r['split'] == 'train' and r['labels_available'] == 'False' for r in subjects), 'Adaptation subject label visibility changed')
    params = csv_rows(resolve_path(config['model']['exact_parameter_table']))
    names = resolve_path(config['model']['exact_parameter_names']).read_text(encoding='utf-8').splitlines()
    require([r['name'] for r in params] == names and len(names) == len(set(names)) == 34, 'Exact trainable names differ')
    require(sum(int(r['numel']) for r in params) == 527233, 'Trainable numel differs')
    require(all(name.startswith(tuple(config['model']['trainable_prefixes'])) for name in names), 'Unexpected trainable parameter')
    require(all(float(r['lr']) == 1e-5 and float(r['weight_decay']) == 0 for r in params), 'Parameter optimizer policy differs')
    budgets = csv_rows(resolve_path(config['downstream_comparison']['manifest']))
    require(len(budgets) == 54 and {r['subject_id'] for r in budgets} == set(expected), 'Downstream subset pool differs')
    for key, size in [('label10', 5), ('label25', 14), ('label50', 27), ('label100', 54)]:
        require(sum(int(r[key]) for r in budgets) == size, 'Downstream subset size differs')
        require(all(int(r[key]) == int(int(r['sampling_rank']) <= size) for r in budgets), 'Downstream rank nesting differs')
    benchmark = json.loads(resolve_path(lock['benchmark_summary']).read_text(encoding='utf-8'))
    require(benchmark['status'] == 'passed' and benchmark['completed_updates'] == 100, 'Required benchmark evidence missing')
    require(benchmark['initial_parameter_sha256'] == config['initialization']['expected_initial_parameter_sha256'], 'Expected initialization differs from verified benchmark')
    return dict(status='passed', config_id=config['config_id'], config_sha256=lock['config_sha256'],
        locked_input_count=len(lock['input_sha256']), adaptation_subjects=54, trainable_parameters=527233,
        planned_updates=10000, formal_training_started=False,
        runtime_binding_required=True, stage_labels_or_npz_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate_locked_files(args.config, args.lock)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
