"""Derive an already preregistered adaptation repeat without changing method.

The original pilot config, lock, trainer, caches and results are immutable inputs.
This prepares files only: zero signal reads and zero optimizer updates.
"""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import yaml
import validate_adapt_lock as contract

ROOT = Path(__file__).resolve().parents[1]
PARENT = contract.DEFAULT_CONFIG
PILOT_EEG = ROOT/'reports/week2_eeg_comparison_20261028/execution'


def changes_for_seed(seed):
    contract.require(seed in (1, 2), 'Only preregistered extension seeds 1/2 are allowed')
    return {
        ('config_id',): f'W2_C_S1_aseed{seed}_v01',
        ('run_id',): f'W2_C_EEG_S1_aseed{seed}_steps10000_v01',
        ('sampling', 'adaptation_seed'): seed,
        ('sampling', 'data_seed'): seed,
        ('sampling', 'noise_seed'): seed,
        ('checkpoints', 'run_root'): f'runs/W2_C_EEG_S1_aseed{seed}_steps10000_v01',
        ('feature_extraction_after_adaptation', 'feature_root_S1'): f'features/eeg_cond14_S1_aseed{seed}_steps10000_v01',
        ('replication', 'current_adaptation_seed'): seed,
    }


def set_nested(config, path, value):
    current = config
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def validate_derivation(parent, repeat, seed):
    contract.require(seed in parent['replication']['extension_seeds_preregistered'], 'Seed was not preregistered')
    expected = copy.deepcopy(parent)
    for path, value in changes_for_seed(seed).items():
        set_nested(expected, path, value)
    contract.require(repeat == expected, 'Repeat differs beyond seed and unique artifact identities/paths')
    contract.validate_contract(repeat)
    return True


def derive(parent, seed):
    repeat = copy.deepcopy(parent)
    for path, value in changes_for_seed(seed).items():
        set_nested(repeat, path, value)
    validate_derivation(parent, repeat, seed)
    return repeat


def prepare(seed):
    contract.validate_locked_files(PARENT)
    parent = yaml.safe_load(PARENT.read_text(encoding='utf-8'))
    original_lock_path = PARENT.with_suffix('.lock.json')
    original_lock = json.loads(original_lock_path.read_text(encoding='utf-8'))
    pilot_run = contract.resolve_path(parent['checkpoints']['run_root'])/'summary.json'
    pilot = json.loads(pilot_run.read_text())
    contract.require(pilot['status'] == 'completed' and pilot['completed_updates'] == pilot['ema_updates'] == 10000,
                     'Original adaptation pilot has not completed')
    eeg_summary = json.loads((PILOT_EEG/'summary.json').read_text())
    eeg_audit = json.loads((PILOT_EEG/'completion_audit.json').read_text())
    contract.require(eeg_summary['status'] == 'complete' and eeg_summary['completed_heads'] == 20
                     and eeg_audit['status'] == 'passed' and not eeg_audit['test_used'],
                     'Pilot EEG artifacts have not completed verification')
    repeat = derive(parent, seed)
    config_path = ROOT/f'configs/adapt_W2_C_S1_aseed{seed}_v01.yaml'
    lock_path = config_path.with_suffix('.lock.json')
    report = ROOT/f'reports/week2_adaptation_repeat_aseed{seed}_20261028'
    contract.require(not config_path.exists() and not lock_path.exists(), 'Repeat config already exists')
    contract.require(not contract.resolve_path(repeat['checkpoints']['run_root']).exists(), 'Repeat run already exists')
    contract.require(not contract.resolve_path(repeat['feature_extraction_after_adaptation']['feature_root_S1']).exists(),
                     'Repeat feature destination already exists')
    report.mkdir(parents=True, exist_ok=False)
    config_path.write_text(yaml.safe_dump(repeat, sort_keys=False, allow_unicode=True), encoding='utf-8')
    inputs = dict(original_lock['input_sha256'])
    provenance = [PARENT, original_lock_path, Path(__file__), ROOT/'tests/test_adaptation_replication.py',
                  ROOT/'scripts/train_sleep_adaptation.py', ROOT/'tests/test_train_sleep_adaptation.py',
                  pilot_run, PILOT_EEG/'summary.json', PILOT_EEG/'completion_audit.json']
    inputs.update({path.relative_to(ROOT).as_posix(): contract.sha(path) for path in provenance})
    lock = dict(original_lock)
    lock.update(config_id=repeat['config_id'], config_path=config_path.relative_to(ROOT).as_posix(),
                config_sha256=contract.sha(config_path), input_sha256=inputs,
                derived_from_config=PARENT.relative_to(ROOT).as_posix(),
                derived_from_config_sha256=contract.sha(PARENT), adaptation_seed=seed,
                created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                derivation_policy='Only three RNG seeds, current seed and unique config/run/feature identities differ',
                pilot_EEG_complete=True, EOG_primary_comparison_complete=False,
                continuation_reason='Preregistered independent repeat; GPU/storage available; not selected by positive outcome')
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    validation = contract.validate_locked_files(config_path)
    result = dict(status='prepared_not_started', adaptation_seed=seed, config=config_path.relative_to(ROOT).as_posix(),
                  config_sha256=contract.sha(config_path), lock_sha256=contract.sha(lock_path),
                  changes={'.'.join(k): v for k, v in changes_for_seed(seed).items()},
                  parent_config_preserved=True, inherited_seed0_weights=False, optimizer_updates=0,
                  formal_trainer_unchanged=True, source_data_read=False, validation=validation)
    (report/'derivation_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[1, 2], required=True)
    print(json.dumps(prepare(parser.parse_args().seed), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
