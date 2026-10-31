"""One-shot continuation of the authorized seed2 repeat, not a scheduled job.

Wait for the existing fixed-step trainer, independently verify its trajectory
and weights, then invoke the already tested prepare/probe/bind/run pipeline.
"""
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/week2_seed2_sequence_20261029'
TRAIN = ROOT/'runs/W2_C_EEG_S1_aseed2_steps10000_v01'
TRAIN_REPORT = ROOT/'reports/week2_adaptation_repeat_aseed2_20261028'
FEATURE_REPORT = ROOT/'reports/week2_seed2_features_heads_20261029'
CONFIG = ROOT/'configs/features_W2_C_S1_aseed2_v01.yaml'
PYTHON = '/opt/eegdm/venv/bin/python'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def check_binding():
    binding = json.loads((REPORT/'continuation_binding.json').read_text())
    for relative, expected in binding['input_sha256'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'Seed2 continuation input changed: {relative}')
    return binding


def verify_training(summary):
    if not (summary['status'] == 'completed' and summary['completed_updates'] == summary['ema_updates'] == 10000):
        raise ValueError('Seed2 trajectory has not completed')
    if summary['config_sha256'] != sha(ROOT/'configs/adapt_W2_C_S1_aseed2_v01.yaml'):
        raise ValueError('Seed2 training configuration mismatch')
    if not summary['frozen_parameters_unchanged'] or summary['frozen_online_sha256'] != '5639fe62da24126cd5e199cf86de81e42bb35fd9a5e00d91257a419d1cb1d63c':
        raise ValueError('Seed2 frozen parameter check failed')
    if any(summary[k] for k in ['stage_labels_read', 'validation_signals_read', 'test_signals_read']):
        raise ValueError('Forbidden adaptation data access')
    with (TRAIN/'adapt_history.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    if [int(r['step']) for r in history] != list(range(1, 10001)):
        raise ValueError('Seed2 training history is incomplete')
    if not all(int(r['ema_updates']) == int(r['step']) and math.isfinite(float(r['loss']))
               and math.isfinite(float(r['gradient_norm_before_clip'])) for r in history):
        raise ValueError('Seed2 loss/gradient/EMA evidence failed')
    for name in ['selected_weight', 'online_last_weight']:
        metadata = summary[name]
        path = TRAIN/metadata['path']
        if path.parent != TRAIN or sha(path) != metadata['sha256'] or path.stat().st_size != metadata['bytes']:
            raise ValueError('Seed2 final weight integrity failed')
    with (TRAIN/'signal_access_audit.csv').open(newline='') as stream:
        if not all(r['split'] == 'train' and r['member'] in ('signal.npy', 'epoch_index.npy') for r in csv.DictReader(stream)):
            raise ValueError('Seed2 signal-access evidence failed')
    save(TRAIN_REPORT/'training_completion_audit.json', dict(status='passed', verified_at=now(),
         completed_at=summary['completed_at_Asia_Shanghai'], completed_updates=10000, EMA_updates=10000,
         history_contiguous=True, losses_and_gradient_norms_finite=True, final_weight_sha256_verified=True,
         frozen_online_digest_matches_original=True, stage_labels_read=False, validation_signals_read=False,
         test_signals_read=False))


def main():
    with (REPORT/'.continuation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_binding()
        launch = json.loads((TRAIN_REPORT/'launch.json').read_text())
        status = dict(status='running', stage='seed2_adaptation_training', pid=os.getpid(),
                      started_at=now(), adaptation_seed=2, completed_updates=0, target_updates=10000,
                      subsequent_new_heads=10, reused_S0_heads=10, test_used=False,
                      feature_report=FEATURE_REPORT.relative_to(ROOT).as_posix())
        save(REPORT/'summary.json', status)
        try:
            previous_step = -1
            while True:
                path = TRAIN/'summary.json'
                summary = json.loads(path.read_text()) if path.exists() else {}
                if summary.get('status') == 'completed':
                    break
                if summary.get('status') == 'failed':
                    raise RuntimeError('Seed2 trainer failed; continuation stopped without changing settings')
                os.kill(launch['pid'], 0)
                step = summary.get('completed_updates', 0)
                if step != previous_step:
                    status.update(completed_updates=step, training_eta_seconds=summary.get('estimated_remaining_seconds'),
                                  updated_at=now())
                    save(REPORT/'summary.json', status)
                    previous_step = step
                time.sleep(20)
            check_binding()
            verify_training(summary)
            for mode in ['prepare', 'probe', 'bind', 'run']:
                check_binding()
                status.update(stage='seed2_features_and_heads' if mode == 'run' else f'seed2_{mode}',
                              completed_updates=10000, updated_at=now())
                save(REPORT/'summary.json', status)
                command = [PYTHON, '-u', str(ROOT/'scripts/run_replication_features_heads.py'), '--mode', mode,
                           '--config', str(CONFIG), '--report', str(FEATURE_REPORT)]
                with (REPORT/f'{mode}_console.log').open('xb', buffering=0) as log:
                    result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'Seed2 {mode} exited {result.returncode}; see {mode}_console.log')
                if mode == 'prepare':
                    shutil.copyfile(ROOT/'reports/week2_seed1_features_heads_20261028_v04/runtime_tests.log',
                                    FEATURE_REPORT/'runtime_tests.log')
            final = json.loads((FEATURE_REPORT/'execution/summary.json').read_text())
            if final['status'] != 'complete' or final['completed_new_heads'] != 10:
                raise ValueError('Seed2 downstream job did not complete all positions')
            status.update(status='complete', stage='seed2_full_repeat_complete', completed_at=now(),
                          completed_new_heads=10, reused_S0_heads=10)
        except BaseException:
            import traceback
            status.update(status='failed', failed_at=now(), error=traceback.format_exc())
            raise
        finally:
            save(REPORT/'summary.json', status)


if __name__ == '__main__':
    main()
