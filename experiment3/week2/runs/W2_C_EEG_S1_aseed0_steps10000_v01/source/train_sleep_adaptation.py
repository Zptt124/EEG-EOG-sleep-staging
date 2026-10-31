"""Formal W2 C adaptation of the locked pilot; no label/validation signal access.

--validate-only loads the exact original EMA, checks runtime/optimizer/new EMA,
and performs ZERO forward/backward/optimizer updates. The training path requires
a separately recorded source/test/initialization binding before update one.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import adapt_preflight as reference
import validate_adapt_lock as contract

ROOT = Path(__file__).resolve().parents[1]
HISTORY_FIELDS = ['step', 'loss', 'gradient_norm_before_clip', 'data_seconds', 'compute_seconds',
    'step_seconds', 'peak_allocated_mib', 'peak_reserved_mib', 'timesteps', 'ema_updates']


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def sha(path):
    return contract.sha(path)


def require(value, message):
    reference.require(value, message)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size)


def parameter_digest(state, parameter_names):
    digest = hashlib.sha256()
    for name in parameter_names:
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class FixedEMA:
    """Exact full state; average only the locked trainable FP32 tensors."""
    def __init__(self, model, names, beta=0.999):
        self.names = tuple(names)
        self.beta = beta
        self.updates = 0
        self.state = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
        require(all(self.state[n].dtype == torch.float32 for n in names), 'EMA needs FP32 trainable weights')

    @torch.no_grad()
    def update(self, model):
        online = dict(model.named_parameters())
        for name in self.names:
            self.state[name].mul_(self.beta).add_(online[name].detach(), alpha=1-self.beta)
        self.updates += 1

    def restore(self, state, updates):
        require(set(state) == set(self.state), 'EMA recovery state keys differ')
        for name, value in state.items():
            self.state[name].copy_(value.to(self.state[name].device))
        self.updates = int(updates)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all([r.cpu() for r in state['torch_cuda']])


def capture_sampler(provider):
    return dict(order=[int(i) for i in provider.order], rng=copy.deepcopy(provider.rng.bit_generator.state),
        examples=copy.deepcopy(provider.examples), accesses=copy.deepcopy(provider.reader.accesses))


def restore_sampler(provider, state):
    provider.order = list(state['order'])
    provider.rng.bit_generator.state = copy.deepcopy(state['rng'])
    provider.examples = copy.deepcopy(state['examples'])
    provider.reader.accesses = copy.deepcopy(state['accesses'])


def environment_check(config):
    required = json.loads(contract.resolve_path(config['environment']['constraints']).read_text(encoding='utf-8'))
    require(sys.version.split()[0] == required['python'], 'Python version differs from lock')
    versions = {name: importlib.metadata.version(name) for name in required['packages']}
    require(versions == required['packages'], 'Package versions differ from lock')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    return dict(python=sys.version.split()[0], packages=versions, gpu=torch.cuda.get_device_name(0),
        total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu',
            '--format=csv,noheader'], text=True).strip())


def validate_binding(path, config_sha):
    binding = json.loads(Path(path).read_text(encoding='utf-8'))
    require(binding['config_sha256'] == config_sha and binding['unit_tests_passed'] >= 6,
            'Missing configuration/test binding')
    require(binding['trainer_sha256'] == sha(__file__), 'Formal trainer changed after binding')
    for relative, expected in binding['file_sha256'].items():
        require(sha(contract.resolve_path(relative)) == expected, f'Runtime binding input changed: {relative}')
    initialization = json.loads(contract.resolve_path(binding['initialization_report']).read_text(encoding='utf-8'))
    require(initialization['status'] == 'passed' and initialization['optimizer_updates'] == initialization['ema_updates'] == 0,
            'Zero-update runtime initialization was not verified')
    return binding


def initialize(config, validation):
    env = environment_check(config)
    train = config['training']
    torch.set_num_threads(train['cpu_threads'])
    torch.manual_seed(config['sampling']['noise_seed'])
    torch.cuda.manual_seed_all(config['sampling']['noise_seed'])
    random.seed(config['sampling']['data_seed'])
    np.random.seed(config['sampling']['data_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(reference.UPSTREAM))
    from model.diffusion_model import Wavenet
    from model import s4standalone
    from diffusers import DDPMScheduler
    commit = subprocess.check_output(['git', '-C', str(reference.UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == config['initialization']['upstream_commit'], 'Upstream revision changed')
    require(not subprocess.check_output(['git', '-c', 'core.autocrlf=true', '-C', str(reference.UPSTREAM),
        'status', '--porcelain'], text=True).strip(), 'Upstream working tree changed')
    ckpt = torch.load(contract.resolve_path(config['initialization']['checkpoint']), map_location='cpu', weights_only=False)
    model = Wavenet(**dict(ckpt['hyper_parameters']['model_kwargs']))
    prefix = config['initialization']['state_dict_prefix']
    model.load_state_dict({k[len(prefix):]: v for k, v in ckpt['state_dict'].items() if k.startswith(prefix)}, strict=True)
    model.requires_grad_(False)
    names = contract.resolve_path(config['model']['exact_parameter_names']).read_text(encoding='utf-8').splitlines()
    selected = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    model.train().cuda()
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    require([n for n, _ in parameters] == names, 'Runtime trainable names differ')
    require(sum(p.numel() for _, p in parameters) == config['model']['trainable_parameter_count'], 'Runtime parameter count differs')
    require(all(getattr(block.layer.kernel.forward, '__func__', None) is s4standalone.SSMKernelDiag.forward for block in model.layers),
            'S4 inference kernel cache detected')
    schedule = dict(ckpt['hyper_parameters']['noise_sch_kwargs'])
    expected = {k: config['diffusion'][k] for k in schedule}
    require(schedule == expected, 'Checkpoint diffusion noise schedule differs')
    scheduler = DDPMScheduler(**schedule)
    groups = []
    for _, p in parameters:
        special = getattr(p, '_optim', {})
        lr = min(train['learning_rate'], special.get('lr', train['learning_rate']))
        groups.append(dict(params=[p], lr=lr, weight_decay=train['weight_decay']))
    optimizer = torch.optim.AdamW(groups, lr=train['learning_rate'], betas=tuple(train['betas']), eps=train['eps'],
        weight_decay=train['weight_decay'], foreach=train['foreach'], amsgrad=train['amsgrad'])
    initial_sha = reference.state_digest(model)
    frozen_sha = reference.state_digest(model, frozen_only=True)
    require(initial_sha == config['initialization']['expected_initial_parameter_sha256'], 'Initial weights are not original EMA')
    require(frozen_sha == config['model']['frozen_parameter_sha256'], 'Frozen initial parameters differ')
    ema = FixedEMA(model, names, config['ema']['beta'])
    require(parameter_digest(ema.state, [n for n, _ in model.named_parameters()]) == initial_sha, 'New EMA did not initialize exactly')
    require(not optimizer.state and ema.updates == 0, 'Optimizer or EMA counters inherited')
    subjects, record_split = reference.validate_split(reference.read_csv(contract.resolve_path(config['data']['split'])))
    reader = reference.SignalOnlyReader(contract.resolve_path(config['data']['processed_root']), record_split)
    provider = reference.TrainingSignalBatches(reader, subjects, train['batch_size'], config['sampling']['data_seed'])
    report = dict(status='passed', optimizer_updates=0, ema_updates=0, stage_labels_or_npz_read=False,
        config_sha256=validation['config_sha256'], trainer_sha256=sha(__file__),
        initial_parameter_sha256=initial_sha, new_ema_initial_parameter_sha256=initial_sha,
        frozen_parameter_sha256=frozen_sha, trainable_tensors=len(names), trainable_parameters=527233,
        environment=env, time_Asia_Shanghai=now())
    return model, optimizer, ema, scheduler, provider, parameters, report


def assert_frozen(model, ema, expected_sha):
    frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
    require(reference.state_digest(model, frozen_only=True) == expected_sha, 'Frozen online parameters changed')
    require(parameter_digest(ema.state, frozen_names) == expected_sha, 'Frozen new EMA parameters changed')
    require(all(p.grad is None for p in model.parameters() if not p.requires_grad), 'Frozen parameter acquired gradient')
    for name, buffer in model.named_buffers():
        require(torch.equal(buffer, ema.state[name]), f'Frozen model buffer changed: {name}')


def assert_trainable_finite(parameters, ema):
    for name, parameter in parameters:
        require(bool(torch.isfinite(parameter).all()), f'Nonfinite online parameter: {name}')
        require(bool(torch.isfinite(ema.state[name]).all()), f'Nonfinite EMA parameter: {name}')


def make_recovery(model, optimizer, ema, provider, step, history, config_sha, binding_sha, extra_compute):
    return dict(schema_version=1, online=cpu_tree(model.state_dict()), new_ema=cpu_tree(ema.state),
        optimizer=cpu_tree(optimizer.state_dict()), completed_updates=step, ema_updates=ema.updates,
        rng=capture_rng(), sampler=capture_sampler(provider), history=copy.deepcopy(history),
        config_sha256=config_sha, runtime_source_sha256=sha(__file__), binding_sha256=binding_sha,
        reexecuted_lost_updates=extra_compute, checkpoint_role='hardware_recovery_only')


def restore_recovery(payload, model, optimizer, ema, provider, config_sha, binding_sha):
    require(payload['config_sha256'] == config_sha and payload['runtime_source_sha256'] == sha(__file__)
        and payload['binding_sha256'] == binding_sha, 'Recovery config or code differs')
    require(payload['completed_updates'] == payload['ema_updates'], 'Recovery online/EMA count mismatch')
    model.load_state_dict(payload['online'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    # CPU checkpoint loading must not leave AdamW moment tensors on the CPU.
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != 'step':
                state[key] = value.to(parameter.device)
    ema.restore(payload['new_ema'], payload['ema_updates'])
    restore_sampler(provider, payload['sampler'])
    restore_rng(payload['rng'])


def write_evidence(run, provider, history):
    if history:
        reference.write_csv(run / 'adapt_history.csv', history)
        reference.write_csv(run / 'history.csv', history)
    if provider.examples:
        reference.write_csv(run / 'signal_batches.csv', provider.examples)
    if provider.reader.accesses:
        reference.write_csv(run / 'signal_access_audit.csv', provider.reader.accesses)


def save_recovery(run, model, optimizer, ema, provider, step, history, config_sha, binding_sha, extra_compute):
    directory = run / 'checkpoints'
    directory.mkdir(exist_ok=True)
    path = directory / f'recovery_step{step:06d}.pt'
    info = atomic_save(path, make_recovery(model, optimizer, ema, provider, step, history, config_sha, binding_sha, extra_compute))
    info['step'] = step
    info['saved_at_Asia_Shanghai'] = now()
    atomic_json(run / 'latest_recovery.json', info)
    # Delete only known individual recovery files inside this run, never recurse.
    old = sorted(directory.glob('recovery_step*.pt'))[:-2]
    for previous in old:
        require(previous.resolve().parent == directory.resolve(), 'Unsafe recovery cleanup path')
        previous.unlink()
    return info


def run_training(args, config, validation, runtime):
    model, optimizer, ema, scheduler, provider, parameters, initial = runtime
    run = contract.resolve_path(config['checkpoints']['run_root'])
    binding = validate_binding(args.binding, validation['config_sha256'])
    binding_sha = sha(args.binding)
    config_sha = validation['config_sha256']
    source_sha = sha(__file__)
    if args.resume:
        require(run.is_dir(), 'No formal run to resume')
    else:
        run.mkdir(parents=True, exist_ok=False)
    with (run / '.process.lock').open('a') as process_lock:
        fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locked_training(args, config, validation, initial, run, binding, binding_sha,
            config_sha, source_sha, model, optimizer, ema, scheduler, provider, parameters)


def locked_training(args, config, validation, initial, run, binding, binding_sha, config_sha,
                    source_sha, model, optimizer, ema, scheduler, provider, parameters):
    summary = dict(status='initializing', run_id=config['run_id'], formal_adaptation=True,
        completed_updates=0, target_updates=10000, ema_updates=0, config_sha256=config_sha,
        trainer_sha256=source_sha, runtime_binding_sha256=binding_sha, pid=os.getpid(),
        started_at_Asia_Shanghai=now(), stage_labels_read=False, validation_signals_read=False,
        test_signals_read=False, downstream_evaluation=False, selected_state='new_EMA_after_step_10000_update')
    history = []
    step = 0
    extra_compute = 0
    attempt_streams = []
    try:
        if args.resume:
            previous = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
            require(previous['status'] not in ['failed', 'completed'], 'Nonfinite/failed/completed runs cannot be resumed')
            index = json.loads((run / 'latest_recovery.json').read_text(encoding='utf-8'))
            path = run / 'checkpoints' / index['path']
            require(path.resolve().parent == (run/'checkpoints').resolve() and sha(path) == index['sha256'], 'Recovery path/hash mismatch')
            payload = torch.load(path, map_location='cpu', weights_only=False)
            restore_recovery(payload, model, optimizer, ema, provider, config_sha, binding_sha)
            step = payload['completed_updates']
            history = payload['history']
            recorded_max = step
            for log in (run/'attempts').glob('attempt_*_history.csv'):
                with log.open(encoding='utf-8', newline='') as stream:
                    for row in csv.DictReader(stream):
                        recorded_max = max(recorded_max, int(row['step']))
            extra_compute = payload['reexecuted_lost_updates'] + max(recorded_max-step, 0)
            summary['started_at_Asia_Shanghai'] = previous['started_at_Asia_Shanghai']
            summary['resumed_from_step'] = step
        else:
            for source, destination in [(args.config, 'config.yaml'), (args.config, 'adapt_config.yaml'),
                    (args.config.with_suffix('.lock.json'), 'config.lock.json'), (args.binding, 'runtime_binding.json')]:
                shutil.copyfile(source, run/destination)
            source_dir = run/'source'
            source_dir.mkdir()
            for source in [Path(__file__), ROOT/'scripts/adapt_preflight.py', ROOT/'scripts/validate_adapt_lock.py']:
                shutil.copyfile(source, source_dir/source.name)
            shutil.copyfile(contract.resolve_path(config['data']['adaptation_subjects']), run/'adapt_subjects.csv')
            shutil.copyfile(contract.resolve_path(config['model']['exact_parameter_names']), run/'trainable_parameters.txt')
            shutil.copyfile(contract.resolve_path(config['model']['exact_parameter_table']), run/'trainable_parameters.csv')
            atomic_json(run/'initialization.json', initial)
            atomic_json(run/'environment.json', initial['environment'])
        assert_frozen(model, ema, config['model']['frozen_parameter_sha256'])
        attempts = run/'attempts'
        attempts.mkdir(exist_ok=True)
        number = len(list(attempts.glob('attempt_*_history.csv'))) + 1
        log = (attempts/f'attempt_{number:04d}_history.csv').open('x', encoding='utf-8', newline='')
        sample_log = (attempts/f'attempt_{number:04d}_signal_batches.csv').open('x', encoding='utf-8', newline='')
        access_log = (attempts/f'attempt_{number:04d}_access.csv').open('x', encoding='utf-8', newline='')
        attempt_streams = [log, sample_log, access_log]
        writer = csv.DictWriter(log, HISTORY_FIELDS, lineterminator='\n')
        writer.writeheader()
        sample_writer = None
        access_writer = None
        (run/'README.md').write_text('# W2 C formal adaptation: running\n\n'
            'Locked 10000-update pilot. Check progress.json / summary.json for the current state. '
            'Only training EEG signals; no stage labels, validation or test signals. '
            'The selected model is the new EMA after step10000. No downstream results yet.\n', encoding='utf-8')
        summary.update(status='running', completed_updates=step, ema_updates=ema.updates,
            reexecuted_lost_updates=extra_compute, attempt=number, frozen_parameters_unchanged=True)
        atomic_json(run/'summary.json', summary)
        atomic_json(run/'progress.json', summary)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        session_start = time.perf_counter()
        starting_step = step
        for step in range(step+1, config['training']['max_successful_optimizer_updates']+1):
            start = time.perf_counter()
            ns, na = len(provider.examples), len(provider.reader.accesses)
            batch = provider.next(step)
            for row in provider.examples[ns:]:
                if sample_writer is None:
                    sample_writer = csv.DictWriter(sample_log, list(row), lineterminator='\n')
                    sample_writer.writeheader()
                sample_writer.writerow(row)
            for row in provider.reader.accesses[na:]:
                if access_writer is None:
                    access_writer = csv.DictWriter(access_log, list(row), lineterminator='\n')
                    access_writer.writeheader()
                access_writer.writerow(row)
            sample_log.flush()
            access_log.flush()
            data_seconds = time.perf_counter()-start
            torch.cuda.synchronize()
            compute_start = time.perf_counter()
            clean = torch.from_numpy(batch).cuda()
            noise = torch.randn_like(clean)
            times = torch.randint(0, config['diffusion']['num_train_timesteps'], (len(clean),), device='cuda')
            condition = torch.full((len(clean), 1), config['input']['lead_condition'], dtype=torch.long, device='cuda')
            noisy = scheduler.add_noise(clean, noise, times)
            target = reference.diffusion_target(scheduler, config['diffusion']['prediction_type'], clean, noise, times)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy, times[:, None], condition)
            require(prediction.shape == target.shape, 'Diffusion target shape mismatch')
            loss = F.mse_loss(prediction, target)
            require(bool(torch.isfinite(loss)), 'Nonfinite loss')
            loss.backward()
            require(all(p.grad is not None for _, p in parameters), 'Missing locked parameter gradient')
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], config['training']['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            ema.update(model)
            require(ema.updates == step, 'EMA successful update counter differs')
            assert_trainable_finite(parameters, ema)
            torch.cuda.synchronize()
            compute_seconds = time.perf_counter()-compute_start
            entry = dict(step=step, loss=float(loss.detach()), gradient_norm_before_clip=float(norm),
                data_seconds=data_seconds, compute_seconds=compute_seconds, step_seconds=time.perf_counter()-start,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                timesteps=';'.join(map(str,times.cpu().tolist())), ema_updates=ema.updates)
            history.append(entry)
            writer.writerow(entry)
            log.flush()
            summary.update(completed_updates=step, ema_updates=ema.updates, last_loss=entry['loss'],
                last_update_at_Asia_Shanghai=now(), peak_allocated_mib=entry['peak_allocated_mib'],
                peak_reserved_mib=entry['peak_reserved_mib'])
            if step == 1 or step % 100 == 0:
                require(sha(__file__) == source_sha and sha(args.config) == config_sha, 'Source/config changed during run')
                validate_binding(args.binding, config_sha)
                assert_frozen(model, ema, config['model']['frozen_parameter_sha256'])
                elapsed = time.perf_counter()-session_start
                mean = elapsed/(step-starting_step)
                summary.update(status='running', session_elapsed_seconds=elapsed, mean_session_seconds_per_step=mean,
                    estimated_remaining_seconds=mean*(10000-step), sampled_subject_count=len({r['subject_id'] for r in provider.examples}))
                atomic_json(run/'progress.json', summary)
                atomic_json(run/'summary.json', summary)
                print(json.dumps(dict(time_Asia_Shanghai=now(), **entry, eta_minutes=summary['estimated_remaining_seconds']/60)), flush=True)
                write_evidence(run, provider, history)
            if step % config['checkpoints']['recovery_every_updates'] == 0:
                contract.validate_locked_files(args.config)
                info = save_recovery(run, model, optimizer, ema, provider, step, history, config_sha, binding_sha, extra_compute)
                summary['latest_recovery_step'] = info['step']
                atomic_json(run/'summary.json', summary)
        assert_frozen(model, ema, config['model']['frozen_parameter_sha256'])
        assert_trainable_finite(parameters, ema)
        require(step == ema.updates == 10000 and len(history) == 10000, 'Incomplete formal trajectory')
        require(len({r['subject_id'] for r in provider.examples}) == 54, 'Incomplete training subject coverage')
        require(all(r['member'] in reference.ALLOWED_MEMBERS and r['split']=='train' for r in provider.reader.accesses), 'Data isolation violation')
        common = dict(config_sha256=config_sha, runtime_source_sha256=source_sha, run_id=config['run_id'],
            completed_updates=step, original_checkpoint_sha256=config['initialization']['checkpoint_sha256'],
            adaptation_seed=config['sampling']['adaptation_seed'], selected_by='fixed_step_10000_no_external_validation')
        selected = atomic_save(run/config['checkpoints']['selected_final_filename'], dict(state_dict=cpu_tree(ema.state),
            ema_updates=ema.updates, weight_role='selected_new_ema', **common))
        online = atomic_save(run/config['checkpoints']['online_final_filename'], dict(state_dict=cpu_tree(model.state_dict()),
            weight_role='online_recovery_diagnostic_only', **common))
        write_evidence(run, provider, history)
        summary.update(status='completed', completed_at_Asia_Shanghai=now(), completed_updates=10000,
            ema_updates=10000, selected_weight=selected, online_last_weight=online,
            new_ema_parameter_sha256=parameter_digest(ema.state, [n for n,_ in model.named_parameters()]),
            frozen_online_sha256=reference.state_digest(model, frozen_only=True),
            sampled_subject_count=54, sample_occurrences=len(provider.examples),
            frozen_parameters_unchanged=True, all_recorded_losses_and_gradients_finite=True)
        (run/'README.md').write_text('# W2 C formal adaptation: completed\n\n'
            '10000 locked updates and 10000 new EMA updates completed. selected_ema_step10000.pt '
            'is the preregistered feature-extraction weight. See summary.json for SHA256. '
            'No sleep labels or validation/test signals were used. No downstream results have been produced yet.\n', encoding='utf-8')
        print(json.dumps(summary), flush=True)
        return summary
    except BaseException:
        summary.update(status='failed', failed_at_Asia_Shanghai=now(), error=traceback.format_exc(),
            attempted_step=step, ema_updates_at_failure=ema.updates,
            failure_action='abort_no_hyperparameter_change_no_resume_from_nonfinite_failure')
        (run/'README.md').write_text('# W2 C formal adaptation: failed\n\n'
            'See summary.json, preserved attempt logs and recovery files. '
            'No successful selected S1 checkpoint is claimed.\n', encoding='utf-8')
        raise
    finally:
        for stream in attempt_streams:
            stream.close()
        write_evidence(run, provider, history)
        atomic_json(run/'summary.json', summary)
        atomic_json(run/'progress.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=contract.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.config = args.config.resolve()
    validation = contract.validate_locked_files(args.config)
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not args.validate_only:
        require(args.binding is not None, 'Formal training requires tested runtime binding')
        validate_binding(args.binding, validation['config_sha256'])
    runtime = initialize(config, validation)
    if args.validate_only:
        require(args.output is not None and not args.output.exists(), 'Validation needs a new output path')
        atomic_json(args.output, runtime[-1])
        print(json.dumps(runtime[-1]), flush=True)
        return
    run_training(args, config, validation, runtime)


if __name__ == '__main__':
    main()
