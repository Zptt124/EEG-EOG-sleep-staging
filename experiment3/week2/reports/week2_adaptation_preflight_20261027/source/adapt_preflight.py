"""Real unlabeled EEG gradient check and a separately reset 100-update benchmark.

This is diagnostic training, not a selected S1 model or downstream evaluation.
Uses the pinned original EMA backbone and its exact diffusion noise schedule.
Only signal.npy and epoch_index.npy can be read from training-subject NPZ files.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
UPSTREAM = ROOT / 'vendor/EEGDM'
SOURCE = ROOT / 'vendor/experiment1-review'
DATA = SOURCE / '交付包/实验①交付包_20261018/2_预处理数据与脚本_给B和C/data/processed'
CHECKPOINT_SHA = '0fe2d385d074834f3c22d9458798da49b2e5a58bd817284f130aae8814f5c1ef'
UPSTREAM_COMMIT = '003b5a2a016b2f02788ea5693d8b7cd00b6bd969'
PREFIXES = ('layers.18.', 'layers.19.', 'out_layer.')
ALLOWED_MEMBERS = frozenset(('signal.npy', 'epoch_index.npy'))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    require(bool(rows), 'Refusing to write an empty evidence table')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def validate_split(rows):
    from collections import Counter
    require(Counter(r['split'] for r in rows) == {'train': 54, 'val': 8, 'test': 16},
            'Expected original 54/8/16 subject split')
    require(len({r['subject_id'] for r in rows}) == 78, 'Duplicate split subject')
    record_split = {}
    train = []
    for row in rows:
        for record in row['record_ids'].split(';'):
            require(record.startswith(row['subject_id']) and record not in record_split,
                    'Invalid or duplicate record in split')
            record_split[record] = row['split']
        if row['split'] == 'train':
            train.append(row)
    return sorted(train, key=lambda r: r['subject_id']), record_split


class SignalOnlyReader:
    """A closed whitelist makes reading hidden labels impossible via this reader."""
    def __init__(self, directory, record_split):
        self.directory = Path(directory)
        self.record_split = record_split
        self.accesses = []

    def read_member(self, record, member):
        import numpy as np
        require(self.record_split.get(record) == 'train', 'Non-training record access rejected')
        require(member in ALLOWED_MEMBERS, 'Hidden labels or unapproved NPZ member access rejected')
        with zipfile.ZipFile(self.directory / f'{record}.npz') as archive:
            with archive.open(member) as stream:
                array = np.load(stream, allow_pickle=False)
        self.accesses.append(dict(record_id=record, split='train', member=member))
        return array


class TrainingSignalBatches:
    def __init__(self, reader, train_subjects, batch_size, seed):
        import numpy as np
        self.reader = reader
        self.subjects = train_subjects
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.order = []
        self.examples = []

    def next(self, step):
        import numpy as np
        from scipy.signal import resample_poly
        from src.util import staged_mu_law
        if not self.order:
            self.order = list(self.rng.permutation(len(self.subjects)))
        subject = self.subjects[self.order.pop()]
        records = subject['record_ids'].split(';')
        record = str(self.rng.choice(records))
        signal = self.reader.read_member(record, 'signal.npy')
        epochs = self.reader.read_member(record, 'epoch_index.npy')
        require(signal.ndim == 3 and signal.shape[1:] == (2, 3000), 'Expected [N,2,3000] A microvolt signal')
        require(len(signal) == len(epochs) and len(epochs) > 0, 'Signal/epoch index mismatch')
        require(epochs.dtype.kind in 'iu' and len(np.unique(epochs)) == len(epochs), 'Invalid original epoch index')
        indices = self.rng.choice(len(signal), self.batch_size, replace=len(signal) < self.batch_size)
        crops = self.rng.integers(0, 6, self.batch_size)
        x = np.asarray(signal[indices, 0, :], dtype=np.float32)
        require(bool(np.isfinite(x).all()), 'Nonfinite EEG signal')
        # Transform the complete original epoch before cropping, matching W1 filter boundaries.
        x = resample_poly(x, 2, 1, axis=-1).astype(np.float32)
        x = staged_mu_law(x / 100)  # Exactly once; never z-score.
        batch = np.stack([row[int(c)*1000:(int(c)+1)*1000] for row, c in zip(x, crops)])[:, None, :]
        require(batch.shape == (self.batch_size, 1, 1000) and bool(np.isfinite(batch).all()), 'Invalid 5-second input')
        for index, crop in zip(indices, crops):
            self.examples.append(dict(step=step, sample_id=f'{record}_{int(epochs[index]):04d}',
                subject_id=subject['subject_id'], record_id=record, epoch_index=int(epochs[index]),
                row_index=int(index), segment_index=int(crop), target_points=1000,
                source_split='train', y_hidden=-1))
        return np.ascontiguousarray(batch)


def state_digest(model, frozen_only=False):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if frozen_only and parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def diffusion_target(scheduler, prediction_type, clean, noise, times):
    if prediction_type == 'v_prediction':
        return scheduler.get_velocity(clean, noise, times)
    if prediction_type == 'epsilon':
        return noise
    if prediction_type == 'sample':
        return clean
    raise ValueError(f'Unsupported checkpoint prediction_type {prediction_type}')


def parameter_evidence(model, before):
    import torch
    rows = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            require(parameter.grad is None, f'Frozen parameter unexpectedly has gradient: {name}')
            continue
        gradient = parameter.grad
        require(gradient is not None and bool(torch.isfinite(gradient).all()), f'Missing/nonfinite gradient: {name}')
        current = parameter.detach().cpu()
        delta = current - before[name]
        require(bool(torch.isfinite(current).all()), f'Nonfinite updated parameter: {name}')
        rows.append(dict(name=name, shape=str(list(parameter.shape)), numel=parameter.numel(),
            gradient_l2=float(gradient.detach().norm()), gradient_abs_max=float(gradient.detach().abs().max()),
            gradient_nonzero=int(torch.count_nonzero(gradient)),
            update_l2=float(delta.norm()), update_abs_max=float(delta.abs().max()),
            changed_values=int(torch.count_nonzero(delta))))
    for prefix in PREFIXES:
        require(any(r['name'].startswith(prefix) and r['gradient_nonzero'] > 0
                    and r['changed_values'] > 0 for r in rows), f'No gradient and actual change in {prefix}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['gradient', 'benchmark'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gradient-report', type=Path)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    require(args.batch_size > 0 and args.lr > 0, 'Positive batch size and learning rate required')
    code_sha = sha_file(__file__)
    if args.mode == 'benchmark':
        require(args.gradient_report is not None, 'Benchmark requires prior passed gradient evidence')
        gate = json.loads(args.gradient_report.read_text(encoding='utf-8'))
        require(gate['status'] == 'passed' and gate['mode'] == 'gradient', 'Gradient check did not pass')
        require(gate['script_sha256'] == code_sha and gate['checkpoint_sha256'] == CHECKPOINT_SHA,
                'Gradient check source/weight mismatch')
        require(gate['batch_size'] == args.batch_size and gate['lr'] == args.lr and gate['seed'] == args.seed,
                'Benchmark settings differ from gradient check')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='running', mode=args.mode, script_sha256=code_sha,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed, formal_adaptation=False,
        downstream_evaluation=False, completed_updates=0)
    save_json(args.output / 'summary.json', summary)
    reader = None
    provider = None
    log = None
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        import yaml
        from diffusers import DDPMScheduler
        sys.path.insert(0, str(UPSTREAM))
        from model.diffusion_model import Wavenet
        from model import s4standalone
        require(torch.cuda.is_available(), 'CUDA is required for this benchmark')
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        checkpoint = ROOT / 'checkpoints/backbone.ckpt'
        require(sha_file(checkpoint) == CHECKPOINT_SHA, 'Original checkpoint hash mismatch')
        upstream_commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
        require(upstream_commit == UPSTREAM_COMMIT, 'Pinned upstream version changed')
        # The pinned checkout was created by Windows Git with CRLF. Normalize
        # text in this read-only check instead of mistaking line endings for edits.
        require(not subprocess.check_output(['git', '-c', 'core.autocrlf=true', '-C', str(UPSTREAM), 'status', '--porcelain'], text=True).strip(),
                'Upstream source has untracked or modified files')
        # Only the verified original artifact is trusted for pickle loading.
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        hparams = ckpt['hyper_parameters']
        model = Wavenet(**dict(hparams['model_kwargs']))
        prefix = 'ema.ema_model.'
        weights = {k[len(prefix):]: v for k, v in ckpt['state_dict'].items() if k.startswith(prefix)}
        model.load_state_dict(weights, strict=True)
        model.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if name.startswith(PREFIXES):
                parameter.requires_grad_(True)
        model.train().cuda()
        # A new process/model; do not import or invoke the frozen extractor's kernel cache.
        require(all(getattr(block.layer.kernel.forward, '__func__', None) is s4standalone.SSMKernelDiag.forward
                    for block in model.layers), 'S4 kernel forward has been wrapped/cached')
        noise_config = dict(hparams['noise_sch_kwargs'])
        scheduler = DDPMScheduler(**noise_config)
        prediction_type = noise_config['prediction_type']
        parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        trainable_before = {n: p.detach().cpu().clone() for n, p in parameters}
        initial_sha = state_digest(model)
        frozen_sha = state_digest(model, frozen_only=True)
        if args.mode == 'benchmark':
            require(initial_sha == gate['initial_parameter_sha256'], 'Benchmark did not reset to original starting weights')
        groups = []
        for name, parameter in parameters:
            special = getattr(parameter, '_optim', {})
            group = dict(params=[parameter], lr=args.lr, weight_decay=0.0)
            if 'lr' in special:
                group['lr'] = min(args.lr, special['lr'])
            groups.append(group)
        optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0, foreach=False)
        train_subjects, record_split = validate_split(read_csv(SOURCE / 'manifests/split_subjects.csv'))
        reader = SignalOnlyReader(DATA, record_split)
        provider = TrainingSignalBatches(reader, train_subjects, args.batch_size, args.seed)
        write_csv(args.output / 'adapt_subjects.csv', [dict(subject_id=r['subject_id'],
            record_ids=r['record_ids'], split='train', labels_available=False) for r in train_subjects])
        write_csv(args.output / 'trainable_parameters.csv', [dict(name=n, shape=str(list(p.shape)),
            numel=p.numel(), lr=g['lr'], weight_decay=g['weight_decay']) for (n, p), g in zip(parameters, groups)])
        (args.output / 'trainable_parameters.txt').write_text('\n'.join(n for n, _ in parameters) + '\n', encoding='utf-8')
        configuration = dict(mode=args.mode, batch_size=args.batch_size, seed=args.seed,
            checkpoint_sha256=CHECKPOINT_SHA, upstream_commit=upstream_commit,
            split_sha256=sha_file(SOURCE / 'manifests/split_subjects.csv'),
            script_sha256=code_sha, train_subjects=54, val_signals_used=False, test_signals_used=False,
            stage_labels_read=False, source_channel=0, source_unit='uV', source_hz=100, target_hz=200,
            preprocessing='30s resample_poly(2,1); uV/100; staged_mu_law once; uniform 5s crop',
            input_shape=[args.batch_size, 1, 1000], lead_condition=14, noise_scheduler=noise_config,
            original_ema_config=dict(hparams['ema_kwargs']),
            ema_policy='No new EMA for diagnostics. Original EMA initializes online model; formal training must reset and explicitly configure its own EMA.',
            optimizer='AdamW', learning_rate=args.lr, weight_decay=0.0, gradient_clip_norm=1.0,
            trainable_prefixes=list(PREFIXES), precision='float32; autocast disabled; TF32 disabled',
            s4_backend='official fallback; no frozen inference kernel memoization',
            deterministic_algorithms=True, benchmark_updates=100 if args.mode == 'benchmark' else 1,
            benchmark_warmup_policy='First 10 of 100 actual updates reported separately, no hidden warmup updates',
            reset_policy='Gradient check and benchmark run in separate processes, both load original EMA; formal training must reload original again',
            sample_policy='Cycle shuffled training subjects; uniform night, epoch row and 5s crop; no stage-based selection')
        (args.output / 'adapt_config.yaml').write_text(yaml.safe_dump(configuration, sort_keys=False), encoding='utf-8')
        environment = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
            cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
            total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
            nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,power.draw,temperature.gpu',
                '--format=csv,noheader'], text=True).strip(),
            cuda_extension=s4standalone.has_cuda_extension, pykeops=s4standalone.has_pykeops,
            git_upstream_commit=upstream_commit, script_sha256=code_sha,
            environment_variables={k: os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS')})
        save_json(args.output / 'environment.json', environment)
        gate_tensors = {}
        handles = []
        if args.mode == 'gradient':
            def capture_gate(index):
                def hook(module, inputs, result):
                    raw = result[0]
                    require(raw.requires_grad, f'No autograd graph at layer {index} gate input')
                    raw.retain_grad()
                    gate_tensors[index] = raw
                return hook
            handles = [model.layers[i].layer.register_forward_hook(capture_gate(i)) for i in (18, 19)]
        log = (args.output / 'adapt_history.csv').open('w', encoding='utf-8', newline='')
        writer = csv.DictWriter(log, ['step', 'loss', 'gradient_norm_before_clip', 'data_seconds',
            'compute_seconds', 'step_seconds', 'peak_allocated_mib', 'peak_reserved_mib', 'timesteps'], lineterminator='\n')
        writer.writeheader()
        history = []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        loop_start = time.perf_counter()
        n_steps = 1 if args.mode == 'gradient' else 100
        for step in range(1, n_steps + 1):
            start = time.perf_counter()
            batch = provider.next(step)
            data_seconds = time.perf_counter() - start
            torch.cuda.synchronize()
            compute_start = time.perf_counter()
            clean = torch.from_numpy(batch).cuda()
            noise = torch.randn_like(clean)
            times = torch.randint(0, noise_config['num_train_timesteps'], (len(clean),), device='cuda')
            condition = torch.full((len(clean), 1), 14, dtype=torch.long, device='cuda')
            noisy = scheduler.add_noise(clean, noise, times)
            target = diffusion_target(scheduler, prediction_type, clean, noise, times)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy, times[:, None], condition)
            require(prediction.shape == target.shape, 'Prediction target shape mismatch')
            loss = F.mse_loss(prediction, target)
            require(bool(torch.isfinite(loss)), 'Nonfinite diffusion loss')
            loss.backward()
            require(all(p.grad is not None for _, p in parameters), 'A selected parameter has no gradient')
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - compute_start
            entry = dict(step=step, loss=float(loss.detach()), gradient_norm_before_clip=float(norm),
                data_seconds=data_seconds, compute_seconds=compute_seconds,
                step_seconds=time.perf_counter() - start,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                timesteps=';'.join(map(str, times.cpu().tolist())))
            writer.writerow(entry)
            log.flush()
            history.append(entry)
            summary['completed_updates'] = step
            if step == 1 or step % 10 == 0:
                print(json.dumps(dict(mode=args.mode, **entry)), flush=True)
                save_json(args.output / 'progress.json', summary)
        loop_seconds = time.perf_counter() - loop_start
        log.close()
        log = None
        gates = []
        for index, tensor in gate_tensors.items():
            gradient = tensor.grad[..., :model.d_model]
            require(bool(torch.isfinite(gradient).all()) and bool(torch.count_nonzero(gradient)),
                    f'No finite nonzero gradient through layer{index} sigmoid gate input')
            gates.append(dict(layer=index, gate_pre_sigmoid_gradient_l2=float(gradient.norm()),
                gate_pre_sigmoid_gradient_abs_max=float(gradient.abs().max()),
                gate_pre_sigmoid_gradient_nonzero=int(torch.count_nonzero(gradient))))
        for handle in handles:
            handle.remove()
        evidence = parameter_evidence(model, trainable_before)
        write_csv(args.output / 'gradient_and_update.csv', evidence)
        require(state_digest(model, frozen_only=True) == frozen_sha, 'Frozen parameters changed')
        require(all(a['member'] in ALLOWED_MEMBERS and a['split'] == 'train' for a in reader.accesses), 'Isolation violation')
        seen_subjects = sorted({r['subject_id'] for r in provider.examples})
        summary.update(status='passed', checkpoint_sha256=CHECKPOINT_SHA,
            initial_parameter_sha256=initial_sha, final_parameter_sha256=state_digest(model),
            frozen_parameter_sha256_before=frozen_sha, frozen_parameter_sha256_after=state_digest(model, True),
            frozen_parameters_unchanged=True, frozen_parameter_gradients_absent=True,
            trainable_parameter_count=sum(p.numel() for _, p in parameters),
            total_parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_tensor_count=len(parameters),
            changed_trainable_tensors=sum(r['changed_values'] > 0 for r in evidence), gate_checks=gates,
            all_trainable_gradients_finite=True, loss_first=history[0]['loss'], loss_last=history[-1]['loss'],
            prediction_type=prediction_type, train_subject_pool=54, sampled_subject_count=len(seen_subjects),
            sampled_subjects=seen_subjects, sampled_record_count=len({r['record_id'] for r in provider.examples}),
            samples_processed=len(provider.examples), signal_member_reads=len(reader.accesses),
            stage_labels_read=False, validation_signals_read=False, test_signals_read=False,
            fresh_original_initialization=True, new_ema_updated=False, mixed_precision=False,
            loop_wall_seconds=loop_seconds, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            mean_compute_seconds=float(np.mean([r['compute_seconds'] for r in history])),
            mean_data_seconds=float(np.mean([r['data_seconds'] for r in history])),
            mean_step_seconds=float(np.mean([r['step_seconds'] for r in history])))
        if args.mode == 'benchmark':
            require(len(seen_subjects) == 54, 'Benchmark did not cover every training subject')
            steady = history[10:]
            compute = np.array([r['compute_seconds'] for r in steady])
            elapsed = np.array([r['step_seconds'] for r in steady])
            summary.update(gradient_report_sha256=sha_file(args.gradient_report), warmup_updates=10,
                steady_updates=90, steady_compute_mean_seconds=float(compute.mean()),
                steady_compute_median_seconds=float(np.median(compute)),
                steady_compute_p95_seconds=float(np.percentile(compute, 95)),
                steady_step_mean_seconds=float(elapsed.mean()),
                steady_compute_crops_per_second=float(args.batch_size/compute.mean()),
                steady_end_to_end_crops_per_second=float(args.batch_size/elapsed.mean()),
                projected_1000_updates_minutes=float(elapsed.mean()*1000/60),
                projected_10000_updates_hours=float(elapsed.mean()*10000/3600))
        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), configuration=configuration,
            completed_updates=n_steps, original_checkpoint_sha256=CHECKPOINT_SHA,
            diagnostic_only=True, formal_training_must_reset=True), args.output / 'diagnostic_last.pt')
        summary['diagnostic_checkpoint_sha256'] = sha_file(args.output / 'diagnostic_last.pt')
        summary['diagnostic_checkpoint_bytes'] = (args.output / 'diagnostic_last.pt').stat().st_size
        readme = (f"# W2 C {args.mode}: passed\n\n"
            f"Real EEG only; {n_steps} diagnostic updates, batch={args.batch_size}, FP32. "
            f"Original EMA reloaded in this process; condition14, target={prediction_type}.\n\n"
            "Only 54 training subjects are eligible; no sleep labels, validation or test signals were read. "
            "Gate/backbone gradient evidence, frozen checksums, exact parameter names and signal accesses are recorded.\n\n"
            "The 100-update benchmark includes its first 10 warmup updates. Its weights are diagnostic only, "
            "not a selected S1 result. Formal adaptation must reload the original EMA and explicitly lock its "
            "total steps, stopping and new EMA policy; no downstream or N1 improvement was evaluated.\n")
        (args.output / 'README.md').write_text(readme, encoding='utf-8')
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
        (args.output / 'README.md').write_text('# Diagnostic failed\n\nSee summary.json and preserved logs. '
            'A failed check does not authorize the benchmark.\n', encoding='utf-8')
        raise
    finally:
        if log is not None:
            log.close()
        if provider is not None and provider.examples:
            write_csv(args.output / 'signal_batches.csv', provider.examples)
        if reader is not None and reader.accesses:
            write_csv(args.output / 'signal_access_audit.csv', reader.accesses)
        save_json(args.output / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
