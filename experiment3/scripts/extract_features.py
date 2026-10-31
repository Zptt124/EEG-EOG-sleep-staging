"""Frozen EEGDM extraction for A's unstandardized microvolt NPZ files.

Only train/val are accepted. Output axes are sample,layer,time_window,hidden.
No labels are supplied to the backbone. Checkpoint is pinned by SHA256.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'vendor/EEGDM'
SOURCE = ROOT.parent
DATA = SOURCE / '交付包/实验①交付包_20261018/2_预处理数据与脚本_给B和C/data/processed'
SHA = '0fe2d385d074834f3c22d9458798da49b2e5a58bd817284f130aae8814f5c1ef'
sys.path.insert(0, str(UPSTREAM))
from model.diffusion_model import Wavenet
from model.classifier import LatentActivityExtractor, LatentActivityReducer
from src.util import staged_mu_law


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def record_splits():
    result = {}
    for row in read_csv(SOURCE / 'manifests/split_subjects.csv'):
        for rec in row['record_ids'].split(';'):
            if rec in result or rec[:5] != row['subject_id']:
                raise ValueError('Invalid subject split')
            result[rec] = row['split']
    return result


def transform(signal_uv):
    x = np.asarray(signal_uv, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3000 or not np.isfinite(x).all():
        raise ValueError('Expected finite (batch,3000) microvolt input')
    x = resample_poly(x, 2, 1, axis=-1).astype(np.float32)
    # Official staged mu-law keeps |x|<=1 unchanged; it is not full mu-law.
    return np.ascontiguousarray(staged_mu_law(x / 100)[:, None, :])


class Extractor:
    def __init__(self, condition=14):
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for this extraction configuration')
        path = ROOT / 'checkpoints/backbone.ckpt'
        with path.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != SHA:
                raise ValueError('Unexpected checkpoint')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = Wavenet(**ckpt['hyper_parameters']['model_kwargs'])
        prefix = 'ema.ema_model.'
        model.load_state_dict({k[len(prefix):]: v for k,v in ckpt['state_dict'].items()
                               if k.startswith(prefix)}, strict=True)
        model.requires_grad_(False).eval().cuda()
        self.model = model
        self.condition = condition
        self.extractor = LatentActivityExtractor(model, diffusion_t=1, query=['gate'],
                                                use_cond=[condition]).eval().cuda()
        self.reducers = {n: LatentActivityReducer(query=['gate'], reduce=['std'], L=n,
                         window_size=200, window_step=200).eval().cuda() for n in (1000,6000)}

    def cache_frozen_kernels(self):
        """Memoize input-independent S4 kernels only in frozen, stateless inference.

        The model must not be moved, cast or updated after enabling this cache.
        Each wrapper owns its original forward and cache via a factory closure.
        """
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Kernel caching requires a frozen eval model')
        def wrap(original):
            cache = {}
            def forward(*, L=None, rate=1.0, state=None):
                if state is not None or self.model.training or torch.is_grad_enabled():
                    raise RuntimeError('Only stateless inference supported by cache')
                key = (L,rate)
                if key not in cache:
                    cache[key] = original(L=L,rate=rate,state=None)
                return cache[key]
            return forward
        for block in self.model.layers:
            block.layer.kernel.forward = wrap(block.layer.kernel.forward)

    @torch.inference_mode()
    def transformed(self, x, mode='native'):
        t = torch.from_numpy(x).cuda()
        if mode == 'native':
            y = self.reducers[6000](self.extractor((t, None)))[:, :, 0, :, 0, 0, :]
        elif mode == 'explicit':
            y = torch.cat([self.reducers[1000](self.extractor((t[:,:,i:i+1000],None)))
                           [:,:,0,:,0,0,:] for i in range(0,6000,1000)], dim=2)
        else:
            raise ValueError(mode)
        y = y.cpu().numpy()
        if y.shape != (len(x),20,30,128) or not np.isfinite(y).all():
            raise ValueError('Invalid feature output')
        return y

    def __call__(self, signal_uv, mode='native'):
        return self.transformed(transform(signal_uv),mode)


def config(modality, condition):
    return dict(version='E3_features_v01', upstream_commit='003b5a2a016b2f02788ea5693d8b7cd00b6bd969',
                source_commit='02f120ee80c1c7a7822c33746ce92017632f1faa', checkpoint_sha256=SHA,
                modality=modality, source_unit='uV', source_fs=100, target_fs=200,
                resampling='scipy.signal.resample_poly(up=2,down=1), default filter',
                amplitude='uV / 100 then official staged_mu_law(mu=255), once',
                z_score=False, condition=condition, diffusion_t=1, random_noise=False,
                query='gate', layers=list(range(20)), reduction='std, torch correction=1',
                window_points=200, stride_points=200, length_points=6000,
                internal_segments=6, continuous_30s_context=False,
                axes=['sample','layer','time_window','hidden'], feature_shape=[20,30,128],
                computation_dtype='float32', storage_dtype='float16', seed=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modality', choices=['eeg','eog'], default='eeg')
    p.add_argument('--condition', type=int, choices=[10,14,18], default=14)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--output', type=Path)
    p.add_argument('--record-limit', type=int)
    p.add_argument('--cache-kernels', action='store_true')
    args = p.parse_args()
    cfg = config(args.modality,args.condition)
    out = args.output or ROOT / 'features' / f'{args.modality}_cond{args.condition}_v01'
    out.mkdir(parents=True,exist_ok=True)
    cfgpath = out / 'feature_config.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text()) != cfg:
        raise ValueError('Output configuration mismatch; use a new output directory')
    cfgpath.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    torch.manual_seed(0)
    extractor = Extractor(args.condition)
    if args.cache_kernels:
        extractor.cache_frozen_kernels()
    splits = record_splits()
    records = [(r,s) for r,s in sorted(splits.items()) if s in ('train','val')]
    if args.record_limit:
        records = records[:args.record_limit]
    index, profiles = [], []
    for rec, partition in records:
        target = out / f'{rec}.npy'
        done = out / f'{rec}.done.json'
        with np.load(DATA / f'{rec}.npz',allow_pickle=False) as z:
            labels, epochs = z['labels'], z['epoch_index']
            n = len(labels)
            if done.exists():
                profile = json.loads(done.read_text())
                cached = np.load(target,mmap_mode='r',allow_pickle=False)
                if cached.shape != (n,20,30,128) or cached.dtype != np.float16:
                    raise ValueError(f'Invalid completed cache {rec}')
                del cached
            else:
                signals = z['signal'][:,0 if args.modality=='eeg' else 1,:]
                tmp = out / f'{rec}.partial.npy'
                dest = np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float16,shape=(n,20,30,128))
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start=time.perf_counter()
                for b in range(0,n,args.batch_size):
                    f = extractor(signals[b:b+args.batch_size])
                    stored=f.astype(np.float16)
                    if not np.isfinite(stored).all():
                        raise ValueError('float16 overflow')
                    dest[b:b+len(f)] = stored
                dest.flush()
                del dest, signals
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                tmp.replace(target)
                profile=dict(record_id=rec,split=partition,n_samples=n,seconds=elapsed,
                             samples_per_second=n/elapsed,
                             peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20,
                             bytes=target.stat().st_size,batch_size=args.batch_size,
                             cache_frozen_kernels=args.cache_kernels)
                done.write_text(json.dumps(profile,indent=2),encoding='utf-8')
            profiles.append(profile)
            for row,(epoch,label) in enumerate(zip(epochs,labels)):
                index.append(dict(sample_id=f'{rec}_{int(epoch):04d}',subject_id=rec[:5],
                    record_id=rec,epoch_index=int(epoch),split=partition,y_true=int(label),
                    feature_path=target.name,row_index=row,config_id=cfg['version']))
            print(json.dumps(profile),flush=True)
        write_csv(out / 'feature_index.csv',index)
        write_csv(out / 'resource_profile.csv',profiles)
    assert len({r['sample_id'] for r in index}) == len(index)
    summary=dict(status='complete',records=len(records),samples=len(index),test_used=False,
                 split_counts={s:sum(r['split']==s for r in index) for s in ('train','val')},
                 total_feature_bytes=sum(p['bytes'] for p in profiles))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
