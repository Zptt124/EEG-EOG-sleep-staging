"""Prepare/probe/extract S1 features only from the preregistered final EMA.

The prepare and probe modes never create the full S1 cache. All extraction
settings and transforms reuse the frozen W1 extractor; stale S0 kernels cannot
be inherited. No NPZ labels are read; labels in the index are copied from S0.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

import adapt_preflight as reference
import extract_features as legacy
import validate_adapt_lock as contract

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT/'configs/features_W2_C_S1_pilot_v01.yaml'
INDEX_FIELDS = ['sample_id','subject_id','record_id','epoch_index','split','y_true','feature_path','row_index','config_id']


def require(value,message):
    reference.require(value,message)


def sha(path):
    return contract.sha(path)


def atomic_json(path,value):
    path=Path(path)
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(path)


def read_csv(path):
    return reference.read_csv(path)


def validate_feature_contract(config):
    require(config['accepted_splits']==['train','val'] and not config['test_signals_or_features'],'Feature split policy changed')
    require(not config['model_receives_sleep_labels'] and config['source_npz_members_read']==['signal.npy','epoch_index.npy'],
            'Feature model/NPZ label access policy changed')
    require(config['required_optimizer_and_ema_updates']==10000 and config['selected_weight_role']=='selected_new_ema',
            'Final weight selection policy changed')
    require(config['source_channel']==0 and config['source_hz']==100 and config['target_hz']==200,'Input channel/rate changed')
    require(config['lead_condition']==14 and config['diffusion_t']==1 and not config['random_input_noise'],'Feature condition/noise changed')
    require(config['feature_shape']==[20,30,128] and config['query']=='gate','Feature layers/query changed')
    require(config['window_points']==config['window_step']==200 and config['reduction']=='std_torch_correction1','Feature reducer changed')
    require(config['storage_dtype']=='float16' and config['precision']=='float32','Feature precision changed')
    require(not config['overwrite_S0'] and not config['reuse_S0_kernel_cache'],'S0/cache policy violated')
    require(config['batch_size']==4 and config['device']=='cuda','Feature runtime differs from prepared setting')


def check_completed_training(summary,config):
    require(summary.get('status')=='completed','Final adaptation training is not completed')
    require(summary.get('completed_updates')==summary.get('ema_updates')==10000,'Final online/EMA update counts differ')
    require(summary.get('config_sha256')==config['adaptation_config_sha256'],'Adaptation configuration differs')
    require(summary.get('frozen_parameters_unchanged') is True,'Frozen parameter audit did not pass')
    require(not any(summary.get(k,True) for k in ['stage_labels_read','validation_signals_read','test_signals_read']),
            'Adaptation data isolation evidence missing')
    selected=summary.get('selected_weight',{})
    require(selected.get('path')==config['selected_filename'],'Wrong selected weight path (intermediate/online not allowed)')
    return selected


def check_selected_metadata(checkpoint,summary,config):
    require(checkpoint.get('weight_role')=='selected_new_ema','Checkpoint is not the selected new EMA')
    require(checkpoint.get('completed_updates')==checkpoint.get('ema_updates')==10000,'Checkpoint is not final step10000')
    require(checkpoint.get('config_sha256')==summary['config_sha256'],'Checkpoint configuration mismatch')
    require(checkpoint.get('runtime_source_sha256')==summary['trainer_sha256'],'Checkpoint trainer version mismatch')
    require(checkpoint.get('run_id')==summary['run_id'],'Checkpoint run mismatch')
    require(checkpoint.get('original_checkpoint_sha256')==config['original_checkpoint_sha256'],'Checkpoint starting source mismatch')
    require(checkpoint.get('selected_by')=='fixed_step_10000_no_external_validation','Backbone selection rule changed')
    require(isinstance(checkpoint.get('state_dict'),dict),'Checkpoint lacks a full model state')


def validate_index(rows,partitions,strict=True):
    require(bool(rows),'Empty S0 feature index')
    ids=[r['sample_id'] for r in rows]
    require(len(ids)==len(set(ids)),'Duplicate sample_id in S0 index')
    grouped=defaultdict(list)
    subjects=defaultdict(set)
    for row in rows:
        rec=row['record_id']; split=row['split']; epoch=int(row['epoch_index'])
        require(split in ['train','val'] and partitions.get(rec)==split,'Forbidden or mismatched record split')
        require(row['subject_id']==rec[:5] and row['sample_id']==f'{rec}_{epoch:04d}','Index identity mismatch')
        require(int(row['y_true']) in range(5),'Invalid index label metadata')
        require(row['feature_path']==f'{rec}.npy','Unsafe feature filename')
        subjects[split].add(row['subject_id'])
        grouped[rec].append(row)
    require(not subjects['train'] & subjects['val'],'Overlapping feature subjects')
    for rec,items in grouped.items():
        require([int(r['row_index']) for r in items]==list(range(len(items))),'Rows are not the original per-record order')
        require(len({int(r['epoch_index']) for r in items})==len(items),'Duplicate original epoch index')
    if strict:
        require(Counter(r['split'] for r in rows)=={'train':132938,'val':21186},'S0 sample count changed')
        require(len(grouped)==122 and len(subjects['train'])==54 and len(subjects['val'])==8,'S0 record/subject count changed')
        require(set(grouped)=={r for r,s in partitions.items() if s in ['train','val']},'S0 index record coverage changed')
    return dict(grouped)


def safe_output_root(config):
    out=contract.resolve_path(config['S1_feature_root'])
    baseline=contract.resolve_path(config['S0_feature_root'])
    require(out!=baseline and not out.is_relative_to(baseline),'S0 cache overwrite/descendant forbidden')
    require(out.parent==ROOT/'features','Unexpected S1 cache location')
    return out


def load_signal_record(directory,record,partitions,expected_rows):
    require(partitions.get(record) in ['train','val'],'Test/unknown signal access forbidden')
    arrays={}
    with zipfile.ZipFile(Path(directory)/f'{record}.npz') as archive:
        for member in ['signal.npy','epoch_index.npy']:
            with archive.open(member) as stream: arrays[member]=np.load(stream,allow_pickle=False)
    signal=arrays['signal.npy']; epochs=arrays['epoch_index.npy']
    require(signal.shape==(len(expected_rows),2,3000),'Signal shape or row coverage mismatch')
    require(epochs.dtype.kind in 'iu' and np.array_equal(epochs,np.array([int(r['epoch_index']) for r in expected_rows])),
            'Original epoch row order mismatch')
    return signal[:,0,:]


def load_selected(config_path=DEFAULT_CONFIG):
    config_path=Path(config_path).resolve()
    config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    validate_feature_contract(config)
    safe_output_root(config)
    parent=contract.resolve_path(config['adaptation_config'])
    validation=contract.validate_locked_files(parent)
    require(validation['config_sha256']==config['adaptation_config_sha256'],'Parent lock mismatch')
    run=contract.resolve_path(config['adaptation_run'])
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    meta=check_completed_training(summary,config)
    path=run/meta['path']
    require(path.parent==run and path.is_file() and sha(path)==meta['sha256'],'Selected weight missing or SHA mismatch')
    original=contract.resolve_path(config['original_checkpoint'])
    require(sha(original)==config['original_checkpoint_sha256'],'Original checkpoint changed')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    check_selected_metadata(checkpoint,summary,config)
    base=torch.load(original,map_location='cpu',weights_only=False)  # SHA-verified original artifact.
    from model.diffusion_model import Wavenet
    model=Wavenet(**dict(base['hyper_parameters']['model_kwargs']))
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    trainable=set(contract.resolve_path('manifests/week2_c_adaptation_v01/trainable_parameters.txt').read_text().splitlines())
    for name,p in model.named_parameters(): p.requires_grad_(name in trainable)
    parent_config=yaml.safe_load(parent.read_text(encoding='utf-8'))
    require(reference.state_digest(model,True)==parent_config['model']['frozen_parameter_sha256'],'Selected EMA changed frozen parameters')
    require(reference.state_digest(model)==summary['new_ema_parameter_sha256'],'Selected EMA parameter digest differs')
    model.requires_grad_(False).eval()
    split=read_csv(contract.resolve_path(config['source_split']))
    _,partitions=reference.validate_split(split)
    index=read_csv(contract.resolve_path(config['reference_index']))
    grouped=validate_index(index,partitions)
    preparation=dict(status='ready_not_extracted',config_id=config['config_id'],
        feature_config_sha256=sha(config_path),adaptation_config_sha256=validation['config_sha256'],
        adaptation_run=summary['run_id'],completed_adaptation_updates=10000,
        selected_weight_path=path.relative_to(ROOT).as_posix(),selected_weight_sha256=meta['sha256'],
        selected_weight_role='selected_new_ema',selected_parameter_sha256=summary['new_ema_parameter_sha256'],
        source_index_sha256=sha(contract.resolve_path(config['reference_index'])),
        source_S0_config_sha256=sha(contract.resolve_path(config['S0_feature_root'])/'feature_config.json'),
        extractor_script_sha256=sha(__file__),legacy_extractor_sha256=sha(ROOT/'scripts/extract_features.py'),
        records=122,samples=154124,split_counts={'train':132938,'val':21186},
        raw_feature_bytes_estimate=154124*20*30*128*2+122*128,
        pooled_feature_bytes_estimate=154124*2560*4+128,
        full_extraction_started=False,S0_overwritten=False,test_signals_used=False)
    return config,preparation,model,index,grouped,partitions


class AdaptedExtractor(legacy.Extractor):
    def __init__(self,model,condition=14):
        require(torch.cuda.is_available(),'CUDA required')
        require(not model.training and not any(p.requires_grad for p in model.parameters()),'Selected model must be frozen eval')
        torch.set_num_threads(4)
        self.model=model.cuda()
        self.condition=condition
        from model.classifier import LatentActivityExtractor,LatentActivityReducer
        self.extractor=LatentActivityExtractor(self.model,diffusion_t=1,query=['gate'],use_cond=[condition]).eval().cuda()
        self.reducers={n:LatentActivityReducer(query=['gate'],reduce=['std'],L=n,
            window_size=200,window_step=200).eval().cuda() for n in (1000,6000)}
        # Inherited memoization is enabled only AFTER loading this new selected model.


def pooled_features(features):
    require(features.ndim==4 and features.shape[1:]==(20,30,128),'Unexpected raw feature shape')
    values=np.asarray(features,dtype=np.float32)
    require(bool(np.isfinite(values).all()),'Nonfinite feature values')
    return values.mean(axis=2,dtype=np.float32).reshape(len(values),2560)


def check_binding(binding_path,preparation):
    binding=json.loads(Path(binding_path).read_text(encoding='utf-8'))
    require(binding['status']=='ready_not_extracted','Preparation binding is not ready')
    for key in ['feature_config_sha256','adaptation_config_sha256','selected_weight_sha256','source_index_sha256',
                'source_S0_config_sha256','extractor_script_sha256','legacy_extractor_sha256']:
        require(binding[key]==preparation[key],f'Extraction binding mismatch: {key}')
    return binding


def probe(config,preparation,model,index,grouped,partitions,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    chosen=[next(rec for rec,rows in sorted(grouped.items()) if rows[0]['split']==split) for split in ['train','val']]
    signals=[]; sample_rows=[]; original_features=[]
    for rec in chosen:
        x=load_signal_record(contract.resolve_path(config['source_data_root']),rec,partitions,grouped[rec])
        signals.append(x[:1]); sample_rows.append(dict(grouped[rec][0]))
        original_features.append(np.load(contract.resolve_path(config['S0_feature_root'])/f'{rec}.npy',mmap_mode='r',allow_pickle=False)[0].copy())
    batch=np.concatenate(signals)
    extractor=AdaptedExtractor(model,config['lead_condition'])
    uncached=extractor(batch)
    extractor.cache_frozen_kernels()
    cached=extractor(batch)
    np.testing.assert_allclose(cached,uncached,rtol=1e-5,atol=1e-6)
    stored=cached.astype(np.float16)
    require(bool(np.isfinite(stored).all()),'Probe float16 overflow')
    baseline=np.stack(original_features).astype(np.float32)
    early_difference=np.abs(stored[:,:18].astype(np.float32)-baseline[:,:18])
    require(np.allclose(stored[:,:18].astype(np.float32),baseline[:,:18],rtol=0.002,atol=1e-5),
            'Frozen early layer features unexpectedly differ from S0')
    pooled=pooled_features(stored)
    np.save(output/'S1_probe_raw.npy',stored,allow_pickle=False)
    np.save(output/'S1_probe_pooled.npy',pooled,allow_pickle=False)
    reference.write_csv(output/'sample_index.csv',sample_rows)
    report=dict(status='passed',selected_weight_sha256=preparation['selected_weight_sha256'],samples=2,
        splits=['train','val'],raw_shape=list(stored.shape),raw_dtype=str(stored.dtype),pooled_shape=list(pooled.shape),
        pooled_dtype=str(pooled.dtype),cached_uncached_max_abs_diff=float(np.abs(cached-uncached).max()),
        frozen_first18_layers_max_abs_diff_vs_S0_float16=float(early_difference.max()),
        last2_layers_max_abs_diff_vs_S0=float(np.abs(stored[:,18:].astype(np.float32)-baseline[:,18:]).max()),
        all_finite=True,new_frozen_kernel_cache_only=True,NPZ_labels_read=False,model_received_labels=False,
        full_extraction_started=False,S0_overwritten=False,backbone_selection_already_fixed=True)
    atomic_json(output/'probe_audit.json',report)
    return report


def extract(config,preparation,model,index,grouped,partitions,binding_path,resume=False):
    check_binding(binding_path,preparation)
    out=safe_output_root(config)
    require(torch.cuda.is_available(),'CUDA required before starting extraction')
    if resume:
        require(out.is_dir(),'No extraction to resume')
    else:
        out.mkdir(parents=True,exist_ok=False)
    with (out/'.extract.lock').open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return extract_locked(config,preparation,model,index,grouped,partitions,out,resume)


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
        representation='S1',adaptation_seed=0,adaptation_steps=10000,weight_role='selected_new_ema',
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','probe','extract'],required=True)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path,help='New preparation JSON path or new probe folder')
    parser.add_argument('--binding',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    validate_feature_contract(config)
    status=json.loads((contract.resolve_path(config['adaptation_run'])/'summary.json').read_text(encoding='utf-8'))
    if status['status']!='completed':
        print(json.dumps(dict(status='waiting_for_final_weight',training_status=status['status'],
            completed_updates=status['completed_updates'],full_extraction_started=False)))
        return
    config,preparation,model,index,grouped,partitions=load_selected(args.config)
    if args.mode=='prepare':
        require(args.output is not None and not args.output.exists(),'Prepare needs a new binding JSON path')
        atomic_json(args.output,preparation); result=preparation
    elif args.mode=='probe':
        require(args.output is not None,'Probe needs a new output directory')
        result=probe(config,preparation,model,index,grouped,partitions,args.output)
    else:
        require(args.binding is not None,'Full extraction requires a checked preparation binding')
        result=extract(config,preparation,model,index,grouped,partitions,args.binding,args.resume)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
