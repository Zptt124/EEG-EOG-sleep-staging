"""Read-only verification of 160 published exploratory C classifier runs.

No training, model reselection or test access. Optional local model checkpoints
are checked but remain external to Git. B's pinned v1 evaluator defines metrics.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
REPORTS=[('week2_fixed50_20261031_v01',40),('week2_weighted50_20261031_v01',40),
         ('week2_layer_ablation_20261031_v01',80)]
EVALUATOR_SHA='4168c650b154f2c2a2c4ff26e8acd8466a41d3f6823dbaeabfa6eb07ab7f7605'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def verify(model_root=None):
    evaluator=ROOT/'evaluation_reference/evaluate.py'
    require(sha(evaluator)==EVALUATOR_SHA,'B evaluator changed')
    spec=importlib.util.spec_from_file_location('B_pinned_supplement_evaluator',evaluator)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    index=rows(ROOT/'features/eeg_cond14_v01/feature_index.csv')
    keys=['sample_id','subject_id','record_id','epoch_index','y_true']
    validation=[r for r in index if r['split']=='val']
    identity=[(r['sample_id'],r['subject_id'],r['record_id'],int(r['epoch_index']),int(r['y_true'])) for r in validation]
    require(len(identity)==21186 and len({r['subject_id'] for r in validation})==8,'Unexpected validation set')
    subsets=rows(ROOT/'manifests/week2_c_adaptation_v01/label_subsets_v2.csv')
    seen=set();checked=[];model_checks=0
    if model_root is not None:
        import torch
    for report,n in REPORTS:
        directory=ROOT/'reports'/report
        require(json.loads((directory/'summary.json').read_text(encoding='utf-8'))['status']=='complete','Incomplete experiment')
        require(json.loads((directory/'completion_audit.json').read_text(encoding='utf-8'))['status']=='passed','Failed original completion audit')
        registry=rows(directory/'run_registry.csv');require(len(registry)==n,'Wrong experiment run count')
        for row in registry:
            run=row['run_id'];require(Path(run).name==run and run.startswith('W2X_C_') and run not in seen,'Unsafe or duplicate run ID')
            seen.add(run);folder=ROOT/'runs'/run
            hashes=json.loads((folder/'artifact_sha256.json').read_text(encoding='utf-8'));files=0
            for name,expected in hashes.items():
                require(Path(name).name==name,'Unsafe archive path')
                if name.endswith(('.pt','.ckpt')):
                    if model_root is not None:require(sha(model_root/run/name)==expected,'External checkpoint SHA mismatch: '+run)
                else:
                    require(sha(folder/name)==expected,'Archive SHA mismatch: '+run+'/'+name);files+=1
            cfg=yaml.safe_load((folder/'config.yaml').read_text(encoding='utf-8'))
            history=json.loads((folder/'history.json').read_text(encoding='utf-8'))
            require(cfg['run_id']==run and cfg['selected_epoch']==50 and cfg['max_epochs']==50 and not cfg['early_stopping']
                    and not cfg['test_used'],'Invalid budget/selection/test policy: '+run)
            require([h['epoch'] for h in history]==list(range(1,51)),'Epoch sequence mismatch: '+run)
            require(all(np.isfinite(h['train_loss']) and np.isfinite(h['val_macro_f1']) for h in history),'Nonfinite history')
            require(cfg['optimizer']=='AdamW' and cfg['learning_rate']==.001 and cfg['weight_decay']==.0001
                    and cfg['batch_size']==128 and cfg['input_dim']==5120,'Classifier drift: '+run)
            selected={r['subject_id'] for r in subsets if r[cfg['budget']]=='1'}
            train=[r for r in index if r['split']=='train' and r['subject_id'] in selected]
            require(set(cfg['training_subjects'])==selected and cfg['training_samples']==len(train),'Labeled subset mismatch')
            if report=='week2_weighted50_20261031_v01':
                counts=np.bincount([int(r['y_true']) for r in train],minlength=5)
                weights=1./np.sqrt(counts.astype(np.float64));weights=(weights/weights.mean()).astype(np.float32)
                require(cfg['loss']=='inverse_sqrt_class_weighted_cross_entropy' and cfg['class_counts_labeled_train']==counts.tolist()
                        and cfg['class_weights']==weights.tolist(),'Training-only class weights mismatch: '+run)
            else:require(cfg['loss']=='unweighted_cross_entropy','Unexpected weighted objective')
            if report=='week2_layer_ablation_20261031_v01':
                layers={'last5':list(range(15,20)),'last2':list(range(18,20))}[cfg['layer_group']]
                require(cfg['EEG_layers_zero_based']==layers and cfg['EOG_layers_zero_based']==list(range(20))
                        and cfg['physical_input_dim']==5120 and cfg['active_EEG_dim']==128*len(layers),'Layer mask mismatch: '+run)
            frame=official.load_predictions(folder/'predictions_val.csv')
            require(list(frame[keys].itertuples(index=False,name=None))==identity and
                    np.isfinite(frame[official.PROB_COLS].to_numpy()).all(),'Prediction identity/finite mismatch: '+run)
            measured=official.compute_metrics(frame.y_true.to_numpy(),frame.y_pred.to_numpy())
            stored=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            require(all(stored[k]==v for k,v in measured.items()),'Official point metrics differ: '+run)
            require(float(row['val_macro_f1'])==measured['macro_f1'] and float(row['n1_f1'])==measured['n1_f1'],'Registry metrics differ')
            audit_file='prefix_replay_audit.json' if report=='week2_fixed50_20261031_v01' else 'paired_design_audit.json'
            require(json.loads((folder/audit_file).read_text(encoding='utf-8'))['status']=='passed','Original paired-design check failed')
            if model_root is not None:
                checkpoint=torch.load(model_root/run/'final_head_epoch50.pt',map_location='cpu',weights_only=True)
                require(checkpoint['epoch']==50 and checkpoint['config']==cfg and tuple(checkpoint['model']['weight'].shape)==(5,5120),
                        'Checkpoint/config mismatch: '+run)
                require(torch.isfinite(checkpoint['mean']).all() and torch.isfinite(checkpoint['std']).all()
                        and (checkpoint['std']>=1e-6).all(),'Invalid saved normalization')
                model_checks+=1
            checked.append(dict(run_id=run,experiment=report,uploaded_artifact_SHA_verified=files,
                                selected_epoch=50,validation_samples=21186,official_metrics_match=True))
    require(len(seen)==160,'Expected 160 new unique runs')
    for snapshot in ['source_snapshot.json','supplement_source_snapshot.json']:
        original=json.loads((ROOT/snapshot).read_text(encoding='utf-8'))['original_files']
        for entry in original:
            require(sha(ROOT/entry['path'])==entry['sha256'],'Source snapshot SHA differs: '+entry['path'])
    return dict(status='passed',scope='160 supplemental predictions, histories, configs and exact source snapshots',
        new_unique_runs=160,reused_reference_runs_not_counted_as_new=40,validation_samples_per_run=21186,
        validation_subjects=8,selected_epoch_all_runs=50,test_used=False,
        external_model_checkpoints_verified=model_checks,training_performed=False,checks=checked)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root',type=Path,help='Optional external directory containing run_id/final_head_epoch50.pt')
    parser.add_argument('--output',type=Path,help='Optional new JSON audit; never overwrites an existing file')
    args=parser.parse_args();require(not args.output or not args.output.exists(),'Choose a new audit output path')
    result=verify(args.model_root)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))


if __name__=='__main__':main()
