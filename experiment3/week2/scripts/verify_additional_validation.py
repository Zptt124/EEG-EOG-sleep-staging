"""Recheck the two published validations from Git evidence and Release ZIPs.

No extraction, network, training, model selection, raw signals, or test access.
Requires NumPy/Pandas; --inspect-backbones also requires PyTorch.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MULTI='EEG_multisubset_20261031_v02'
HELD='subject_holdout_20261101_v01'
EVALUATOR_SHA='4168c650b154f2c2a2c4ff26e8acd8466a41d3f6823dbaeabfa6eb07ab7f7605'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def identity(frame):return list(frame[['sample_id','subject_id','record_id','epoch_index','y_true']].itertuples(index=False,name=None))


def verify(assets,inspect_backbones=False):
    manifest=json.loads((ROOT/'additional_source_snapshot.json').read_text(encoding='utf-8'))
    require(manifest['new_final_models']==96 and manifest['new_inner_fits']==216,'Wrong study matrix')
    for item in manifest['original_main_files']:
        p=ROOT/item['path']
        require(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Main snapshot differs: '+item['path'])
    zips={}
    for asset in manifest['release_assets']:
        p=assets/asset['filename']
        require(p.stat().st_size==asset['bytes'] and sha(p)==asset['sha256'],'Release asset differs: '+p.name)
        zips[p.name]=zipfile.ZipFile(p)
    try:
        members={}
        for item in manifest['release_files']:
            relative=Path(item['path'])
            require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe relative archive path')
            archive=zips[item['asset']];info=archive.getinfo(item['member'])
            with archive.open(item['member']) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            require(info.file_size==item['bytes'] and actual==item['sha256'],'ZIP member differs: '+item['path'])
            members[item['path']]=(archive,item)
        require(len(members)==len(manifest['release_files']),'Duplicate archive member')
        def content(relative):
            archive,item=members[relative];return archive.read(item['member'])
        evaluator=ROOT/'evaluation_reference/evaluate.py'
        require(sha(evaluator)==EVALUATOR_SHA,'Pinned B evaluator changed')
        spec=importlib.util.spec_from_file_location('B_additional_publication_v1',evaluator)
        official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
        def frame(relative):
            value=official.load_predictions(io.BytesIO(content(relative)))
            require(np.isfinite(value[official.PROB_COLS].to_numpy()).all(),'Nonfinite probabilities')
            return value
        def metric_check(value,stored):
            measured=official.compute_metrics(value.y_true.to_numpy(),value.y_pred.to_numpy())
            require(all(stored[k]==v for k,v in measured.items()),'B point metric mismatch')
            return measured
        index=pd.read_csv(ROOT/'features/eeg_cond14_v01/feature_index.csv')
        require(len(index)==154124 and set(index.split)=={'train','val'},'Reference identity mismatch')
        val=index[index.split=='val'];train=index[index.split=='train']
        require(len(val)==21186 and val.subject_id.nunique()==8 and len(train)==132938 and train.subject_id.nunique()==54,'Subject/sample coverage mismatch')
        multi=ROOT/'reports'/MULTI;held=ROOT/'reports'/HELD
        require(json.loads((multi/'summary.json').read_text())['status']=='complete','Multi-roster incomplete')
        require(json.loads((held/'summary.json').read_text())['status']=='complete','Holdout incomplete')
        require(json.loads((multi/'validation_embargo_release.json').read_text())['validation_evaluations_before_release']==0,'Multi embargo violation')
        require(json.loads((held/'outer_prediction_release.json').read_text())['outer_evaluations_before_release']==0,'Outer embargo violation')
        for directory,count in [(multi,78),(held,234)]:
            registry=read(directory/'optimizer_fit_registry.csv')
            require(len(registry)==count and all(float(r['gradient_inf'])<=1e-5 for r in registry),'Unconverged/incomplete optimizer registry')
        subsets=read(multi/'label_subsets.csv');single={};official_checks=0
        for entry in read(multi/'run_registry.csv'):
            run=entry['run_id'];relative=f'reports/{MULTI}/predictions/{run}.csv'
            require(hashlib.sha256(content(relative)).hexdigest()==entry['predictions_sha256'],'Original prediction SHA mismatch')
            value=frame(relative);require(identity(value)==identity(val),'Multi-roster validation identity/order mismatch')
            metric_check(value,json.loads((multi/'evaluation'/(run+'.json')).read_text()))
            cfg=json.loads((multi/'final_models'/run/'config.json').read_text())
            selected={r['subject_id'] for r in subsets if int(r['roster_seed'])==int(entry['roster_seed']) and r[entry['budget']]=='1'}
            require(set(cfg['normalization_fitted_subjects'])==selected and not selected&set(val.subject_id),'Multi normalization leakage')
            require(not cfg['development_validation_used'] and not cfg['test_used'],'Multi training/evaluation policy changed')
            single[run]=value;official_checks+=1
        require(len(single)==24,'Missing multi-roster predictions')
        assignment={r['subject_id']:int(r['heldout_fold']) for r in read(held/'outer_subjects.csv')}
        require(set(assignment)==set(train.subject_id) and all(sum(v==f for v in assignment.values())==18 for f in range(3)),'Invalid outer fold assignment')
        rosters=read(held/'label_subsets.csv');fold_frames={}
        for entry in read(held/'fold_run_registry.csv'):
            run=entry['run_id'];fold=int(entry['fold']);relative=f'reports/{HELD}/predictions/{run}.csv'
            require(hashlib.sha256(content(relative)).hexdigest()==entry['predictions_sha256'],'Outer prediction SHA mismatch')
            value=frame(relative)
            expected=train[train.subject_id.map(assignment)==fold]
            require(identity(value)==identity(expected),'Outer heldout sample identity/order mismatch')
            measured=metric_check(value,json.loads((held/'predictions'/(run+'_metrics.json')).read_text()))
            require(float(entry['macro_f1'])==measured['macro_f1'] and float(entry['n1_f1'])==measured['n1_f1'],'Outer registry metric mismatch')
            cfg=json.loads((held/'final_models'/run/'config.json').read_text())
            selected={r['subject_id'] for r in rosters if int(r['fold'])==fold and int(r['roster_seed'])==int(entry['roster_seed']) and r[entry['budget']]=='1'}
            require(len(selected)==int(entry['labeled_subjects']) and set(cfg['training_subjects'])==selected
                    and set(cfg['normalization_fitted_subjects'])==selected and not selected&set(value.subject_id),'Outer training/normalization leakage')
            require(not cfg['outer_labels_used'] and set(cfg['outer_heldout_subjects'])==set(value.subject_id),'Outer policy drift')
            fold_frames[(entry['roster_seed'],entry['budget'],entry['source_key'],fold)]=value;official_checks+=1
        require(len(fold_frames)==72,'Missing outer fold predictions')
        merged={}
        for roster in ['2024','4101','4102']:
            for budget in ['n4','n9']:
                for source in ['S0','S1_aseed0','S1_aseed1','S1_aseed2']:
                    relative=f'reports/{HELD}/OOF54/{source}_{budget}_r{roster}.csv'
                    value=frame(relative)
                    reconstructed=pd.concat([fold_frames[(roster,budget,source,f)] for f in range(3)],ignore_index=True).sort_values('sample_id').reset_index(drop=True)
                    require(identity(value)==identity(reconstructed) and np.array_equal(value.y_pred,reconstructed.y_pred),'OOF merge identity/prediction mismatch')
                    np.testing.assert_allclose(value[official.PROB_COLS],reconstructed[official.PROB_COLS],rtol=1e-14,atol=1e-15)
                    require(len(value)==132938 and value.subject_id.nunique()==54,'OOF54 incomplete')
                    metric_check(value,json.loads((held/'OOF54'/f'{source}_{budget}_r{roster}.metrics.json').read_text()))
                    merged[(roster,budget,source)]=value;official_checks+=1
        del fold_frames
        paired=0
        for entry in read(multi/'paired_adaptation_deltas.csv'):
            actual=official.paired_bootstrap_delta(single[entry['S1_run_id']],single[entry['S0_run_id']],n_boot=1000,seed=0)
            saved=json.loads((multi/'paired_bootstrap'/(entry['S1_run_id']+'.json')).read_text())
            require(actual==saved,'Multi paired bootstrap mismatch');paired+=1
        for entry in read(held/'OOF54_paired_deltas.csv'):
            key=(entry['roster_seed'],entry['budget']);a=merged[key+(f"S1_aseed{entry['adaptation_seed']}",)];b=merged[key+('S0',)]
            actual=official.paired_bootstrap_delta(a,b,n_boot=1000,seed=0)
            saved=json.loads((held/'OOF54'/f"delta_seed{entry['adaptation_seed']}_{entry['budget']}_r{entry['roster_seed']}.json").read_text())
            require(actual==saved,'OOF paired bootstrap mismatch')
            require(abs(100*actual['macro_f1']['point']-float(entry['delta_macro_f1_pp']))<1e-12,'Paired table mismatch');paired+=1
        require(paired==36,'Wrong pair count')
        adaptations=0
        for fold in range(3):
            allowed={s for s,f in assignment.items() if f!=fold}
            for seed in range(3):
                folder=held/'adaptations'/f'fold{fold}_seed{seed}'
                status=json.loads((folder/'summary.json').read_text());initial=json.loads((folder/'initialization.json').read_text())
                require(status['status']=='complete' and status['completed_updates']==status['ema_updates']==10000,'Incomplete EMA')
                require(initial['optimizer_updates']==initial['ema_updates']==0 and set(initial['adaptation_subjects'])==allowed,'Initialization/pool mismatch')
                batches=read(folder/'signal_batches.csv');access=read(folder/'signal_access_audit.csv')
                require(len(batches)==20000 and {x['subject_id'] for x in batches}==allowed
                        and all(x['subject_id'] in allowed and int(x['y_hidden'])==-1 for x in batches),'Adaptation person/label leakage')
                require(len(access)==20000 and all(x['record_id'][:5] in allowed and x['member'] in ['signal.npy','epoch_index.npy'] for x in access),'Forbidden adaptation access')
                relative=f'reports/{HELD}/adaptations/fold{fold}_seed{seed}/selected_ema_step10000.pt'
                require(hashlib.sha256(content(relative)).hexdigest()==status['selected_weight']['sha256'],'Selected EMA checksum mismatch')
                if inspect_backbones:
                    import torch
                    torch.set_num_threads(4)
                    ckpt=torch.load(io.BytesIO(content(relative)),map_location='cpu',weights_only=True)
                    require(ckpt['completed_updates']==ckpt['ema_updates']==10000 and ckpt['fold']==fold and ckpt['seed']==seed
                        and set(ckpt['adaptation_subjects'])==allowed and set(ckpt['outer_heldout_subjects'])==set(assignment)-allowed
                        and ckpt['weight_role']=='selected_new_ema','Selected EMA metadata mismatch')
                    require(all(bool(torch.isfinite(v).all()) for v in ckpt['state_dict'].values()),'Nonfinite EMA state')
                    del ckpt
                adaptations+=1
        npz=0
        for relative in members:
            if not relative.endswith('/model.npz'):continue
            model_folder=ROOT/Path(relative).parent
            receipt=json.loads((model_folder/'artifact_sha256.json').read_text())
            require(hashlib.sha256(content(relative)).hexdigest()==receipt['model.npz'],'Classifier model checksum mismatch')
            with np.load(io.BytesIO(content(relative)),allow_pickle=False) as saved:
                require(saved['theta'].shape==(12805,) and saved['theta'].dtype==np.float64 and saved['mean'].shape==(2560,)
                    and saved['std'].shape==(2560,) and (saved['std']>=1e-6).all() and np.array_equal(saved['classes'],np.arange(5)),'Classifier schema mismatch')
                require(all(np.isfinite(saved[k]).all() for k in saved.files),'Nonfinite classifier')
            npz+=1
        require(npz==314,'Wrong classifier weight count')
        return dict(status='passed',main_source_files_verified=len(manifest['original_main_files']),
            release_members_sha_verified=len(members),final_prediction_tables=96,merged_OOF_tables=24,
            official_point_metric_recomputations=official_checks,paired_bootstrap_recomputations=paired,
            converged_classifier_fits=312,failed_v01_models_archived_separately=2,classifier_NPZ_verified=npz,
            fold_adaptations_verified=adaptations,backbone_metadata_inspected=inspect_backbones,
            outer_people_excluded_from_adaptation=True,original_dev8_excluded_from_holdout=True,test_used=False,
            raw_signals_read=False,training_started=False,large_feature_arrays_verified=False)
    finally:
        for archive in zips.values():archive.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True,help='Directory with the two downloaded Release ZIP files')
    p.add_argument('--inspect-backbones',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args();result=verify(args.assets,args.inspect_backbones)
    if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
