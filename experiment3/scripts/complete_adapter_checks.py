"""Finish training-only pilot adapter checks; never use validation/test for ranking."""
import csv
import argparse
import json
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from extract_features import ROOT, DATA, Extractor, write_csv

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-length',action='store_true',help='Reuse completed length comparison CSV')
    args=parser.parse_args()
    # These matrices have only ~250 rows; many BLAS threads cause overhead.
    thread_limit=threadpool_limits(limits=1)
    out=ROOT/'reports/final_week1'
    out.mkdir(parents=True,exist_ok=True)
    pilot=ROOT/'reports/feature_pilot'
    with (pilot/'pilot_samples.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==296 and {r['split'] for r in rows}=={'train'}
    signals=np.empty((len(rows),3000),np.float32)
    grouped={}
    for i,r in enumerate(rows):
        grouped.setdefault(r['record_id'],[]).append((i,int(r['epoch_index'])))
    for rec,entries in grouped.items():
        with np.load(DATA/f'{rec}.npz',allow_pickle=False) as z:
            lookup={int(epoch):j for j,epoch in enumerate(z['epoch_index'])}
            for i,epoch in entries:
                j=lookup[epoch]
                assert int(z['labels'][j])==int(rows[i]['y_true'])
                signals[i]=z['signal'][j,0]
    torch.manual_seed(0)
    length=[]
    if args.skip_length:
        with (out/'length_comparison.csv').open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                r['condition']=int(r['condition'])
                r['samples']=int(r['samples'])
                for k in ('native_seconds','explicit_seconds','max_abs_difference','max_abs_difference_vs_stored_fp16'):
                    r[k]=float(r[k])
                r['allclose']=r['allclose']=='True'
                r['shape']=[296,20,30,128]
                length.append(r)
        assert len(length)==3 and all(r['allclose'] for r in length)
    for condition in (() if args.skip_length else (14,18,10)):
        model=Extractor(condition)
        model.cache_frozen_kernels()
        model(signals[:1])
        error=0.
        seconds={}
        outputs={}
        for mode in ('native','explicit'):
            torch.cuda.synchronize()
            start=time.perf_counter()
            outputs[mode]=np.concatenate([model(signals[b:b+4],mode=mode) for b in range(0,len(signals),4)])
            torch.cuda.synchronize()
            seconds[mode]=time.perf_counter()-start
        native,explicit=outputs['native'],outputs['explicit']
        assert native.shape==explicit.shape==(296,20,30,128)
        assert np.isfinite(native).all() and np.isfinite(explicit).all()
        assert np.allclose(native,explicit,atol=1e-5,rtol=1e-4)
        existing=np.load(pilot/f'features_cond{condition}.npy',allow_pickle=False).astype(np.float32)
        assert np.allclose(native,existing,atol=0.0002,rtol=0.002)
        result=dict(condition=condition,samples=296,native_seconds=seconds['native'],
            explicit_seconds=seconds['explicit'],max_abs_difference=float(abs(native-explicit).max()),
            max_abs_difference_vs_stored_fp16=float(abs(native-existing).max()),
            allclose=True,shape=[296,20,30,128])
        length.append(result)
        print(json.dumps(result),flush=True)
        del model,outputs,native,explicit,existing
        torch.cuda.empty_cache()
    write_csv(out/'length_comparison.csv',length)
    y=np.array([int(r['y_true']) for r in rows])
    groups=np.array([r['subject_id'] for r in rows])
    folds=[]
    scores=[]
    for condition in (14,18,10):
        # Only stored pilot features; fit normalization/classifier on five subjects.
        x=np.load(pilot/f'features_cond{condition}.npy').astype(np.float32).mean(2).reshape(len(rows),2560)
        pred=np.empty_like(y)
        for train,test in LeaveOneGroupOut().split(x,y,groups):
            scaler=StandardScaler().fit(x[train])
            model=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,random_state=0)
            model.fit(scaler.transform(x[train]),y[train])
            assert max(model.n_iter_)<2000
            pred[test]=model.predict(scaler.transform(x[test]))
            folds.append(dict(condition=condition,held_out_subject=groups[test][0],samples=len(test),
                macro_f1=float(f1_score(y[test],pred[test],labels=range(5),average='macro',zero_division=0))))
            print(json.dumps(folds[-1]),flush=True)
        scores.append(dict(condition=condition,pooled_oof_macro_f1=float(f1_score(y,pred,labels=range(5),average='macro',zero_division=0)),
            mean_subject_macro_f1=float(np.mean([r['macro_f1'] for r in folds if r['condition']==condition]))))
    write_csv(out/'channel_pilot_folds.csv',folds)
    write_csv(out/'channel_pilot_comparison.csv',scores)
    best=max(scores,key=lambda r:r['pooled_oof_macro_f1'])['condition']
    summary=dict(status='passed',test_used=False,validation_used=False,samples=296,training_subjects=6,
        length_comparisons=length,channel_comparison=scores,best_on_this_pilot=best,
        official_condition_unchanged=14,
        probe='Training-only leave-one-subject-out; train-fold StandardScaler + LogisticRegression C=1 lbfgs',
        warning='Pilot is deliberately class-balanced and uses only 6 training subjects. Descriptive post-hoc check; does not establish optimal channel or replace official runs.')
    (out/'adapter_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
