"""EEG-only, three nested label rosters and four existing frozen representations.

Train-only grouped S0 CV selects one shared L2 value per roster/budget. Float64
multinomial logistic regression is optimized to a gradient tolerance. The eight
development validation subjects are evaluated only after every fit converges.
"""
import argparse
import csv
from datetime import datetime,timedelta,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.optimize import minimize
import scipy
from threadpoolctl import threadpool_limits,threadpool_info

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/EEG_multisubset_20261031_v02'
REFERENCE=ROOT/'manifests/week2_c_adaptation_v01/label_subsets_v2.csv'
SPLIT=ROOT/'vendor/experiment1-review/manifests/split_subjects.csv'
EVALUATOR=ROOT/'vendor/experiment2-review-6e7df99/src/evaluation/evaluate.py'
EVALUATOR_SHA='4168c650b154f2c2a2c4ff26e8acd8466a41d3f6823dbaeabfa6eb07ab7f7605'
SEEDS=[2024,4101,4102]
BUDGETS={'label10':5,'label25':14}
LAMBDAS=[.01,.001,.0001]  # Strict score improvement; ties retain larger lambda.
GTOL=1e-5
SOURCES=[
 dict(key='S0',adaptation_seed=None,feature_root='features/eeg_cond14_v01',
      pool='reports/week2_eeg_comparison_20261028/execution/S0_pooled.npy',
      audit='reports/week2_eeg_comparison_20261028/execution/S0_feature_audit.json'),
 dict(key='S1_aseed0',adaptation_seed=0,feature_root='features/eeg_cond14_S1_aseed0_steps10000_v01',
      pool='reports/week2_eeg_comparison_20261028/execution/S1_pooled.npy',
      audit='reports/week2_eeg_comparison_20261028/execution/S1_feature_audit.json'),
 dict(key='S1_aseed1',adaptation_seed=1,feature_root='features/eeg_cond14_S1_aseed1_steps10000_v02',
      pool='reports/week2_seed1_features_heads_20261028_v04/execution/S1_pooled.npy',
      audit='reports/week2_seed1_features_heads_20261028_v04/execution/S1_feature_audit.json'),
 dict(key='S1_aseed2',adaptation_seed=2,feature_root='features/eeg_cond14_S1_aseed2_steps10000_v01',
      pool='reports/week2_seed2_features_heads_20261029/execution/S1_pooled.npy',
      audit='reports/week2_seed2_features_heads_20261029/execution/S1_feature_audit.json')]


def require(condition,message):
    if not condition:raise ValueError(message)


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def official_module():
    require(sha(EVALUATOR)==EVALUATOR_SHA,'Evaluator changed')
    spec=importlib.util.spec_from_file_location('B_multisubset_pinned',EVALUATOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def make_orders(train_subjects,reference):
    subjects=sorted(train_subjects)
    require(len(subjects)==len(set(subjects))==54,'Expected exactly 54 train subjects')
    ranked=sorted(reference,key=lambda r:int(r['sampling_rank']))
    require([int(r['sampling_rank']) for r in ranked]==list(range(1,55)),'Reference ranks differ')
    require({r['subject_id'] for r in ranked}==set(subjects),'Reference roster differs from train split')
    orders={2024:[r['subject_id'] for r in ranked]}
    for seed in [4101,4102]:orders[seed]=np.random.Generator(np.random.PCG64(seed)).permutation(subjects).tolist()
    return orders


def inner_folds(subjects):
    require(len(subjects)==len(set(subjects)) and len(subjects)>=3,'Duplicate/insufficient subjects')
    return {subject:rank%3 for rank,subject in enumerate(subjects)}


def labeled_positions(index,subjects):
    allowed=set(subjects)
    ids=np.array([i for i,r in enumerate(index) if r['split']=='train' and r['subject_id'] in allowed],dtype=np.int64)
    require({index[i]['subject_id'] for i in ids}==allowed,'Missing labeled subject')
    return ids


def normalize_training(values):
    x=np.array(values,dtype=np.float64,order='C',copy=True)
    require(x.ndim==2 and np.isfinite(x).all(),'Invalid training features')
    mean=x.mean(axis=0,dtype=np.float64)
    std=np.maximum(x.std(axis=0,dtype=np.float64),1e-6)
    x-=mean;x/=std
    return x,mean,std


def apply_normalization(values,mean,std):
    x=np.array(values,dtype=np.float64,order='C',copy=True);x-=mean;x/=std
    require(np.isfinite(x).all(),'Invalid normalized evaluation features');return x


def loss_gradient(theta,x,y,regularization):
    d=x.shape[1];w=theta[:5*d].reshape(5,d);bias=theta[5*d:]
    scores=x@w.T+bias;scores-=scores.max(axis=1,keepdims=True)
    probability=np.exp(scores);denominator=probability.sum(axis=1,keepdims=True)
    nll=np.mean(np.log(denominator[:,0])-scores[np.arange(len(y)),y])
    loss=float(nll+.5*regularization*np.sum(w*w))
    probability/=denominator;probability[np.arange(len(y)),y]-=1.;probability/=len(y)
    gw=probability.T@x+regularization*w;gb=probability.sum(axis=0)
    return loss,np.concatenate([gw.reshape(-1),gb])


def predict_probabilities(theta,x):
    d=x.shape[1];scores=x@theta[:5*d].reshape(5,d).T+theta[5*d:]
    scores-=scores.max(axis=1,keepdims=True);p=np.exp(scores);p/=p.sum(axis=1,keepdims=True)
    require(np.isfinite(p).all(),'Nonfinite probabilities');return p


def solve(x,y,regularization,progress=None,maxiter=10000,gtol=GTOL):
    require(x.dtype==np.float64 and x.ndim==2 and np.isfinite(x).all(),'Optimizer requires finite float64 features')
    require(y.dtype==np.int64 and y.shape==(len(x),) and np.isin(y,np.arange(5)).all()
            and (np.bincount(y,minlength=5)>0).all(),'Missing/invalid training class')
    require(regularization>0,'Regularization must be positive')
    theta=np.zeros(5*x.shape[1]+5,dtype=np.float64);trace=[];stages=[];calls=0;cached={}
    began=time.perf_counter()
    def fun(parameters):
        nonlocal calls
        value,gradient=loss_gradient(parameters,x,y,regularization);calls+=1
        require(np.isfinite(value) and np.isfinite(gradient).all(),'Nonfinite optimizer state')
        cached.update(theta=parameters.copy(),loss=value,gradient_inf=float(np.max(np.abs(gradient))))
        return value,gradient
    def callback(parameters):
        if not np.array_equal(parameters,cached.get('theta')):fun(parameters)
        row=dict(iteration=len(trace)+1,objective=cached['loss'],gradient_inf=cached['gradient_inf'],
                 seconds=time.perf_counter()-began)
        trace.append(row)
        if progress is not None:progress(row)
    # A deterministic second pass may refine the SAME objective; there is no
    # validation-based budget change or hyperparameter search in this solver.
    for stage,ftol in enumerate([1e-15,0.]):
        result=minimize(fun,theta,method='L-BFGS-B',jac=True,callback=callback,
            options=dict(maxiter=maxiter,maxfun=25000,maxls=50,maxcor=20,gtol=gtol,ftol=ftol))
        theta=result.x;value,gradient=loss_gradient(theta,x,y,regularization)
        gradient_inf=float(np.max(np.abs(gradient)))
        stages.append(dict(pass_number=stage+1,scipy_success=bool(result.success),message=str(result.message),
                           iterations=int(result.nit),objective=float(value),gradient_inf=gradient_inf))
        if np.isfinite(value) and np.isfinite(gradient_inf) and gradient_inf<=gtol:break
    converged=bool(np.isfinite(theta).all() and gradient_inf<=gtol)
    info=dict(converged=converged,gradient_inf=gradient_inf,gradient_tolerance=gtol,objective=float(value),
              iterations=sum(s['iterations'] for s in stages),function_calls=calls,seconds=time.perf_counter()-began,
              refinement_passes=stages,normalization_dtype='float64',optimizer='SciPy L-BFGS-B',zero_initialization=True)
    require(converged,'Optimizer failed declared gradient tolerance: '+json.dumps(info))
    return theta,info,trace


def choose_lambda(scores):
    require(set(scores)==set(LAMBDAS),'Incomplete regularization grid')
    require(all(np.isfinite(v) for v in scores.values()),'Nonfinite CV score')
    return max(LAMBDAS,key=lambda value:(scores[value],value))


def run_plan():
    return [dict(run_id=f'CX_EEG_LR_{source["key"]}_{budget}_roster{seed}_v02',source_key=source['key'],
                 adaptation_seed=source['adaptation_seed'],representation='S0' if source['key']=='S0' else 'S1',
                 roster_seed=seed,budget=budget,labeled_subjects=size)
            for seed in SEEDS for budget,size in BUDGETS.items() for source in SOURCES]


def prepare():
    require(not REPORT.exists(),'Create a new version; report already exists')
    index=read_csv(ROOT/'features/eeg_cond14_v01/feature_index.csv')
    train=sorted({r['subject_id'] for r in index if r['split']=='train'})
    orders=make_orders(train,read_csv(REFERENCE));REPORT.mkdir()
    roster_rows=[];profile=[];fold_rows=[]
    for seed,order in orders.items():
        for rank,subject in enumerate(order,1):
            roster_rows.append(dict(roster_seed=seed,subject_id=subject,sampling_rank=rank,label10=int(rank<=5),label25=int(rank<=14)))
        for budget,size in BUDGETS.items():
            selected=order[:size];ids=labeled_positions(index,selected);mapping=inner_folds(selected)
            y=np.array([int(index[i]['y_true']) for i in ids],dtype=np.int64);counts=np.bincount(y,minlength=5)
            require((counts>0).all(),'A preselected roster lacks a class; do not redraw silently')
            for subject in selected:fold_rows.append(dict(roster_seed=seed,budget=budget,subject_id=subject,validation_fold=mapping[subject]))
            for f in range(3):
                inner=[int(index[i]['y_true']) for i in ids if mapping[index[i]['subject_id']]!=f]
                require((np.bincount(inner,minlength=5)>0).all(),'An inner training fold lacks a class')
            profile.append(dict(roster_seed=seed,budget=budget,subjects=size,records=len({index[i]['record_id'] for i in ids}),
                samples=len(ids),**dict(zip(['W','N1','N2','N3','REM'],counts.tolist()))))
    config=dict(solver_budget_revision='v01 stopped at lambda0.0001 after2000 total iterations, gradient_inf2.8542793781732546e-5; no dev validation; v02 expands solver cap only',experiment='EEG_only_multisubset_converged_logistic_v02',registered_at=now(),
        roster_seeds=SEEDS,new_roster_rng='NumPy PCG64 permutation of sorted 54 train IDs; seed2024 rank retained exactly',
        budgets=BUDGETS,sources=SOURCES,planned_final_models=24,planned_CV_fits=54,
        adaptation_seeds=[0,1,2],new_backbone_training=False,new_feature_extraction=False,EOG_features_read=False,
        features='all20 gate, existing float32 mean30 time pooling to2560',
        classifier='five-class float64 multinomial logistic regression',objective='mean cross_entropy + lambda/2 * sum(W**2); bias unregularized',
        lambda_grid=LAMBDAS,lambda_selection='S0-only 3-fold subject-grouped OOF pooled Macro-F1 within current labeled subjects',
        lambda_shared_by_all_four_representations=True,tie_rule='larger lambda',
        inner_fold_rule='rank within selected roster modulo3',normalization='fit only each inner training fold; final fit only selected labeled train subjects',
        normalization_dtype='float64',std_floor=1e-6,gradient_inf_tolerance=GTOL,max_iterations_per_pass=10000,
        predetermined_solver_passes=[dict(ftol=1e-15,maxfun=25000),dict(ftol=0.,maxfun=25000)],
        only_refine_same_objective_if_needed=True,zero_initialization=True,CPU_threads=4,
        development_validation_embargo='all54 CV fits and24 final fits converged before any final validation predictions',
        development_validation_subjects=8,test_used=False,
        adaptation_unlabeled_subjects=54,adapted_backbones_have_seen_all54_training_subject_signals=True,
        interpretation='Repeated label rosters on same8 development subjects; not new unseen-subject validation or unique diffusion-objective evidence')
    write_csv(REPORT/'label_subsets.csv',roster_rows);write_csv(REPORT/'budget_profiles.csv',profile)
    write_csv(REPORT/'inner_fold_subjects.csv',fold_rows);write_csv(REPORT/'run_plan.csv',run_plan());write_json(REPORT/'config.json',config)
    (REPORT/'实验方案.md').write_text('''# EEG-only＋三份标注名单：收敛线性读出

v02仅扩大求解器上限：v01最弱正则在2000步后梯度2.8543e-5未达1e-5，故按规则停止，开发验证评价次数为0。v01原始脚本和失败记录保留；v02全量重启，不复用已拟合权重。目标、名单、折、正则网格与收敛阈值不变。每遍maxfun25000。

探索性新方案，固定原54/8/16按人划分，不启封测试。表征为已有S0及三个S1，全部20层2560维EEG池化特征，不读取EOG或其他成员新结果，不训练主干、不重提特征。

三份排序固定为2024（原名单逐位复用）、4101、4102（PCG64随机排列排序后的54人）。每份前5/14人构成嵌套预算；不按结果挑名单。实际片段/类别数见budget_profiles.csv。

每个名单/预算只在S0特征上做3折按人内层验证。验证折为选中排序rank从0起模3，训练折与验证者互斥，每人恰好被留出一次。每折均仅用内层训练者拟合float64均值/标准差。正则lambda预设0.01/0.001/0.0001，以所有OOF预测合并的Macro-F1选定，同分取较强正则；选择出的一个lambda由该名单/预算的全部四个表征共用。这是S0所选读出器下的保守比较，不声称S1在所有独立调参策略下无效。

优化mean交叉熵+lambda/2*||W||²，bias不正则，float64、零初始化、SciPy L-BFGS-B。以最终梯度无穷范数≤1e-5作为收敛判据，不能仅因迭代到上限或SciPy返回success便当作收敛。预先允许同目标最多两次各10000迭代的确定性精修，第一遍ftol1e-15，必要时第二遍ftol0；未满足梯度标准则失败留存，不改参数或重新抽人。

总计54次内层拟合（3名单×2预算×3折×3lambda）和24个最终模型（3名单×2预算×4表征）。全部拟合收敛后才统一评价原8名开发验证者；不按其分数改lambda、模型、名单或停止规则。没有下游分类随机种子重复，变异来源主要为标注名单与已有的3份适配主干。

保存所有内层/最终模型、标准化、训练者清单、收敛轨迹与SHA；最后输出24份完整预测、18组S1−S0配对、按人bootstrap、名单/适配种子分解和审计。相同8人已多次用于研究，这仍是开发集探索，不是新的未见受试者验证；以后按训练者重新分折时必须重新适配，不能直接复用看过全部54人的S1。
''',encoding='utf-8')
    write_json(REPORT/'summary.json',dict(status='prepared_not_started',stage='tests_and_binding_pending',
        completed_cv_fits=0,expected_cv_fits=54,completed_final_models=0,expected_final_models=24,validation_evaluations=0,test_used=False))


def bind():
    require(not (REPORT/'runtime_binding.json').exists(),'Do not replace binding')
    require('\nOK\n' in (REPORT/'runtime_tests.log').read_text(),'Tests did not pass')
    files=[Path(__file__),ROOT/'tests/test_eeg_multisubset_v02.py',REFERENCE,SPLIT,EVALUATOR,
        REPORT/'runtime_tests.log',REPORT/'config.json',REPORT/'label_subsets.csv',REPORT/'budget_profiles.csv',
        REPORT/'inner_fold_subjects.csv',REPORT/'run_plan.csv',REPORT/'实验方案.md']
    for source in SOURCES:
        files += [ROOT/source['audit'],ROOT/source['feature_root']/'feature_config.json',ROOT/source['feature_root']/'feature_index.csv']
    write_json(REPORT/'runtime_binding.json',dict(status='bound_before_new_fitting',bound_at=now(),
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        versions=dict(numpy=np.__version__,scipy=scipy.__version__),test_used=False))


def check_binding():
    for relative,value in json.loads((REPORT/'runtime_binding.json').read_text())['input_sha256'].items():
        require(sha(ROOT/relative)==value,'Bound input changed: '+relative)


def verify_sources():
    index=read_csv(ROOT/'features/eeg_cond14_v01/feature_index.csv')
    require(len(index)==154124 and {r['split'] for r in index}=={'train','val'},'Unexpected source split')
    require(sum(r['split']=='train' for r in index)==132938 and sum(r['split']=='val' for r in index)==21186,'Source coverage differs')
    keys=['sample_id','subject_id','record_id','epoch_index','split','y_true','feature_path','row_index']
    canonical=[tuple(r[k] for k in keys) for r in index];paths={};audits=[]
    for source in SOURCES:
        audit=json.loads((ROOT/source['audit']).read_text());folder=ROOT/source['feature_root'];p=ROOT/source['pool']
        require(audit['status']=='passed' and audit['raw_all_finite'] and audit['samples']==154124,'Source audit failed')
        require(sha(p)==audit['pooled_sha256'],'Pool SHA mismatch')
        require(sha(folder/'feature_index.csv')==audit['source_index_sha256'] and sha(folder/'feature_config.json')==audit['source_config_sha256'],'Source metadata changed')
        require([tuple(r[k] for k in keys) for r in read_csv(folder/'feature_index.csv')]==canonical,'Representation row order differs')
        x=np.load(p,mmap_mode='r',allow_pickle=False);require(x.shape==(154124,2560) and x.dtype==np.float32,'Wrong EEG pool');del x
        paths[source['key']]=p;audits.append(dict(source_key=source['key'],pool_sha256=audit['pooled_sha256'],finite_source_receipt_verified=True))
        print(json.dumps(dict(stage='verify_existing_EEG_pool',source=source['key'])),flush=True)
    write_json(REPORT/'pooled_reuse_audit.json',dict(status='passed',EEG_only=True,EOG_read=False,sources=audits))
    return index,paths


def save_fit(folder,x,y,mean,std,regularization,metadata,status):
    folder.mkdir(parents=True,exist_ok=False);write_json(folder/'config.json',metadata)
    last_write=0.
    def progress(row):
        nonlocal last_write
        if time.perf_counter()-last_write>=5:
            write_json(REPORT/'optimizer_progress.json',dict(fit=metadata['fit_id'],**row));last_write=time.perf_counter()
    theta,info,trace=solve(x,y,regularization,progress=progress)
    with (folder/'model.npz').open('xb') as f:
        np.savez_compressed(f,theta=theta,mean=mean,std=std,regularization=np.array(regularization),classes=np.arange(5))
    if trace:write_csv(folder/'optimizer_history.csv',trace)
    write_json(folder/'optimizer_summary.json',info)
    write_json(folder/'artifact_sha256.json',{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    status['completed_optimizer_fits']=status.get('completed_optimizer_fits',0)+1
    print(json.dumps(dict(stage='fit_converged',fit=metadata['fit_id'],iterations=info['iterations'],gradient_inf=info['gradient_inf'],seconds=info['seconds'])),flush=True)
    return theta,info


def regularization_search(x,y,subjects,regularizations,fit_callback,metric_function):
    require(set(regularizations)==set(LAMBDAS),'Regularization grid drift')
    selected=list(dict.fromkeys(subjects.tolist()));mapping=inner_folds(selected)
    fold=np.array([mapping[s] for s in subjects]);oof={value:np.full(len(y),-1,dtype=np.int8) for value in regularizations}
    fold_records=[]
    for f in range(3):
        train=np.flatnonzero(fold!=f);val=np.flatnonzero(fold==f)
        require(set(subjects[train]).isdisjoint(set(subjects[val])),'Inner person leakage')
        tx,mean,std=normalize_training(x[train]);vx=apply_normalization(x[val],mean,std)
        for value in regularizations:
            theta,info=fit_callback(tx,y[train],mean,std,value,f,sorted(set(subjects[train])),sorted(set(subjects[val])))
            oof[value][val]=predict_probabilities(theta,vx).argmax(axis=1)
            fold_records.append(dict(fold=f,regularization=value,train_subjects=';'.join(sorted(set(subjects[train]))),
                heldout_train_subjects=';'.join(sorted(set(subjects[val]))),gradient_inf=info['gradient_inf'],
                fold_macro_f1=metric_function(y[val],oof[value][val])['macro_f1']))
        del tx,vx
    require(all((v>=0).all() for v in oof.values()),'Missing OOF predictions')
    scores={value:metric_function(y,pred)['macro_f1'] for value,pred in oof.items()}
    return choose_lambda(scores),scores,fold_records,oof


def run():
    check_binding();require(not (REPORT/'cv_fits').exists(),'Existing run cannot be overwritten')
    with (REPORT/'.multisubset.lock').open('a') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(status='running',stage='verify_EEG_sources',pid=os.getpid(),started_at=now(),completed_cv_fits=0,
            expected_cv_fits=54,completed_final_models=0,expected_final_models=24,validation_evaluations=0,test_used=False)
        write_json(REPORT/'summary.json',status)
        try:
            index,paths=verify_sources();official=official_module()
            reference=read_csv(REFERENCE);orders=make_orders(sorted({r['subject_id'] for r in index if r['split']=='train'}),reference)
            folds_registered=read_csv(REPORT/'inner_fold_subjects.csv');lambdas={};cv_records=[];fit_rows=[]
            (REPORT/'cv_fits').mkdir();(REPORT/'final_models').mkdir()
            write_json(REPORT/'threadpools.json',threadpool_info())
            status['stage']='train_only_S0_grouped_regularization_selection';write_json(REPORT/'summary.json',status)
            s0=np.load(paths['S0'],mmap_mode='r',allow_pickle=False)
            for seed in SEEDS:
                for budget,size in BUDGETS.items():
                    selected=orders[seed][:size];ids=labeled_positions(index,selected)
                    x=np.asarray(s0[ids]);y=np.array([int(index[i]['y_true']) for i in ids],dtype=np.int64)
                    # Preserve the predeclared roster rank when making grouped folds.
                    rank={s:i for i,s in enumerate(selected)}
                    order=np.argsort([rank[index[i]['subject_id']] for i in ids],kind='stable')
                    x=x[order];y=y[order];subject=np.array([index[i]['subject_id'] for i in ids])[order]
                    registered={r['subject_id']:int(r['validation_fold']) for r in folds_registered if int(r['roster_seed'])==seed and r['budget']==budget}
                    require(inner_folds(selected)==registered,'Registered inner folds differ')
                    def fit_cv(tx,ty,mean,std,value,f,training_subjects,heldout_subjects):
                        key=f'roster{seed}_{budget}_fold{f}_lambda{value:g}'
                        status.update(current_fit=key);write_json(REPORT/'summary.json',status)
                        cfg=dict(fit_id=key,role='training_only_S0_inner_fit',roster_seed=seed,budget=budget,
                            source_key='S0',regularization=value,normalization_fitted_subjects=training_subjects,
                            inner_heldout_subjects=heldout_subjects,training_samples=len(tx),test_used=False,
                            development_validation_used=False,binding_sha256=sha(REPORT/'runtime_binding.json'))
                        theta,info=save_fit(REPORT/'cv_fits'/key,tx,ty,mean,std,value,cfg,status)
                        status['completed_cv_fits']+=1;write_json(REPORT/'summary.json',status)
                        fit_rows.append(dict(fit_id=key,role='inner_CV',roster_seed=seed,budget=budget,source_key='S0',
                                             regularization=value,iterations=info['iterations'],gradient_inf=info['gradient_inf'],seconds=info['seconds']))
                        write_csv(REPORT/'optimizer_fit_registry.csv',fit_rows)
                        return theta,info
                    chosen,scores,records,oof=regularization_search(x,y,subject,LAMBDAS,fit_cv,official.compute_metrics)
                    lambdas[(seed,budget)]=chosen
                    cv_records += [dict(roster_seed=seed,budget=budget,**r) for r in records]
                    write_csv(REPORT/'cv_fold_metrics.csv',cv_records)
                    write_json(REPORT/f'cv_roster{seed}_{budget}.json',dict(selected_lambda=chosen,OOF_macro_f1={str(k):v for k,v in scores.items()},
                        selected_only_from_labeled_training_subjects=True,normalization_fitted_per_inner_training_fold=True))
                    oof_rows=[]
                    for position,original_id in enumerate(ids[order]):
                        original=index[original_id]
                        item={k:original[k] for k in ['sample_id','subject_id','record_id','epoch_index','y_true']}
                        item['heldout_fold']=registered[original['subject_id']]
                        item.update({f'pred_lambda_{value:g}':int(oof[value][position]) for value in LAMBDAS})
                        oof_rows.append(item)
                    write_csv(REPORT/f'cv_roster{seed}_{budget}_OOF_predictions.csv',oof_rows)
                    del x,y,subject
            del s0
            require(status['completed_cv_fits']==54,'Missing CV fits')
            write_csv(REPORT/'selected_regularization.csv',[dict(roster_seed=seed,budget=budget,regularization=value,selection_source='S0 train-only OOF Macro-F1')
                for (seed,budget),value in lambdas.items()])
            status['stage']='fit24_final_EEG_only_models_validation_embargoed';write_json(REPORT/'summary.json',status)
            final=[]
            for entry in run_plan():
                selected=orders[entry['roster_seed']][:entry['labeled_subjects']];ids=labeled_positions(index,selected)
                pool=np.load(paths[entry['source_key']],mmap_mode='r',allow_pickle=False)
                x,mean,std=normalize_training(pool[ids]);y=np.array([int(index[i]['y_true']) for i in ids],dtype=np.int64)
                value=lambdas[(entry['roster_seed'],entry['budget'])]
                status['current_fit']=entry['run_id'];write_json(REPORT/'summary.json',status)
                cfg=dict(entry,fit_id=entry['run_id'],role='final_fit_not_yet_evaluated',regularization=value,
                    input_dim=2560,classifier='LinearLogistic(2560,5)',normalization_dtype='float64',
                    normalization_fitted_subjects=sorted(selected),training_samples=len(x),test_used=False,
                    development_validation_used=False,unlabeled_adaptation_subjects=0 if entry['source_key']=='S0' else 54,
                    binding_sha256=sha(REPORT/'runtime_binding.json'))
                theta,info=save_fit(REPORT/'final_models'/entry['run_id'],x,y,mean,std,value,cfg,status)
                final.append(dict(entry,regularization=value,gradient_inf=info['gradient_inf'],iterations=info['iterations'],seconds=info['seconds']))
                status['completed_final_models']+=1;write_json(REPORT/'summary.json',status)
                fit_rows.append(dict(fit_id=entry['run_id'],role='final',roster_seed=entry['roster_seed'],budget=entry['budget'],source_key=entry['source_key'],
                    regularization=value,iterations=info['iterations'],gradient_inf=info['gradient_inf'],seconds=info['seconds']))
                write_csv(REPORT/'optimizer_fit_registry.csv',fit_rows);del x,y,pool,theta
            require(status['completed_final_models']==24,'Missing final models')
            check_binding();write_json(REPORT/'validation_embargo_release.json',dict(released_at=now(),completed_cv_fits=54,completed_final_models=24,
                every_fit_converged=True,validation_evaluations_before_release=0,test_used=False))
            status.update(stage='final_development_validation_once_per_model',current_fit=None);write_json(REPORT/'summary.json',status)
            results=evaluate_final(final,index,paths,official,status)
            status['stage']='paired_roster_and_adaptation_analysis';write_json(REPORT/'summary.json',status)
            analyze(results,index,official);check_binding()
            status.update(status='complete',stage='EEG_multisubset_analysis_complete',completed_at=now())
        except BaseException:
            status.update(status='failed',failed_at=now(),error=traceback.format_exc());raise
        finally:write_json(REPORT/'summary.json',status)


def evaluate_final(final,index,paths,official,status):
    (REPORT/'predictions').mkdir();(REPORT/'evaluation').mkdir()
    val_ids=np.array([i for i,r in enumerate(index) if r['split']=='val']);val=[index[i] for i in val_ids]
    require(len(val)==21186 and len({r['subject_id'] for r in val})==8,'Wrong development set')
    results=[]
    for entry in final:
        folder=REPORT/'final_models'/entry['run_id']
        hashes=json.loads((folder/'artifact_sha256.json').read_text());require(all(sha(folder/name)==value for name,value in hashes.items()),'Final fit archive changed')
        with np.load(folder/'model.npz',allow_pickle=False) as saved:
            theta=saved['theta'];mean=saved['mean'];std=saved['std'];require(float(saved['regularization'])==entry['regularization'],'Wrong lambda')
        pool=np.load(paths[entry['source_key']],mmap_mode='r',allow_pickle=False)
        x=apply_normalization(pool[val_ids],mean,std);probs=predict_probabilities(theta,x)
        output=REPORT/'predictions'/(entry['run_id']+'.csv');rows=[]
        for row,p in zip(val,probs):
            item={k:row[k] for k in ['sample_id','subject_id','record_id','epoch_index','y_true']}
            item['y_pred']=int(p.argmax());item.update(zip(official.PROB_COLS,map(float,p)));rows.append(item)
        write_csv(output,rows);frame=official.load_predictions(output);metrics=official.evaluate(output,n_boot=1000,seed=0)
        write_json(REPORT/'evaluation'/(entry['run_id']+'.json'),metrics)
        official.per_subject_table(frame).to_csv(REPORT/'evaluation'/(entry['run_id']+'_subjects.csv'),index=False)
        result=dict(entry,macro_f1=metrics['macro_f1'],n1_precision=metrics['n1_precision'],n1_recall=metrics['n1_recall'],n1_f1=metrics['n1_f1'],
            rem_f1=metrics['per_class_f1']['REM'],accuracy=metrics['accuracy'],predictions_sha256=sha(output),test_used=False)
        results.append(result);write_csv(REPORT/'run_registry.csv',results)
        status['validation_evaluations']+=1;write_json(REPORT/'summary.json',status)
        print(json.dumps(dict(stage='evaluated_final_model',run_id=entry['run_id'],macro_f1=metrics['macro_f1'],n1_f1=metrics['n1_f1'])),flush=True)
        del x,pool
    return results


def analyze(results,index,official):
    require(len(results)==24,'Incomplete final matrix');pairs=[];person=[];summary=[]
    directory=REPORT/'paired_bootstrap';directory.mkdir()
    for seed in SEEDS:
        for budget in BUDGETS:
            cell=[r for r in results if r['roster_seed']==seed and r['budget']==budget]
            base=next(r for r in cell if r['source_key']=='S0')
            for adapted in [r for r in cell if r['source_key']!='S0']:
                a=official.load_predictions(REPORT/'predictions'/(adapted['run_id']+'.csv'))
                b=official.load_predictions(REPORT/'predictions'/(base['run_id']+'.csv'))
                delta=official.paired_bootstrap_delta(a,b,n_boot=1000,seed=0)
                write_json(directory/(adapted['run_id']+'.json'),delta)
                row=dict(roster_seed=seed,budget=budget,adaptation_seed=adapted['adaptation_seed'],S0_run_id=base['run_id'],S1_run_id=adapted['run_id'])
                for key in ['macro_f1','n1_precision','n1_recall','n1_f1','rem_f1','accuracy']:row['delta_'+key+'_pp']=100*(adapted[key]-base[key])
                pairs.append(row)
                for subject in sorted(a.subject_id.unique()):
                    af,bf=a[a.subject_id==subject],b[b.subject_id==subject]
                    am=official.compute_metrics(af.y_true.to_numpy(),af.y_pred.to_numpy());bm=official.compute_metrics(bf.y_true.to_numpy(),bf.y_pred.to_numpy())
                    person.append(dict(roster_seed=seed,budget=budget,adaptation_seed=adapted['adaptation_seed'],subject_id=subject,
                        delta_macro_f1_pp=100*(am['macro_f1']-bm['macro_f1']),delta_n1_f1_pp=100*(am['n1_f1']-bm['n1_f1'])))
            for metric in ['macro_f1','n1_precision','n1_recall','n1_f1','rem_f1','accuracy']:
                adapted_mean=float(np.mean([r[metric] for r in cell if r['source_key']!='S0']))
                summary.append(dict(roster_seed=seed,budget=budget,metric=metric,S0=base[metric],S1_mean_three_adaptations=adapted_mean,delta_pp=100*(adapted_mean-base[metric])))
    write_csv(REPORT/'paired_adaptation_deltas.csv',pairs);write_csv(REPORT/'per_subject_paired.csv',person);write_csv(REPORT/'roster_summary.csv',summary)
    for r in results:
        frame=official.load_predictions(REPORT/'predictions'/(r['run_id']+'.csv'))
        require(len(frame)==21186 and frame.subject_id.nunique()==8 and sha(REPORT/'predictions'/(r['run_id']+'.csv'))==r['predictions_sha256'],'Prediction archive mismatch')
        expected=[(row['sample_id'],row['subject_id'],row['record_id'],int(row['epoch_index']),int(row['y_true'])) for row in index if row['split']=='val']
        require(list(frame[['sample_id','subject_id','record_id','epoch_index','y_true']].itertuples(index=False,name=None))==expected,'Final prediction metadata mismatch')
        measured=official.compute_metrics(frame.y_true.to_numpy(),frame.y_pred.to_numpy())
        stored=json.loads((REPORT/'evaluation'/(r['run_id']+'.json')).read_text())
        require(all(stored[k]==v for k,v in measured.items()),'Final official metric mismatch')
    require(len(pairs)==18 and len(person)==144,'Incomplete pair/person matrix')
    configs=[json.loads((REPORT/'final_models'/r['run_id']/'config.json').read_text()) for r in results]
    require(all(not cfg['development_validation_used'] and not cfg['test_used'] for cfg in configs),'Validation leaked into fitting')
    registry=read_csv(REPORT/'optimizer_fit_registry.csv')
    require(len(registry)==78 and all(float(r['gradient_inf'])<=GTOL for r in registry),'Missing or unconverged fits')
    write_json(REPORT/'completion_audit.json',dict(status='passed',CV_fits=54,final_models=24,paired_contrasts=18,
        per_subject_pairs=144,all_78_fits_meet_gradient_tolerance=True,lambda_selection_training_only=True,
        validation_evaluated_only_after_all_fits=True,EOG_read=False,test_used=False))
    lines=['# EEG-only＋三份名单：完成报告','',f'完成于 {now()}。54次内层拟合和24个最终模型均满足梯度阈值，全部完成后才评价固定8名开发验证者。','',
        '| 名单seed | 预算 | 指标 | S0 | S1三主干均值 | 差值（百分点） |','| --- | --- | --- | ---: | ---: | ---: |']
    for row in summary:
        if row['metric'] in ['macro_f1','n1_f1']:
            lines.append(f"| {row['roster_seed']} | {row['budget']} | {row['metric']} | {100*row['S0']:.3f}% | {100*row['S1_mean_three_adaptations']:.3f}% | {row['delta_pp']:+.3f} |")
    lines += ['', '每份名单内5/14人嵌套；lambda由S0的训练者内部OOF Macro-F1决定，再由4个表征共用。不同名单选择不同lambda是预定训练内选择程序的结果，不使用固定8人验证分数。S1仅有3个独立适配主干，9个名单×适配组合不是9个独立主干或独立验证集。', '',
        '逐名单差值见roster_summary.csv，18组配对及按8人bootstrap见paired_adaptation_deltas.csv和paired_bootstrap/，按人差值见per_subject_paired.csv。本轮仍是同一开发验证集的探索，不能证明未见受试者泛化、扩散目标独有优势或独立测试收益。既有S1曾使用全部54名训练者无标签EEG，新的跨人留出实验需重新按折适配。', '',
        '未读取EOG、未重新训练主干/提特征，测试集始终封存。原公共名单、旧运行与已发布结果不修改。本轮结果尚未同步GitHub。']
    (REPORT/'EEG多名单_完成报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','bind','run']);args=p.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='bind':
        bind();value=json.loads((REPORT/'runtime_binding.json').read_text());print(json.dumps(dict(status=value['status'],bound_inputs=len(value['input_sha256']))))
    else:run()


if __name__=='__main__':main()
