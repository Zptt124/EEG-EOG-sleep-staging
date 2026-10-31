"""Exploratory inverse-sqrt class-weighted CE versus locked unweighted final50.

Forty new heads; no re-extraction, backbone updates, new subset or test access.
Weights use only the current budget's labeled training labels, for all 5 classes.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml

import run_fixed50_fusion as baseline

ROOT=baseline.ROOT
REPORT=ROOT/'reports/week2_weighted50_20261031_v01'
heads,prior=baseline.heads,baseline.prior
require,sha,write_json,write_csv=baseline.require,baseline.sha,baseline.write_json,baseline.write_csv
METRICS=['macro_f1','n1_precision','n1_recall','n1_f1','rem_f1','accuracy']


def class_weights(labels):
    y=np.asarray(labels)
    require(y.ndim==1 and len(y)>0 and np.issubdtype(y.dtype,np.integer),'Use 1-D training integer labels')
    require(np.isin(y,np.arange(5)).all(),'Training labels outside 0..4')
    counts=np.bincount(y,minlength=5)
    require((counts>0).all(),'Every stage must exist in the labeled training subset')
    weights=1./np.sqrt(counts.astype(np.float64))
    weights=(weights/weights.mean()).astype(np.float32)
    return counts,weights


def model_digest(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def expected_unweighted_schedule(seed,samples):
    # The unweighted source uses the same Linear initializer then one randperm
    # per epoch. Validation and AdamW consume no random numbers in that source.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=torch.nn.Linear(5120,5)
        initial=model_digest(model);digest=hashlib.sha256()
        for _ in range(50):
            digest.update(torch.randperm(samples).numpy().tobytes())
    return initial,digest.hexdigest()


def plan():
    return [dict(e,fixed50_run_id=e['run_id'],run_id=e['run_id'].replace('W2X_C_F50_','W2X_C_WCE50_',1))
            for e in baseline.plan()]


def train_weighted50(entry,train_x,train_y,val_x,val_y,mean,std,val_rows,official,
                     reference_folder,output,binding_sha):
    output.mkdir(parents=True,exist_ok=False)
    old_cfg=yaml.safe_load((reference_folder/'config.yaml').read_text())
    ref=torch.load(reference_folder/'final_head_epoch50.pt',map_location='cpu',weights_only=True)
    require(ref['epoch']==50 and not old_cfg['early_stopping'],'Reference must use fixed final50')
    require(np.array_equal(mean,ref['mean'].numpy()) and np.array_equal(std,ref['std'].numpy()),'Normalization changed')
    counts,weights=class_weights(train_y.numpy())
    cfg=dict(old_cfg,run_id=entry['run_id'],unweighted_fixed50_run_id=entry['fixed50_run_id'],
             role='exploratory_class_weighted_final50',loss='inverse_sqrt_class_weighted_cross_entropy',
             class_order=['W','N1','N2','N3','REM'],class_counts_labeled_train=counts.tolist(),
             class_weights=weights.tolist(),class_weight_formula='(1/sqrt(n_c))/mean_c(1/sqrt(n_c))',
             class_weight_dtype='float32',class_weight_source='current budget labeled training labels only',
             loss_reduction='PyTorch weighted mean: sum(w_y * CE)/sum(w_y) in each minibatch',
             train_loss_logging='sample-count-weighted mean of per-minibatch weighted-mean losses',
             unweighted_runtime_binding_sha256=old_cfg['runtime_binding_sha256'],
             runtime_binding_sha256=binding_sha,trainer_sha256=sha(__file__),
             optimizer_updates=50*math.ceil(len(train_x)/128),
             declared_after_fixed50_results=True,weight_search_performed=False)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    expected_initial,expected_order=expected_unweighted_schedule(entry['downstream_seed'],len(train_x))
    model,optimizer=heads.new_head(entry['downstream_seed'],dimension=5120)
    initial=model_digest(model);require(initial==expected_initial,'Initialization recipe differs')
    batch_digest=hashlib.sha256();weight_tensor=torch.from_numpy(weights)
    history=[];began=time.perf_counter()
    for epoch in range(1,51):
        model.train()
        total=0.
        for ids in torch.randperm(len(train_x)).split(128):
            batch_digest.update(ids.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(train_x[ids]),train_y[ids],weight=weight_tensor)
            require(bool(torch.isfinite(loss)),'Nonfinite classifier loss')
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                    'Nonfinite/missing classifier gradient')
            optimizer.step()
            total+=loss.item()*len(ids)
        model.eval()
        pred=heads.predict(model,val_x)
        score=official.compute_metrics(val_y,pred)['macro_f1']
        history.append(dict(epoch=epoch,train_loss=total/len(train_x),val_macro_f1=score))
        write_json(output/'history.json',history)
        print(json.dumps(dict(stage='weighted50_train',run_id=entry['run_id'],**history[-1])),flush=True)
    require(batch_digest.hexdigest()==expected_order,'Minibatch schedule differs from unweighted recipe')
    torch.save(dict(model=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),
                    epoch=50,config=cfg),output/'final_head_epoch50.pt')
    selected=torch.load(output/'final_head_epoch50.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['model'])
    probs=heads.predict(model,val_x,True)
    require(np.isfinite(probs).all() and len(probs)==len(val_rows)==len(val_y),'Prediction shape/finite mismatch')
    predictions=[]
    for row,probability in zip(val_rows,probs):
        item={k:row[k] for k in ['sample_id','subject_id','record_id','epoch_index','y_true']}
        item['y_pred']=int(probability.argmax());item.update(zip(official.PROB_COLS,map(float,probability)))
        predictions.append(item)
    write_csv(output/'predictions_val.csv',predictions)
    frame=official.load_predictions(output/'predictions_val.csv')
    metrics=official.evaluate(output/'predictions_val.csv',n_boot=1000,seed=0)
    write_json(output/'metrics.json',metrics)
    official.per_subject_table(frame).to_csv(output/'per_subject_metrics.csv',index=False)
    write_csv(output/'confusion_matrix.csv',[dict(true_stage=stage,**dict(zip(official.STAGE_NAMES,row)))
              for stage,row in zip(official.STAGE_NAMES,metrics['confusion_matrix'])])
    audit=dict(status='passed',normalization_matches_unweighted_exactly=True,
        class_counts=counts.tolist(),class_weights=weights.tolist(),class_weights_train_only=True,
        initial_model_sha256=initial,minibatch_order_sha256=batch_digest.hexdigest(),
        matches_locked_unweighted_initializer_and_rng_recipe=True,
        reference_batch_order_recomputed_from_source_not_claimed_as_archived_trace=True,
        selected_epoch=50,weighted_objective_is_only_training_change=True)
    write_json(output/'paired_design_audit.json',audit)
    elapsed=time.perf_counter()-began
    result=dict(status='complete',run_id=entry['run_id'],fixed50_run_id=entry['fixed50_run_id'],
        representation=entry['representation'],adaptation_seed=entry['adaptation_seed'],source_key=entry['source_key'],
        budget=entry['budget'],seed=entry['downstream_seed'],epochs=50,selected_epoch=50,
        val_macro_f1=metrics['macro_f1'],n1_precision=metrics['n1_precision'],n1_recall=metrics['n1_recall'],
        n1_f1=metrics['n1_f1'],rem_f1=metrics['per_class_f1']['REM'],accuracy=metrics['accuracy'],
        seconds=elapsed,paired_design_verified=True,test_used=False)
    write_csv(output/'resource_profile.csv',[dict(device='cpu',threads=4,seconds=elapsed,epochs=50,
        optimizer_updates=cfg['optimizer_updates'],train_samples=len(train_x),val_samples=len(val_x),test_used=False)])
    (output/'run_summary.md').write_text(f"# {entry['run_id']}\n\nExploratory weighted CE; fixed final epoch50. "
        f"Macro-F1 {metrics['macro_f1']:.8f}; N1 P/R/F1 {metrics['n1_precision']:.8f}/"
        f"{metrics['n1_recall']:.8f}/{metrics['n1_f1']:.8f}. All scores from the same final head. Test unused.\n",encoding='utf-8')
    write_json(output/'summary.json',result)
    write_json(output/'artifact_sha256.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    return result


def verify_reference():
    status=json.loads((baseline.REPORT/'summary.json').read_text())
    audit=json.loads((baseline.REPORT/'completion_audit.json').read_text())
    require(status['status']=='complete' and status['completed_heads']==40 and audit['status']=='passed' and audit['runs']==40,
            'Fixed50 comparison is incomplete/unverified')
    baseline.check_binding()
    for e in plan():
        folder=ROOT/'runs'/e['fixed50_run_id'];hashes=json.loads((folder/'artifact_sha256.json').read_text())
        require(all(sha(folder/name)==value for name,value in hashes.items()),'Reference artifact SHA changed')
        require(json.loads((folder/'prefix_replay_audit.json').read_text())['status']=='passed','Reference prefix replay failed')


def prepare():
    verify_reference()
    REPORT.mkdir(exist_ok=False)
    entries=plan();require(len(entries)==len({e['run_id'] for e in entries})==40,'Bad matrix')
    require(all(not (ROOT/'runs'/e['run_id']).exists() for e in entries),'Would overwrite a run')
    canonical=heads.read_csv(heads.SOURCES['S0']/'feature_index.csv');subsets=heads.read_csv(heads.SUBSETS)
    weights_by_budget={}
    for budget in ['label10','label25']:
        selected,ids,_=heads.selected_positions(canonical,subsets,budget)
        counts,weights=class_weights(np.array([int(canonical[i]['y_true']) for i in ids],dtype=np.int64))
        weights_by_budget[budget]=dict(subjects=sorted(selected),training_samples=len(ids),counts=counts.tolist(),weights=weights.tolist())
    cfg=dict(experiment='exploratory_inverse_sqrt_class_weighted_final50_v01',registered_at=baseline.now(),
        declared_after_original_and_fixed50_results=True,locked_before_any_new_training=True,
        planned_new_heads=40,reused_unweighted_final50_heads=40,budgets=['label10','label25'],
        adaptation_seeds=[0,1,2],downstream_seeds=list(range(5)),
        weighting=dict(formula='(1/sqrt(n_c))/mean_c(1/sqrt(n_c))',dtype='float32',
                       counts_source='current labeled training subset only',weight_search=False,all_five_classes=True),
        weights_by_budget=weights_by_budget,epochs=50,selection='final_epoch_50',early_stopping=False,
        optimizer='AdamW lr0.001 wd0.0001',batch_size=128,input_dim=5120,
        loss_reduction='PyTorch weighted-mean cross_entropy in each batch',
        comparator='locked unweighted final50, never original earlystop when attributing loss effect',
        test_used=False,new_backbone_training=False,new_feature_extraction=False,
        EOG_source='C local reconstruction using original D frozen protocol',D_full_cache_received=False,
        interpretation='Loss sensitivity and precision/recall tradeoffs; gain alone does not prove the former model ignored N1')
    write_json(REPORT/'config.json',cfg);write_csv(REPORT/'run_plan.csv',entries)
    (REPORT/'实验方案.md').write_text('''# 类别加权：固定50轮对照

这是看过原始及固定50轮结果后登记的探索实验，不替换原预登记结论。固定50轮的40个未加权头已经完成并核验，作为唯一损失效应对照；新训练40个加权头，矩阵为S0及3份S1、label10/25、下游seed0–4。

只改变交叉熵类别权重：对当前预算的有标签训练样本统计五类计数n_c，令w_c=1/sqrt(n_c)，再将五类权重归一化至算术均值1，转换float32。PyTorch按每个batch的标签权重和归一化损失。权重在同一预算全部S0/S1及种子间相同；不读取验证标签来定权，不搜索倍率，不单独加强N1。缺失训练类别直接失败，不借用验证计数。

仍固定训练50轮并仅取最后一轮，不早停或按验证曲线选模。模型、初始化、批次顺序、归一化、名单和优化器保持原固定预算配方；逐run核对归一化、初始化与批次随机数配方。原未加权训练未归档批次摘要，本实验的参考摘要从其锁定源码和seed重新计算，不冒充原始观测日志。

主分析：40组加权−未加权配对差，报告Macro-F1、N1 precision/recall/F1、REM F1、accuracy及每类指标；另保留30组S1−S0在两种损失下的差值。按同一8名验证者做配对bootstrap，不能把分类种子当独立主干或受试者。若N1召回上升但精确率或Macro-F1下降，应报告权衡；任何涨分都不自动证明原目标忽略N1，不能据此宣称测试泛化。

同一份C本地重建冻结EOG用于全部设置，D原始缓存未接收；不依赖新成员产物。原固定50轮运行、正式适配配置/锁和结果保持原字节。无新主干训练或特征提取，测试集继续封存。结束后自动生成40组损失效应、30组适配对比、按人区间、完整性审计及完成报告。
''',encoding='utf-8')
    write_json(REPORT/'summary.json',dict(status='prepared_not_started',stage='tests_and_binding_pending',completed_heads=0,expected_heads=40,test_used=False))


def bind():
    require(not (REPORT/'runtime_binding.json').exists(),'Do not replace binding')
    require('\nOK\n' in (REPORT/'runtime_tests.log').read_text(),'Tests must pass')
    verify_reference()
    files=[Path(__file__),ROOT/'tests/test_weighted50_fusion.py',Path(baseline.__file__),Path(heads.__file__),
           Path(prior.__file__),ROOT/'scripts/validate_adapt_lock.py',ROOT/'scripts/replication_feature_head_runtime.py',
           REPORT/'config.json',REPORT/'run_plan.csv',REPORT/'实验方案.md',REPORT/'runtime_tests.log',
           baseline.REPORT/'summary.json',baseline.REPORT/'completion_audit.json',baseline.REPORT/'runtime_binding.json',
           baseline.REPORT/'run_registry.csv',baseline.REPORT/'pooled_reuse_audit.json',heads.SUBSETS,heads.SPLIT,heads.EVALUATOR]
    parent=json.loads((baseline.REPORT/'runtime_binding.json').read_text())
    files += [ROOT/p for p in parent['input_sha256']]
    for e in plan():
        folder=ROOT/'runs'/e['fixed50_run_id']
        files += [folder/p for p in json.loads((folder/'artifact_sha256.json').read_text())]+[folder/'artifact_sha256.json']
    write_json(REPORT/'runtime_binding.json',dict(status='bound_before_weighted_training',bound_at=baseline.now(),
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},test_used=False))


def check_binding():
    value=json.loads((REPORT/'runtime_binding.json').read_text())
    for relative,digest in value['input_sha256'].items():
        require(sha(ROOT/relative)==digest,'Locked input changed: '+relative)


def verify_pools():
    previous=json.loads((baseline.REPORT/'pooled_reuse_audit.json').read_text())
    require(previous['status']=='passed' and len(previous['sources'])==5,'Invalid reference pools')
    paths={};checks=[]
    for source in previous['sources']:
        p=ROOT/source['path'];require(sha(p)==source['sha256'],'Pool changed')
        x=np.load(p,mmap_mode='r',allow_pickle=False)
        require(x.shape==(154124,2560) and x.dtype==np.float32,'Invalid pool shape/dtype');del x
        paths[source['source']]=p;checks.append(dict(source,sha_reverified=True))
        print(json.dumps(dict(stage='verify_existing_pool',source=source['source'])),flush=True)
    canonical=heads.read_csv(heads.SOURCES['S0']/'feature_index.csv');heads.audit_index(canonical,heads.read_csv(heads.SPLIT))
    write_json(REPORT/'pooled_reuse_audit.json',dict(status='passed',sources=checks))
    return canonical,paths


def metric(metrics,key):
    return metrics['per_class_f1']['REM'] if key=='rem_f1' else metrics[key]


def analyze(results,official):
    contrasts=[];aggregates=[];cache={};audits=[]
    canonical=[r for r in heads.read_csv(heads.SOURCES['S0']/'feature_index.csv') if r['split']=='val']
    identity=[(r['sample_id'],r['subject_id'],r['record_id'],int(r['epoch_index']),int(r['y_true'])) for r in canonical]
    bootstrap_dir=REPORT/'loss_effect_bootstrap';bootstrap_dir.mkdir()
    for result in results:
        folder=ROOT/'runs'/result['run_id'];reference=ROOT/'runs'/result['fixed50_run_id']
        hashes=json.loads((folder/'artifact_sha256.json').read_text())
        require(all(sha(folder/name)==value for name,value in hashes.items()),'Weighted artifact changed')
        checkpoint=torch.load(folder/'final_head_epoch50.pt',map_location='cpu',weights_only=True)
        history=json.loads((folder/'history.json').read_text())
        cfg=yaml.safe_load((folder/'config.yaml').read_text())
        require(checkpoint['config']==cfg and checkpoint['epoch']==50 and len(history)==50,'Wrong selected head/budget')
        paired=json.loads((folder/'paired_design_audit.json').read_text());require(paired['status']=='passed','Pair design failed')
        weighted=official.load_predictions(folder/'predictions_val.csv');unweighted=official.load_predictions(reference/'predictions_val.csv')
        for frame in [weighted,unweighted]:
            actual=list(frame[['sample_id','subject_id','record_id','epoch_index','y_true']].itertuples(index=False,name=None))
            require(actual==identity and np.isfinite(frame[official.PROB_COLS].to_numpy()).all(),'Validation identity differs')
        w=official.compute_metrics(weighted.y_true.to_numpy(),weighted.y_pred.to_numpy())
        u=official.compute_metrics(unweighted.y_true.to_numpy(),unweighted.y_pred.to_numpy())
        stored=json.loads((folder/'metrics.json').read_text())
        require(all(stored[k]==v for k,v in w.items()),'Official weighted metric mismatch')
        row={k:result[k] for k in ['run_id','fixed50_run_id','budget','representation','adaptation_seed','seed']}
        for key in METRICS:
            row['unweighted_'+key]=metric(u,key);row['weighted_'+key]=metric(w,key)
            row['delta_'+key+'_pp']=100*(metric(w,key)-metric(u,key))
        contrasts.append(row)
        write_json(bootstrap_dir/(result['run_id']+'.json'),official.paired_bootstrap_delta(weighted,unweighted,n_boot=1000,seed=0))
        cache[(result['budget'],result['adaptation_seed'],result['seed'])]=dict(weighted=w,unweighted=u)
        audits.append(dict(run_id=result['run_id'],artifact_files_verified=len(hashes),epochs=50,selected_epoch=50,
                           weights_train_only=True,validation_identity_verified=True,official_metrics_match=True,paired_design_verified=True))
    write_csv(REPORT/'weighted_minus_unweighted.csv',contrasts)
    for budget in ['label10','label25']:
        for representation in ['S0','S1']:
            group=[r for r in contrasts if r['budget']==budget and r['representation']==representation]
            for key in METRICS:
                unweighted=float(np.mean([r['unweighted_'+key] for r in group]));weighted=float(np.mean([r['weighted_'+key] for r in group]))
                aggregates.append(dict(budget=budget,representation=representation,metric=key,
                    unweighted_mean=unweighted,weighted_mean=weighted,delta_pp=100*(weighted-unweighted),
                    independent_adaptations=0 if representation=='S0' else 3,heads=len(group)))
    write_csv(REPORT/'overall_loss_effect.csv',aggregates)
    adaptation=[]
    for budget in ['label10','label25']:
        for a in range(3):
            for seed in range(5):
                b,s=cache[(budget,None,seed)],cache[(budget,a,seed)]
                row=dict(budget=budget,adaptation_seed=a,downstream_seed=seed)
                for key in METRICS:
                    for loss in ['weighted','unweighted']:
                        row[loss+'_S1_minus_S0_'+key+'_pp']=100*(metric(s[loss],key)-metric(b[loss],key))
                adaptation.append(row)
    write_csv(REPORT/'adaptation_effect_by_loss.csv',adaptation)
    for a in range(3):
        out=REPORT/f'weighted_comparison_aseed{a}';out.mkdir()
        heads.paired_report([r for r in results if r['representation']=='S0' or r['adaptation_seed']==a],out,official)
    check_binding()
    write_json(REPORT/'completion_audit.json',dict(status='passed',runs=40,checks=audits,test_used=False))
    lines=['# 类别加权固定50轮：完成报告','',f'完成于 {baseline.now()}。40个加权头与40个既有未加权固定50轮头配对。',
        '', '| 预算 | 表征 | 指标 | 未加权 | 加权 | 差值（百分点） |','| --- | --- | --- | ---: | ---: | ---: |']
    for r in aggregates:
        lines.append(f"| {r['budget']} | {r['representation']} | {r['metric']} | {100*r['unweighted_mean']:.3f}% | {100*r['weighted_mean']:.3f}% | {r['delta_pp']:+.3f} |")
    lines += ['', 'S0每预算5个头；S1先对每个主干5个头求均值，再等权平均3主干。头部种子不是独立适配或受试者。40组损失配对见weighted_minus_unweighted.csv，30组适配效果见adaptation_effect_by_loss.csv；按8人bootstrap区间保留每格结果，不将挑出的最佳格推广为总体效果。', '',
        '只改变五类逆平方根权重，其计数仅来自各预算允许的有标签训练者；不按验证分数定权或重新选轮次。N1召回、精确率与Macro-F1必须一起解释；任何改善仅支持当前设定的损失敏感性，不能自动证明原模型忽略N1，也不能替代原早停正式结果或证明独立测试泛化。', '',
        '测试集未使用。EOG为C按D原始冻结协议本地重建，D原缓存未收到。原始适配及固定50轮记录均保留。本实验未重新训练主干或提取特征。']
    (REPORT/'类别加权_完成报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    check_binding();require(not (REPORT/'run_registry.csv').exists(),'Do not overwrite execution')
    with (REPORT/'.weighted50.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(status='running',stage='verify_existing_pools',pid=os.getpid(),started_at=baseline.now(),
                    completed_heads=0,expected_heads=40,test_used=False)
        write_json(REPORT/'summary.json',status)
        try:
            canonical,paths=verify_pools();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
            official=heads.official_module();subsets=heads.read_csv(heads.SUBSETS);results=[]
            eog=np.load(paths['EOG'],mmap_mode='r',allow_pickle=False)
            for entry in plan():
                selected,train_ids,val_ids=heads.selected_positions(canonical,subsets,entry['budget'])
                eeg=np.load(paths[entry['source_key']],mmap_mode='r',allow_pickle=False)
                train=prior.concatenate(eeg[train_ids],eog[train_ids]);val=prior.concatenate(eeg[val_ids],eog[val_ids])
                mean,std=heads.fit_normalization(train);train-=mean;train/=std;val-=mean;val/=std
                require(np.isfinite(train).all() and np.isfinite(val).all(),'Nonfinite normalized features')
                train_x,val_x=torch.from_numpy(train),torch.from_numpy(val)
                train_y=torch.tensor([int(canonical[i]['y_true']) for i in train_ids],dtype=torch.int64)
                val_rows=[canonical[i] for i in val_ids];val_y=np.array([int(r['y_true']) for r in val_rows])
                require(len(val_rows)==21186 and len({r['subject_id'] for r in val_rows})==8,'Invalid validation coverage')
                counts,weights=class_weights(train_y.numpy());locked=json.loads((REPORT/'config.json').read_text())['weights_by_budget'][entry['budget']]
                require(counts.tolist()==locked['counts'] and weights.tolist()==locked['weights'],'Class weights changed')
                status.update(stage='weighted50_head_training',current_run=entry['run_id']);write_json(REPORT/'summary.json',status)
                result=train_weighted50(entry,train_x,train_y,val_x,val_y,mean,std,val_rows,official,
                    ROOT/'runs'/entry['fixed50_run_id'],ROOT/'runs'/entry['run_id'],sha(REPORT/'runtime_binding.json'))
                results.append(result);write_csv(REPORT/'run_registry.csv',results)
                status.update(completed_heads=len(results));write_json(REPORT/'summary.json',status)
                del eeg,train,val,train_x,val_x,train_y
            status.update(stage='paired_loss_and_adaptation_analysis',current_run=None);write_json(REPORT/'summary.json',status)
            analyze(results,official)
            status.update(status='complete',stage='weighted50_matrix_and_analysis_complete',completed_at=baseline.now(),
                          selected_epoch_all_runs=50,paired_design_verified_runs=40)
        except BaseException:
            status.update(status='failed',error=traceback.format_exc(),failed_at=baseline.now());raise
        finally:
            write_json(REPORT/'summary.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','bind','run']);args=p.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='bind':
        bind();value=json.loads((REPORT/'runtime_binding.json').read_text())
        print(json.dumps(dict(status=value['status'],bound_inputs=len(value['input_sha256']))))
    else:run()


if __name__=='__main__':main()
