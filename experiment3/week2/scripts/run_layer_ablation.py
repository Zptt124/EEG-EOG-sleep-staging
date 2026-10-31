"""Exploratory EEG layer masking: all20 reference versus last5 and last2.

Use pooled per-layer coordinates; keep EOG and the 5120-D Linear head unchanged.
Ordinary CE, 50 epochs, final checkpoint only. Original runs remain immutable.
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
from run_weighted50_fusion import model_digest,expected_unweighted_schedule

ROOT=baseline.ROOT
REPORT=ROOT/'reports/week2_layer_ablation_20261031_v01'
heads,prior=baseline.heads,baseline.prior
require,sha,write_json,write_csv=baseline.require,baseline.sha,baseline.write_json,baseline.write_csv
GROUPS={'all20':list(range(20)),'last5':list(range(15,20)),'last2':list(range(18,20))}
METRICS=['macro_f1','n1_precision','n1_recall','n1_f1','rem_f1','accuracy']


def retained_columns(group):
    require(group in GROUPS,'Unregistered layer group')
    return np.array([128*layer+j for layer in GROUPS[group] for j in range(128)],dtype=np.int64)


def mask_input(values,group):
    require(values.ndim==2 and values.shape[1]==5120 and values.dtype==np.float32,'Invalid normalized fusion input')
    require(np.isfinite(values).all(),'Nonfinite features before masking')
    keep=np.zeros(2560,dtype=bool);keep[retained_columns(group)]=True
    values[:,np.flatnonzero(~keep)]=0.
    return values


def plan():
    return [dict(e,layer_group=group,reference_run_id=e['run_id'],
                 run_id=e['run_id'].replace('W2X_C_F50_',f'W2X_C_LA_{group}_F50_',1))
            for group in ['last5','last2'] for e in baseline.plan()]


def validate_reference():
    status=json.loads((baseline.REPORT/'summary.json').read_text())
    audit=json.loads((baseline.REPORT/'completion_audit.json').read_text())
    require(status['status']=='complete' and status['completed_heads']==40 and audit['status']=='passed' and audit['runs']==40,
            'All20 fixed50 reference incomplete')
    baseline.check_binding()
    for entry in baseline.plan():
        folder=ROOT/'runs'/entry['run_id'];hashes=json.loads((folder/'artifact_sha256.json').read_text())
        require(all(sha(folder/name)==value for name,value in hashes.items()),'Reference artifact changed')
        cfg=yaml.safe_load((folder/'config.yaml').read_text())
        require(cfg['loss']=='unweighted_cross_entropy' and not cfg['early_stopping'] and cfg['selected_epoch']==50,
                'Reference is not unweighted final50')


def prepare():
    validate_reference()
    REPORT.mkdir(exist_ok=False)
    entries=plan();require(len(entries)==len({r['run_id'] for r in entries})==80,'Wrong ablation matrix')
    require(all(not (ROOT/'runs'/r['run_id']).exists() for r in entries),'Would overwrite an output')
    config=dict(experiment='exploratory_EEG_layer_masking_final50_v01',registered_at=baseline.now(),
        selected_after_previous_results=True,groups_fixed_before_layer_shift_analysis=True,
        layer_groups_zero_based=GROUPS,new_groups=['last5','last2'],reference_group='all20',
        planned_new_heads=80,reused_all20_heads=40,adaptation_seeds=[0,1,2],downstream_seeds=list(range(5)),
        budgets=['label10','label25'],mask_modality='EEG only',EOG_layers='all20 unchanged',
        implementation='zero excluded EEG coordinates after fitting/applying training-only per-feature normalization',
        physical_head_dim=5120,nominal_head_parameters=25605,
        active_EEG_dimensions={k:len(v)*128 for k,v in GROUPS.items()},
        active_input_dimensions={k:len(v)*128+2560 for k,v in GROUPS.items()},
        classifier=dict(name='Linear(5120,5)',optimizer='AdamW',learning_rate=.001,weight_decay=.0001,
                        batch_size=128,epochs=50,early_stopping=False,selection='final_epoch_50',loss='unweighted_cross_entropy'),
        layer_feature_layout='layer-major flattened [20,128], float32 mean over 30 time windows within each layer',
        drift_analysis=dict(subjects='all54 training subjects only',labels_used=False,statistics_fit_classifier=False),
        original_features_were_concatenated_not_averaged_across_layers=True,
        interpretation='Masks reduce effective EEG capacity despite constant nominal head dimension; effects do not by themselves establish hidden adaptation information',
        test_used=False,new_backbone_training=False,new_feature_extraction=False,
        EOG_source='C local reconstruction using original D frozen protocol',D_full_cache_received=False)
    write_json(REPORT/'config.json',config);write_csv(REPORT/'run_plan.csv',entries)
    (REPORT/'实验方案.md').write_text('''# EEG取层消融：all20、last5、last2

本方案在看过前序结果后登记，是探索性补充，三个取层分组在观察逐层变化统计前固定。原始20层特征是拼接，不是层间平均，因此不能预设末层信息被平均稀释。

只改变EEG可见层：all20为零基0–19；last5为15–19；last2为18–19（即人类编号16–20、19–20）。EOG始终保留全部20层。同预算的S0与三个S1都执行相同掩码，下游seed0–4，10%/25%两档。all20复用40个已完成的未加权固定50轮结果，last5/last2各新增40个头，总计80新头、120个比较结果。

缓存保留[N,20,128]的层坐标，直接在既有2560维池化向量中选层；不重新读取原始信号或提取特征。归一化先按当前预算有标签训练者拟合，原5120维EEG+EOG向量标准化后将未选EEG维度置零，EOG不变。三组均使用同一5120维Linear头（名义25605参数）、同一初始化与批次顺序。活跃输入维度仍从5120降至3200/2816，故有效容量/正则化影响尚未被完全消除，不能只因涨分就断言适配信息被原表征掩盖。

只用普通交叉熵，不叠加类别加权；固定50轮并取第50轮，不早停或按验证曲线选头。模型、学习率、weight decay、批次、名单、标准化及评价规则沿用未加权final50。逐run核验归一化与all20一致，初始化和批次随机数配方一致；参考批次摘要由锁定源码/seed重算，不冒充原归档轨迹。

预先做无监督逐层变化检查：只使用54名训练者的原始池化特征，计算三份S1相对S0每层的相对L2、RMSE、余弦和逐值完全相同占比；不参考分期标签，不参与选层或分类器标准化。数值变化不等于新增的分期信息。

主分析：每种取层的S1−S0配对差，以及该差相对all20的变化；同时报告每个S0/S1相对all20的效果、N1 precision/recall/F1、Macro-F1、REM F1和accuracy。保留全部主干/预算/种子，不按最佳格挑结果。只有同一8名开发验证者，分类种子不是独立主干或独立验证集。

EOG为C按D同协议本地重建，D原缓存未接收；无需其他成员新产物。测试集继续封存，原正式、固定50轮和加权实验均不改写。结束后自动生成逐层变化图、配对表、区间、完整性审计及完成报告。
''',encoding='utf-8')
    write_json(REPORT/'summary.json',dict(status='prepared_not_started',stage='tests_and_binding_pending',
        completed_heads=0,expected_heads=80,reused_all20_heads=40,test_used=False))


def bind():
    require(not (REPORT/'runtime_binding.json').exists(),'Do not replace runtime binding')
    require('\nOK\n' in (REPORT/'runtime_tests.log').read_text(),'Tests must pass')
    validate_reference()
    files=[Path(__file__),ROOT/'tests/test_layer_ablation.py',Path(baseline.__file__),Path(heads.__file__),Path(prior.__file__),
           ROOT/'scripts/run_weighted50_fusion.py',ROOT/'scripts/replication_feature_head_runtime.py',ROOT/'scripts/validate_adapt_lock.py',
           ROOT/'scripts/extract_features.py',ROOT/'vendor/EEGDM/model/classifier.py',
           REPORT/'config.json',REPORT/'run_plan.csv',REPORT/'实验方案.md',REPORT/'runtime_tests.log',
           baseline.REPORT/'summary.json',baseline.REPORT/'completion_audit.json',baseline.REPORT/'run_registry.csv',
           baseline.REPORT/'runtime_binding.json',baseline.REPORT/'pooled_reuse_audit.json',heads.SPLIT,heads.SUBSETS,heads.EVALUATOR]
    parent=json.loads((baseline.REPORT/'runtime_binding.json').read_text())
    files += [ROOT/name for name in parent['input_sha256']]
    for e in baseline.plan():
        folder=ROOT/'runs'/e['run_id']
        files += [folder/name for name in json.loads((folder/'artifact_sha256.json').read_text())]+[folder/'artifact_sha256.json']
    write_json(REPORT/'runtime_binding.json',dict(status='bound_before_layer_analysis_and_training',bound_at=baseline.now(),
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},test_used=False))


def check_binding():
    for name,value in json.loads((REPORT/'runtime_binding.json').read_text())['input_sha256'].items():
        require(sha(ROOT/name)==value,'Bound input changed: '+name)


def verify_pools():
    previous=json.loads((baseline.REPORT/'pooled_reuse_audit.json').read_text())
    require(previous['status']=='passed' and len(previous['sources'])==5,'Invalid source pools')
    paths={};checks=[]
    for source in previous['sources']:
        path=ROOT/source['path'];require(sha(path)==source['sha256'],'Source pool checksum differs')
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        require(values.shape==(154124,2560) and values.dtype==np.float32,'Invalid pooled layout');del values
        paths[source['source']]=path;checks.append(dict(source,sha_reverified=True))
        print(json.dumps(dict(stage='verify_existing_pool',source=source['source'])),flush=True)
    cfg=json.loads((prior.REPORT/'fusion_config.json').read_text())
    for source in cfg['EEG_sources']:
        feature=json.loads((ROOT/source['feature_root']/'feature_config.json').read_text())
        require(feature['layers']==list(range(20)) and feature['axes']==['sample','layer','time_window','hidden']
                and feature['feature_shape']==[20,30,128],'Layer axis order changed')
    canonical=heads.read_csv(heads.SOURCES['S0']/'feature_index.csv');heads.audit_index(canonical,heads.read_csv(heads.SPLIT))
    write_json(REPORT/'pooled_reuse_audit.json',dict(status='passed',sources=checks,layer_major_order_verified=True))
    return canonical,paths


def layer_shift(paths,canonical):
    train_mask=np.array([r['split']=='train' for r in canonical]);require(int(train_mask.sum())==132938,'Wrong training set')
    s0=np.load(paths['S0'],mmap_mode='r',allow_pickle=False);rows=[]
    for seed in range(3):
        s1=np.load(paths[f'S1_aseed{seed}'],mmap_mode='r',allow_pickle=False)
        aa=np.zeros(20);bb=np.zeros(20);ab=np.zeros(20);dd=np.zeros(20)
        exact=np.zeros(20,dtype=np.int64);maximum=np.zeros(20);elements=0
        for start in range(0,len(s0),1024):
            ids=np.flatnonzero(train_mask[start:start+1024])+start
            if not len(ids):continue
            a=np.asarray(s0[ids].reshape(-1,20,128),dtype=np.float64)
            b=np.asarray(s1[ids].reshape(-1,20,128),dtype=np.float64);d=b-a
            require(np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite layer drift input')
            aa+=(a*a).sum(axis=(0,2));bb+=(b*b).sum(axis=(0,2));ab+=(a*b).sum(axis=(0,2));dd+=(d*d).sum(axis=(0,2))
            exact+=(a==b).sum(axis=(0,2));maximum=np.maximum(maximum,np.max(np.abs(d),axis=(0,2)))
            elements+=len(ids)*128
        for layer in range(20):
            rows.append(dict(adaptation_seed=seed,layer_zero_based=layer,layer_one_based=layer+1,
                train_samples=132938,labels_used=False,rmse=float(np.sqrt(dd[layer]/elements)),
                S0_rms=float(np.sqrt(aa[layer]/elements)),S1_rms=float(np.sqrt(bb[layer]/elements)),
                relative_l2=float(np.sqrt(dd[layer]/max(aa[layer],1e-30))),
                cosine=float(np.clip(ab[layer]/max(np.sqrt(aa[layer]*bb[layer]),1e-30),-1,1)),
                exact_fraction=float(exact[layer]/elements),max_abs_delta=float(maximum[layer])))
        del s1
        print(json.dumps(dict(stage='training_only_layer_shift',adaptation_seed=seed,layers=20)),flush=True)
    write_csv(REPORT/'layer_shift_train_only.csv',rows)
    return rows


def train_masked50(entry,train_x,train_y,val_x,val_y,mean,std,val_rows,official,reference_folder,output,binding_sha):
    output.mkdir(parents=True,exist_ok=False)
    old_cfg=yaml.safe_load((reference_folder/'config.yaml').read_text())
    reference=torch.load(reference_folder/'final_head_epoch50.pt',map_location='cpu',weights_only=True)
    require(reference['epoch']==50 and not old_cfg['early_stopping'] and old_cfg['loss']=='unweighted_cross_entropy','Invalid all20 reference')
    require(np.array_equal(mean,reference['mean'].numpy()) and np.array_equal(std,reference['std'].numpy()),'Normalization differs from all20')
    group=entry['layer_group'];keep=retained_columns(group)
    excluded=np.setdiff1d(np.arange(2560),keep)
    require(bool((train_x[:,excluded]==0).all()) and bool((val_x[:,excluded]==0).all()),'Excluded EEG coordinates are not zero')
    cfg=dict(old_cfg,run_id=entry['run_id'],all20_reference_run_id=entry['reference_run_id'],
        role='exploratory_EEG_layer_masking_final50',layer_group=group,EEG_layers_zero_based=GROUPS[group],EOG_layers_zero_based=list(range(20)),
        active_EEG_dim=len(keep),active_input_dim=len(keep)+2560,physical_input_dim=5120,
        mask_location='after training-only per-feature normalization',nominal_head_parameters=25605,
        all20_runtime_binding_sha256=old_cfg['runtime_binding_sha256'],runtime_binding_sha256=binding_sha,
        trainer_sha256=sha(__file__),optimizer_updates=50*math.ceil(len(train_x)/128))
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    expected_initial,expected_order=expected_unweighted_schedule(entry['downstream_seed'],len(train_x))
    model,optimizer=heads.new_head(entry['downstream_seed'],dimension=5120)
    initial=model_digest(model);require(initial==expected_initial,'Head initialization differs')
    batch_digest=hashlib.sha256();history=[];began=time.perf_counter()
    for epoch in range(1,51):
        model.train()
        total=0.
        for ids in torch.randperm(len(train_x)).split(128):
            batch_digest.update(ids.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(train_x[ids]),train_y[ids])
            require(bool(torch.isfinite(loss)),'Nonfinite classifier loss')
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),'Nonfinite/missing classifier gradient')
            optimizer.step()
            total+=loss.item()*len(ids)
        model.eval();pred=heads.predict(model,val_x);score=official.compute_metrics(val_y,pred)['macro_f1']
        history.append(dict(epoch=epoch,train_loss=total/len(train_x),val_macro_f1=score))
        write_json(output/'history.json',history)
        print(json.dumps(dict(stage='layer_mask_train',run_id=entry['run_id'],**history[-1])),flush=True)
    require(batch_digest.hexdigest()==expected_order,'Batch order differs from reference recipe')
    torch.save(dict(model=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),epoch=50,config=cfg),output/'final_head_epoch50.pt')
    model.load_state_dict(torch.load(output/'final_head_epoch50.pt',map_location='cpu',weights_only=True)['model'])
    probs=heads.predict(model,val_x,True);require(np.isfinite(probs).all() and len(probs)==len(val_rows)==len(val_y),'Prediction shape/finite mismatch')
    predictions=[]
    for row,prob in zip(val_rows,probs):
        item={k:row[k] for k in ['sample_id','subject_id','record_id','epoch_index','y_true']}
        item['y_pred']=int(prob.argmax());item.update(zip(official.PROB_COLS,map(float,prob)));predictions.append(item)
    write_csv(output/'predictions_val.csv',predictions);frame=official.load_predictions(output/'predictions_val.csv')
    metrics=official.evaluate(output/'predictions_val.csv',n_boot=1000,seed=0);write_json(output/'metrics.json',metrics)
    official.per_subject_table(frame).to_csv(output/'per_subject_metrics.csv',index=False)
    write_csv(output/'confusion_matrix.csv',[dict(true_stage=stage,**dict(zip(official.STAGE_NAMES,row))) for stage,row in zip(official.STAGE_NAMES,metrics['confusion_matrix'])])
    write_json(output/'paired_design_audit.json',dict(status='passed',normalization_matches_all20_exactly=True,
        nominal_head_parameters=25605,layer_group=group,excluded_EEG_coordinates_zero=True,
        initial_model_sha256=initial,minibatch_order_sha256=batch_digest.hexdigest(),
        matches_locked_unweighted_initializer_and_rng_recipe=True,reference_batch_order_is_recomputed_recipe=True,
        selected_epoch=50,class_weighting_used=False))
    elapsed=time.perf_counter()-began
    result=dict(status='complete',run_id=entry['run_id'],reference_run_id=entry['reference_run_id'],layer_group=group,
        representation=entry['representation'],adaptation_seed=entry['adaptation_seed'],source_key=entry['source_key'],
        budget=entry['budget'],seed=entry['downstream_seed'],epochs=50,selected_epoch=50,
        val_macro_f1=metrics['macro_f1'],n1_precision=metrics['n1_precision'],n1_recall=metrics['n1_recall'],
        n1_f1=metrics['n1_f1'],rem_f1=metrics['per_class_f1']['REM'],accuracy=metrics['accuracy'],seconds=elapsed,
        paired_design_verified=True,test_used=False)
    write_csv(output/'resource_profile.csv',[dict(device='cpu',threads=4,seconds=elapsed,epochs=50,
        optimizer_updates=cfg['optimizer_updates'],train_samples=len(train_x),val_samples=len(val_x),test_used=False)])
    (output/'run_summary.md').write_text(f"# {entry['run_id']}\n\nEEG {group}; EOG all20; unweighted CE, final epoch50. "
        f"Macro-F1 {metrics['macro_f1']:.8f}; N1 F1 {metrics['n1_f1']:.8f}. Test unused.\n",encoding='utf-8')
    write_json(output/'summary.json',result);write_json(output/'artifact_sha256.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    return result


def get_metric(metrics,key):
    return metrics['per_class_f1']['REM'] if key=='rem_f1' else metrics[key]


def analyze(results,official,shift):
    all_results=[dict(r,layer_group='all20',reference_run_id=r['run_id']) for r in heads.read_csv(baseline.REPORT/'run_registry.csv')]
    # CSV references are parsed explicitly; new result dictionaries keep native types.
    for r in all_results:
        r['seed']=int(r['seed']);r['adaptation_seed']=None if r['adaptation_seed']=='' else int(r['adaptation_seed'])
        for key in ['val_macro_f1','n1_precision','n1_recall','n1_f1','rem_f1','accuracy']:r[key]=float(r[key])
    all_results+=results;require(len(all_results)==120,'Incomplete total matrix')
    point={};frames={};audits=[]
    canonical=[r for r in heads.read_csv(heads.SOURCES['S0']/'feature_index.csv') if r['split']=='val']
    expected=[(r['sample_id'],r['subject_id'],r['record_id'],int(r['epoch_index']),int(r['y_true'])) for r in canonical]
    for r in all_results:
        folder=ROOT/'runs'/r['run_id'];hashes=json.loads((folder/'artifact_sha256.json').read_text())
        require(all(sha(folder/name)==value for name,value in hashes.items()),'Run artifact checksum mismatch')
        cfg=yaml.safe_load((folder/'config.yaml').read_text());checkpoint=torch.load(folder/'final_head_epoch50.pt',map_location='cpu',weights_only=True)
        require(checkpoint['config']==cfg and checkpoint['epoch']==50 and cfg['loss']=='unweighted_cross_entropy' and not cfg['early_stopping'],'Run selection/loss drift')
        require(len(json.loads((folder/'history.json').read_text()))==50,'Incomplete training budget')
        frame=official.load_predictions(folder/'predictions_val.csv')
        require(list(frame[['sample_id','subject_id','record_id','epoch_index','y_true']].itertuples(index=False,name=None))==expected,'Validation identity mismatch')
        measured=official.compute_metrics(frame.y_true.to_numpy(),frame.y_pred.to_numpy());stored=json.loads((folder/'metrics.json').read_text())
        require(all(stored[k]==v for k,v in measured.items()),'Official point metric mismatch')
        key=(r['layer_group'],r['budget'],r['adaptation_seed'],r['seed']);require(key not in point,'Duplicate matrix position')
        point[key]=measured;frames[key]=frame
        audits.append(dict(run_id=r['run_id'],layer_group=r['layer_group'],selected_epoch=50,artifact_files_verified=len(hashes),official_metrics_match=True))
    aggregates=[];adaptation=[];layer_pairs=[];bootstrap=REPORT/'layer_effect_bootstrap';bootstrap.mkdir()
    for group in GROUPS:
        for budget in ['label10','label25']:
            for rep in ['S0','S1']:
                keys=[k for k in point if k[0]==group and k[1]==budget and ((k[2] is None)==(rep=='S0'))]
                for metric in METRICS:
                    values=[get_metric(point[k],metric) for k in keys]
                    aggregates.append(dict(layer_group=group,budget=budget,representation=rep,metric=metric,
                        mean=float(np.mean(values)),head_sd=float(np.std(values,ddof=1)),heads=len(keys),independent_adaptations=0 if rep=='S0' else 3))
            for a in range(3):
                for seed in range(5):
                    s,b=point[(group,budget,a,seed)],point[(group,budget,None,seed)]
                    ref_s,ref_b=point[('all20',budget,a,seed)],point[('all20',budget,None,seed)]
                    row=dict(layer_group=group,budget=budget,adaptation_seed=a,downstream_seed=seed)
                    for metric in METRICS:
                        delta=100*(get_metric(s,metric)-get_metric(b,metric))
                        row['S1_minus_S0_'+metric+'_pp']=delta
                        row['change_vs_all20_in_'+metric+'_pp']=delta-100*(get_metric(ref_s,metric)-get_metric(ref_b,metric))
                    adaptation.append(row)
    for r in results:
        key=(r['layer_group'],r['budget'],r['adaptation_seed'],r['seed']);reference=('all20',*key[1:])
        row={k:r[k] for k in ['run_id','reference_run_id','layer_group','budget','representation','adaptation_seed','seed']}
        for metric in METRICS:row['delta_'+metric+'_pp']=100*(get_metric(point[key],metric)-get_metric(point[reference],metric))
        layer_pairs.append(row)
        write_json(bootstrap/(r['run_id']+'.json'),official.paired_bootstrap_delta(frames[key],frames[reference],n_boot=1000,seed=0))
    write_csv(REPORT/'all120_run_registry.csv',[{k:r[k] for k in ['run_id','reference_run_id','layer_group','budget','representation','adaptation_seed','seed']} for r in all_results])
    write_csv(REPORT/'aggregate_metrics.csv',aggregates);write_csv(REPORT/'adaptation_effect_by_layer.csv',adaptation)
    write_csv(REPORT/'layer_minus_all20.csv',layer_pairs)
    for group in ['last5','last2']:
        for a in range(3):
            directory=REPORT/f'comparison_{group}_aseed{a}';directory.mkdir()
            heads.paired_report([r for r in results if r['layer_group']==group and (r['representation']=='S0' or r['adaptation_seed']==a)],directory,official)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,3.5))
    for a in range(3):
        subset=[r for r in shift if r['adaptation_seed']==a]
        axes[0].plot([r['layer_one_based'] for r in subset],[100*r['relative_l2'] for r in subset],label=f'adaptation seed {a}')
        axes[1].plot([r['layer_one_based'] for r in subset],[100*(1-r['exact_fraction']) for r in subset],label=f'adaptation seed {a}')
    for ax,label in zip(axes,['Relative L2 change (%)','Changed pooled coordinates (%)']):
        ax.set_xlabel('EEG layer (1-based)');ax.set_ylabel(label);ax.set_xticks([1,5,10,15,19,20]);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Training subjects only; labels unused');fig.tight_layout();fig.savefig(REPORT/'layer_shift_train_only.png',dpi=170);plt.close(fig)
    lines=['# EEG取层消融：完成报告','',f'完成于 {baseline.now()}。80个新头与40个all20参考共120格已核验。','',
        '| EEG层组 | 预算 | 表征 | Macro-F1 | N1 F1 |','| --- | --- | --- | ---: | ---: |']
    for group in GROUPS:
        for budget in ['label10','label25']:
            for rep in ['S0','S1']:
                values={r['metric']:r['mean'] for r in aggregates if r['layer_group']==group and r['budget']==budget and r['representation']==rep}
                lines.append(f"| {group} | {budget} | {rep} | {100*values['macro_f1']:.3f}% | {100*values['n1_f1']:.3f}% |")
    lines += ['', 'S0每格5个头，S1为3主干各5个头的等权均值；分类种子不是独立主干或受试者。完整六指标见aggregate_metrics.csv，90组适配配对见adaptation_effect_by_layer.csv，80组相对all20配对及按8人bootstrap见layer_minus_all20.csv与layer_effect_bootstrap/。', '',
        '![逐层变化](layer_shift_train_only.png)', '',
        '原20层是拼接，未在层间平均。此实验只在标准化后遮蔽部分EEG坐标；EOG、5120维物理输入、初始化和批次随机数配方相同，普通交叉熵与final50规则固定。活跃EEG维度减少仍改变有效容量和正则化，任何改善不能单独证明适配信息被掩盖。', '',
        '逐层变化仅来自54名训练者的无标签特征，不参与预定取层分组或分类器拟合。数值变化不等于新增的分期信息。当前结果是同一8名开发验证者上的探索性观察，不替换原早停主比较，不证明测试泛化。测试集仍封存；EOG仍为C按D同协议本地重建。']
    (REPORT/'取层消融_完成报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    check_binding();write_json(REPORT/'completion_audit.json',dict(status='passed',new_heads=80,reused_heads=40,total_positions=120,checks=audits,test_used=False))


def run():
    check_binding();require(not (REPORT/'run_registry.csv').exists(),'Do not overwrite execution')
    with (REPORT/'.layer_ablation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(status='running',stage='verify_pooled_feature_sources',pid=os.getpid(),started_at=baseline.now(),
                    completed_heads=0,expected_heads=80,reused_all20_heads=40,test_used=False)
        write_json(REPORT/'summary.json',status)
        try:
            canonical,paths=verify_pools();status['stage']='training_only_layer_shift';write_json(REPORT/'summary.json',status)
            shift=layer_shift(paths,canonical);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
            official=heads.official_module();subsets=heads.read_csv(heads.SUBSETS);eog=np.load(paths['EOG'],mmap_mode='r',allow_pickle=False);results=[]
            for entry in plan():
                selected,train_ids,val_ids=heads.selected_positions(canonical,subsets,entry['budget'])
                eeg=np.load(paths[entry['source_key']],mmap_mode='r',allow_pickle=False)
                train=prior.concatenate(eeg[train_ids],eog[train_ids]);val=prior.concatenate(eeg[val_ids],eog[val_ids])
                mean,std=heads.fit_normalization(train);train-=mean;train/=std;val-=mean;val/=std
                mask_input(train,entry['layer_group']);mask_input(val,entry['layer_group'])
                train_x,val_x=torch.from_numpy(train),torch.from_numpy(val);train_y=torch.tensor([int(canonical[i]['y_true']) for i in train_ids],dtype=torch.int64)
                val_rows=[canonical[i] for i in val_ids];val_y=np.array([int(r['y_true']) for r in val_rows])
                require(len(val_rows)==21186 and len({r['subject_id'] for r in val_rows})==8,'Validation coverage differs')
                status.update(stage='layer_mask_head_training',current_run=entry['run_id'],current_layer_group=entry['layer_group']);write_json(REPORT/'summary.json',status)
                result=train_masked50(entry,train_x,train_y,val_x,val_y,mean,std,val_rows,official,
                    ROOT/'runs'/entry['reference_run_id'],ROOT/'runs'/entry['run_id'],sha(REPORT/'runtime_binding.json'))
                results.append(result);write_csv(REPORT/'run_registry.csv',results);status['completed_heads']=len(results);write_json(REPORT/'summary.json',status)
                del eeg,train,val,train_x,val_x,train_y
            status.update(stage='paired_layer_and_adaptation_analysis',current_run=None);write_json(REPORT/'summary.json',status)
            analyze(results,official,shift);status.update(status='complete',stage='layer_ablation_and_analysis_complete',completed_at=baseline.now())
        except BaseException:
            status.update(status='failed',error=traceback.format_exc(),failed_at=baseline.now());raise
        finally:write_json(REPORT/'summary.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','bind','run']);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='bind':
        bind();b=json.loads((REPORT/'runtime_binding.json').read_text());print(json.dumps(dict(status=b['status'],bound_inputs=len(b['input_sha256']))))
    else:run()


if __name__=='__main__':main()
