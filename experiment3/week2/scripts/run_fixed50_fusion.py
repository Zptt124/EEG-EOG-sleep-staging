"""Exploratory stopping/selection sensitivity: forty matched final-epoch heads.

No feature extraction, backbone fitting, test data or validation-based selection.
The original early-stop runs remain immutable and must replay exactly in prefix.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import yaml

import run_primary_fusion as prior
import run_s0_s1_eeg_comparison as heads

ROOT = heads.ROOT
REPORT = ROOT/'reports/week2_fixed50_20261031_v01'
require, sha, write_json, write_csv = heads.require, heads.sha, heads.write_json, heads.write_csv


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def plan():
    cfg = json.loads((prior.REPORT/'fusion_config.json').read_text())
    rows = prior.plan(cfg)
    return [dict(row, original_run_id=row['run_id'], run_id=row['run_id'].replace('W2_C_', 'W2X_C_F50_', 1))
            for row in rows]


def train_final50(entry, train_x, train_y, val_x, val_y, mean, std, val_rows, official,
                  original, output, binding_sha):
    output.mkdir(parents=True, exist_ok=False)
    old_cfg = yaml.safe_load((original/'config.yaml').read_text())
    old_history = json.loads((original/'history.json').read_text())
    old_checkpoint = torch.load(original/'best_head.pt', map_location='cpu', weights_only=True)
    require(np.array_equal(mean, old_checkpoint['mean'].numpy()) and
            np.array_equal(std, old_checkpoint['std'].numpy()), 'Normalization differs from original')
    cfg = dict(old_cfg, run_id=entry['run_id'], original_run_id=entry['original_run_id'],
               role='exploratory_stopping_selection_sensitivity', max_epochs=50, patience=None,
               early_stopping=False, selection='fixed final epoch 50; validation never selects weights',
               selected_epoch=50, original_runtime_binding_sha256=old_cfg['runtime_binding_sha256'],
               runtime_binding_sha256=binding_sha, trainer_sha256=sha(__file__),
               declared_after_original_validation_results=True,
               optimizer_updates=50*math.ceil(len(train_x)/128))
    (output/'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    model, optimizer = heads.new_head(entry['downstream_seed'], dimension=5120)
    history = []
    prefix_weights_match = False
    started = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        total = 0.
        for ids in torch.randperm(len(train_x)).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(train_x[ids]), train_y[ids])
            require(bool(torch.isfinite(loss)), 'Nonfinite classifier loss')
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                    'Nonfinite/missing classifier gradient')
            optimizer.step()
            total += loss.item()*len(ids)
        model.eval()
        pred = heads.predict(model, val_x)
        score = official.compute_metrics(val_y, pred)['macro_f1']
        history.append(dict(epoch=epoch, train_loss=total/len(train_x), val_macro_f1=score))
        if epoch <= len(old_history):
            require(history[-1] == old_history[epoch-1], f'Original history prefix differs at epoch {epoch}')
        if epoch == old_checkpoint['epoch']:
            prefix_weights_match = all(torch.equal(value, old_checkpoint['model'][key])
                                       for key, value in model.state_dict().items())
            require(prefix_weights_match, 'Original selected head not reproduced exactly')
        write_json(output/'history.json', history)
        print(json.dumps(dict(stage='fixed50_train', run_id=entry['run_id'], **history[-1])), flush=True)
    require(prefix_weights_match and len(history)==50, 'Incomplete fixed budget or replay')
    checkpoint = dict(model=model.state_dict(), mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                      epoch=50, config=cfg)
    torch.save(checkpoint, output/'final_head_epoch50.pt')
    # Evaluation reloads only the predetermined final checkpoint.
    model.load_state_dict(torch.load(output/'final_head_epoch50.pt', map_location='cpu', weights_only=True)['model'])
    probs = heads.predict(model, val_x, True)
    require(np.isfinite(probs).all(), 'Nonfinite final probabilities')
    require(len(val_rows)==len(probs)==len(val_y), 'Prediction metadata coverage differs')
    predictions = []
    for row, probability in zip(val_rows, probs):
        result = {k:row[k] for k in ('sample_id','subject_id','record_id','epoch_index','y_true')}
        result['y_pred'] = int(probability.argmax())
        result.update(zip(official.PROB_COLS, map(float, probability)))
        predictions.append(result)
    write_csv(output/'predictions_val.csv', predictions)
    frame = official.load_predictions(output/'predictions_val.csv')
    metrics = official.evaluate(output/'predictions_val.csv', n_boot=1000, seed=0)
    write_json(output/'metrics.json', metrics)
    official.per_subject_table(frame).to_csv(output/'per_subject_metrics.csv', index=False)
    write_csv(output/'confusion_matrix.csv', [dict(true_stage=stage, **dict(zip(official.STAGE_NAMES, counts)))
              for stage, counts in zip(official.STAGE_NAMES, metrics['confusion_matrix'])])
    elapsed = time.perf_counter()-started
    replay = dict(status='passed', original_run_id=entry['original_run_id'],
                  original_history_epochs=len(old_history), all_original_history_rows_exact=True,
                  original_selected_epoch=old_checkpoint['epoch'], original_selected_model_tensors_exact=True,
                  original_normalization_exact=True, new_selected_epoch=50)
    write_json(output/'prefix_replay_audit.json', replay)
    result = dict(status='complete', run_id=entry['run_id'], original_run_id=entry['original_run_id'],
                  representation=entry['representation'], adaptation_seed=entry['adaptation_seed'],
                  source_key=entry['source_key'], budget=entry['budget'], seed=entry['downstream_seed'],
                  epochs=50, selected_epoch=50, val_macro_f1=metrics['macro_f1'], n1_f1=metrics['n1_f1'],
                  n1_precision=metrics['n1_precision'], n1_recall=metrics['n1_recall'],
                  rem_f1=metrics['per_class_f1']['REM'], accuracy=metrics['accuracy'], seconds=elapsed,
                  original_selected_epoch=old_checkpoint['epoch'], original_trained_epochs=len(old_history),
                  original_history_and_selected_weights_reproduced=True, test_used=False)
    write_csv(output/'resource_profile.csv', [dict(device='cpu',threads=4,seconds=elapsed,epochs=50,
              optimizer_updates=cfg['optimizer_updates'],train_samples=len(train_x),val_samples=len(val_x),test_used=False)])
    (output/'run_summary.md').write_text(f"# {entry['run_id']}\n\nExploratory fixed 50-epoch final head. "
        f"Macro-F1 {metrics['macro_f1']:.8f}; N1 F1 {metrics['n1_f1']:.8f}. "
        "Original optimization prefix reproduced exactly. No validation selection or test use.\n",encoding='utf-8')
    write_json(output/'summary.json', result)
    write_json(output/'artifact_sha256.json', {p.name:sha(p) for p in output.iterdir() if p.is_file()})
    return result


def prepare():
    REPORT.mkdir(exist_ok=False)
    entries = plan()
    require(len(entries)==40 and len({e['run_id'] for e in entries})==40, 'Invalid run matrix')
    require(all(not (ROOT/'runs'/e['run_id']).exists() for e in entries), 'Run would overwrite existing output')
    config = dict(experiment='exploratory_fixed50_stopping_selection_sensitivity_v01', registered_at=now(),
        preregistered_before_original_results=False, locked_before_any_new_training=True,
        original_experiment='W2_primary_fusion_localEOG_v01', planned_heads=40,
        budgets=['label10','label25'],adaptation_seeds=[0,1,2],downstream_seeds=list(range(5)),
        classifier=dict(name='Linear(5120,5)',optimizer='AdamW',learning_rate=.001,weight_decay=.0001,
                        batch_size=128,epochs=50,loss='unweighted_cross_entropy',early_stopping=False,
                        selection='final_epoch_50',normalization='current labeled training subjects only'),
        unchanged=['features','data split and label budget','initialization seed','minibatch order',
                   'optimizer and loss','per-feature normalization','official B evaluator'],
        prefix_exact_replay_required=True, test_used=False, D_full_cache_received=False,
        EOG_source='local_regeneration_using_identical_D_reference_configuration',
        team_acceptance_complete=False, validation_metrics_logged_but_never_used_for_selection=True,
        interpretation='Sensitivity to stopping AND checkpoint selection jointly; not independent test evidence or attribution to early stopping alone')
    write_json(REPORT/'config.json', config)
    write_csv(REPORT/'run_plan.csv', entries)
    (REPORT/'实验方案.md').write_text('''# 固定50轮：停止与选模规则敏感性补充实验

本实验在看过第二周结果后提出，属于探索性补充，不替换原预登记结果。40个新双模态分类头：S0及三个S1、label10/25、下游seed0–4。所有头固定训练50轮，唯一评价权重为第50轮，验证曲线仅记录。训练名单、归一化、初始化、批次顺序、优化器、普通交叉熵与原实验相同；不增加类别加权或其他方法。

先核验5份既有池化缓存SHA、元数据对齐和原40份run归档。每个新run必须精确复现原实验全部已训练轮次的loss/Macro-F1，并在原选中轮次复现其权重张量和归一化参数。任何不一致立即失败，不继续解释为停止规则效应。

主分析比较原早停/最佳轮选模与固定50轮/最后轮选模下的S1−S0配对差，报告两者的差值及每个主干/种子；S0只训练10个，不重复计数。最差原格单独追踪，全部其他格保留。规则同时改变停止时间与最终检查点选择，不能将影响仅归因于其中一项。

复用同一份本地按D协议重建的原始冻结EOG，D原文件未接收；无需等待其他成员。适配主干不再训练，原始特征不再提取，测试集保持封存。原适配配置、锁、历史、结果及GitHub档案不修改。脚本 run_fixed50_fusion.py，运行状态 summary.json，结束后自动生成比较表与完成报告。
''', encoding='utf-8')
    write_json(REPORT/'summary.json',dict(status='prepared_not_started',stage='tests_and_binding_pending',
               completed_heads=0,expected_heads=40,test_used=False,prepared_at=now()))
    return config


def bind():
    require(not (REPORT/'runtime_binding.json').exists(), 'Do not overwrite binding')
    cfg=json.loads((REPORT/'config.json').read_text())
    require(cfg['classifier']['epochs']==50 and not cfg['classifier']['early_stopping'], 'Budget drift')
    tests=(REPORT/'runtime_tests.log').read_text()
    require('\nOK\n' in tests, 'Fixed-budget tests did not pass')
    files=[Path(__file__),ROOT/'tests/test_fixed50_fusion.py',REPORT/'runtime_tests.log',REPORT/'config.json',
           REPORT/'run_plan.csv',REPORT/'实验方案.md',Path(heads.__file__),Path(prior.__file__),
           ROOT/'scripts/train_primary_fusion_head.py',ROOT/'scripts/replication_feature_head_runtime.py',
           ROOT/'scripts/validate_adapt_lock.py',heads.SUBSETS,heads.SPLIT,heads.EVALUATOR,
           prior.REPORT/'runtime_binding.json',prior.REPORT/'fusion_config.json']
    for e in plan():
        run=ROOT/'runs'/e['original_run_id']
        hashes=json.loads((run/'artifact_sha256.json').read_text())
        require(all(sha(run/name)==expected for name,expected in hashes.items()),'Original run artifact changed')
        files += [run/name for name in hashes] + [run/'artifact_sha256.json',run/'summary.json']
    fusion=json.loads((prior.REPORT/'fusion_config.json').read_text())
    for source in fusion['EEG_sources']:
        files += [ROOT/source['audit'],ROOT/source['feature_root']/'feature_index.csv',ROOT/source['feature_root']/'feature_config.json']
    files += [prior.REPORT/'execution/EOG_audit/S0_feature_audit.json',
              ROOT/fusion['EOG_root']/'feature_index.csv',ROOT/fusion['EOG_root']/'feature_config.json']
    binding=dict(status='bound_before_new_training',bound_at=now(),expected_new_heads=40,
                 input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},test_used=False)
    write_json(REPORT/'runtime_binding.json',binding)
    return binding


def check_binding():
    binding=json.loads((REPORT/'runtime_binding.json').read_text())
    for name,expected in binding['input_sha256'].items():
        require(sha(ROOT/name)==expected,'Bound input changed: '+name)


def verify_pools():
    cfg=json.loads((prior.REPORT/'fusion_config.json').read_text())
    canonical=heads.read_csv(heads.SOURCES['S0']/'feature_index.csv')
    heads.audit_index(canonical,heads.read_csv(heads.SPLIT))
    checks=[]; paths={}
    sources=cfg['EEG_sources']+[dict(key='EOG',feature_root=cfg['EOG_root'],
        audit='reports/week2_fusion_20261029/execution/EOG_audit/S0_feature_audit.json',
        pooled='reports/week2_fusion_20261029/execution/EOG_audit/S0_pooled.npy')]
    for source in sources:
        audit=json.loads((ROOT/source['audit']).read_text())
        require(audit['status']=='passed' and audit['raw_all_finite'] and audit['samples']==154124,'Source audit failed')
        folder=ROOT/source['feature_root'];p=ROOT/source['pooled']
        require(sha(folder/'feature_index.csv')==audit['source_index_sha256'] and
                sha(folder/'feature_config.json')==audit['source_config_sha256'],'Feature metadata changed')
        heads.align_indices(canonical,heads.read_csv(folder/'feature_index.csv'))
        require(sha(p)==audit['pooled_sha256'],'Pooled feature SHA changed')
        values=np.load(p,mmap_mode='r',allow_pickle=False)
        require(values.shape==(154124,2560) and values.dtype==np.float32,'Invalid pool shape/dtype')
        del values
        paths[source['key']]=p
        checks.append(dict(source=source['key'],path=source['pooled'],sha256=audit['pooled_sha256'],
                           prior_full_finite_audit_verified_by_cache_SHA=True))
        print(json.dumps(dict(stage='verify_existing_pool',source=source['key'])),flush=True)
    write_json(REPORT/'pooled_reuse_audit.json',dict(status='passed',sources=checks))
    return canonical,paths


def analyze(results,official):
    pairs=[]
    for budget in ['label10','label25']:
        for adaptation in range(3):
            for seed in range(5):
                by_rep={r['representation']:r for r in results if r['budget']==budget and r['seed']==seed
                        and (r['representation']=='S0' or r['adaptation_seed']==adaptation)}
                base,adapted=by_rep['S0'],by_rep['S1']
                original={rep:json.loads((ROOT/'runs'/r['original_run_id']/'metrics.json').read_text()) for rep,r in by_rep.items()}
                row=dict(budget=budget,adaptation_seed=adaptation,downstream_seed=seed,
                         S0_run_id=base['run_id'],S1_run_id=adapted['run_id'])
                for newkey,oldkey in [('val_macro_f1','macro_f1'),('n1_f1','n1_f1'),('rem_f1','REM')]:
                    a=original['S1']['per_class_f1']['REM'] if oldkey=='REM' else original['S1'][oldkey]
                    b=original['S0']['per_class_f1']['REM'] if oldkey=='REM' else original['S0'][oldkey]
                    label='macro_f1' if newkey=='val_macro_f1' else newkey
                    row['earlystop_delta_'+label+'_pp']=100*(a-b)
                    row['fixed50_delta_'+label+'_pp']=100*(adapted[newkey]-base[newkey])
                    row['change_in_delta_'+label+'_pp']=row['fixed50_delta_'+label+'_pp']-row['earlystop_delta_'+label+'_pp']
                pairs.append(row)
    write_csv(REPORT/'paired_stopping_sensitivity.csv',pairs)
    summary=[]
    for budget in ['label10','label25']:
        group=[r for r in results if r['budget']==budget]
        for key,oldkey in [('val_macro_f1','macro_f1'),('n1_f1','n1_f1'),('rem_f1','REM')]:
            means={rep:float(np.mean([r[key] for r in group if r['representation']==rep])) for rep in ['S0','S1']}
            label='macro_f1' if key=='val_macro_f1' else key
            old_delta=float(np.mean([p['earlystop_delta_'+label+'_pp'] for p in pairs if p['budget']==budget]))
            delta=100*(means['S1']-means['S0'])
            summary.append(dict(budget=budget,metric=label,S0_fixed50_mean=means['S0'],S1_fixed50_mean=means['S1'],
                fixed50_delta_pp=delta,earlystop_delta_pp=old_delta,change_in_delta_pp=delta-old_delta))
    write_csv(REPORT/'overall_stopping_sensitivity.csv',summary)
    for adaptation in range(3):
        out=REPORT/f'comparison_aseed{adaptation}';out.mkdir()
        heads.paired_report([r for r in results if r['representation']=='S0' or r['adaptation_seed']==adaptation],out,official)
    audited=[]
    for r in results:
        folder=ROOT/'runs'/r['run_id']
        hashes=json.loads((folder/'artifact_sha256.json').read_text())
        require(all(sha(folder/name)==digest for name,digest in hashes.items()),'Final artifact SHA differs')
        ckpt=torch.load(folder/'final_head_epoch50.pt',map_location='cpu',weights_only=True)
        hist=json.loads((folder/'history.json').read_text())
        require(ckpt['epoch']==50 and len(hist)==50 and not ckpt['config']['early_stopping'],'Final budget mismatch')
        require(json.loads((folder/'prefix_replay_audit.json').read_text())['status']=='passed','Replay failed')
        measured=official.compute_metrics(*official.load_predictions(folder/'predictions_val.csv')[['y_true','y_pred']].to_numpy().T)
        stored=json.loads((folder/'metrics.json').read_text())
        require(all(stored[k]==v for k,v in measured.items()),'Recomputed final point metrics differ')
        audited.append(dict(run_id=r['run_id'],epochs=50,selected_epoch=50,artifact_files_verified=len(hashes),prefix_exact=True))
    check_binding()
    write_json(REPORT/'completion_audit.json',dict(status='passed',runs=40,checks=audited,test_used=False))
    worst=next(p for p in pairs if p['budget']=='label10' and p['adaptation_seed']==0 and p['downstream_seed']==0)
    lines=['# 固定50轮停止/选模敏感性：完成报告','',f'完成于 {now()}。40/40个新头全部固定训练50轮并取最后一轮，原训练前缀与原选中权重精确复现。',
           '', '| 预算 | 指标 | 原早停 S1−S0（百分点） | 固定50轮 S1−S0（百分点） | 差值变化 |',
           '| --- | --- | ---: | ---: | ---: |']
    for row in summary:
        lines.append(f"| {row['budget']} | {row['metric']} | {row['earlystop_delta_pp']:+.3f} | {row['fixed50_delta_pp']:+.3f} | {row['change_in_delta_pp']:+.3f} |")
    lines += ['',f"原最差预定格（label10/适配seed0/分类seed0）：Macro-F1配对差 {worst['earlystop_delta_macro_f1_pp']:+.3f} → {worst['fixed50_delta_macro_f1_pp']:+.3f} 个百分点；N1 F1配对差 {worst['earlystop_delta_n1_f1_pp']:+.3f} → {worst['fixed50_delta_n1_f1_pp']:+.3f} 个百分点。",'',
        'S0每预算5个头；S1先按每个主干求5种子均值再对3主干等权平均。30组配对全部保留；15个S1分类头不是15个独立适配。逐格结果见paired_stopping_sensitivity.csv，逐主干汇总与按8人配对bootstrap见comparison_aseed*/。', '',
        '本实验同时改变早停和最佳轮选模两项规则，衡量它们的联合敏感性；无法单独证明早停是唯一原因。只在相同8名开发验证者上探索，无独立测试证据，不能事后挑选更好的规则替换原结果。固定50轮可能过拟合，完整曲线保留。', '',
        'EOG仍为本地按D协议重建，D原始缓存未接收/字节等价未验证；B/D团队验收状态未改变。原40份早停结果及所有适配锁不修改，未启动类别加权或新主干适配。']
    (REPORT/'固定50轮_完成报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    check_binding()
    require(not (REPORT/'run_registry.csv').exists(),'Existing execution cannot be overwritten')
    with (REPORT/'.fixed50.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(status='running',stage='verify_existing_pools',pid=os.getpid(),started_at=now(),
                    completed_heads=0,expected_heads=40,test_used=False)
        write_json(REPORT/'summary.json',status)
        try:
            canonical,paths=verify_pools()
            torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
            official=heads.official_module();subsets=heads.read_csv(heads.SUBSETS);results=[]
            eog=np.load(paths['EOG'],mmap_mode='r',allow_pickle=False)
            for entry in plan():
                selected,train_ids,val_ids=heads.selected_positions(canonical,subsets,entry['budget'])
                eeg=np.load(paths[entry['source_key']],mmap_mode='r',allow_pickle=False)
                train=prior.concatenate(eeg[train_ids],eog[train_ids]);val=prior.concatenate(eeg[val_ids],eog[val_ids])
                mean,std=heads.fit_normalization(train);train-=mean;train/=std;val-=mean;val/=std
                require(np.isfinite(train).all() and np.isfinite(val).all(),'Invalid normalized features')
                train_x,val_x=torch.from_numpy(train),torch.from_numpy(val)
                train_y=torch.tensor([int(canonical[i]['y_true']) for i in train_ids],dtype=torch.int64)
                val_rows=[canonical[i] for i in val_ids];val_y=np.array([int(r['y_true']) for r in val_rows])
                require(len(val_rows)==21186 and len({r['subject_id'] for r in val_rows})==8,'Validation coverage differs')
                status.update(stage='fixed50_head_training',current_run=entry['run_id']);write_json(REPORT/'summary.json',status)
                result=train_final50(entry,train_x,train_y,val_x,val_y,mean,std,val_rows,official,
                    ROOT/'runs'/entry['original_run_id'],ROOT/'runs'/entry['run_id'],sha(REPORT/'runtime_binding.json'))
                results.append(result);write_csv(REPORT/'run_registry.csv',results)
                status.update(completed_heads=len(results));write_json(REPORT/'summary.json',status)
                del eeg,train,val,train_x,val_x,train_y
            status.update(stage='matched_stopping_sensitivity_analysis',current_run=None);write_json(REPORT/'summary.json',status)
            analyze(results,official)
            status.update(status='complete',stage='fixed50_matrix_and_analysis_complete',completed_at=now(),
                          original_prefix_exact_runs=40,selected_epoch_all_runs=50)
        except BaseException:
            status.update(status='failed',error=traceback.format_exc(),failed_at=now())
            raise
        finally:
            write_json(REPORT/'summary.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','bind','run'])
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='bind':
        binding=bind()
        print(json.dumps(dict(status=binding['status'],bound_input_files=len(binding['input_sha256']),expected_new_heads=40)))
    else:run()


if __name__=='__main__':main()
