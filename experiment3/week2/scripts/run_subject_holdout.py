"""Three subject-disjoint outer folds; refit diffusion adaptation in each fold.

The original train54 is the complete universe. Original dev8/test16 are excluded.
Each of nine new EMA backbones starts from S0, then uses only its outer train36.
All outer predictions remain embargoed until the 162 inner and 72 final fits pass.
"""
import argparse
from collections import defaultdict
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from threadpoolctl import threadpool_limits

import adapt_preflight as ref
import train_sleep_adaptation as adapt
import run_eeg_multisubset_v02 as linear

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/subject_holdout_20261101_v01'
PARENT = ROOT/'configs/adapt_W2_C_S1_pilot_v01.yaml'
INDEX = ROOT/'features/eeg_cond14_v01/feature_index.csv'
S0 = ROOT/'reports/week2_eeg_comparison_20261028/execution/S0_pooled.npy'
S0_AUDIT = S0.parent/'S0_feature_audit.json'
SPLIT = ROOT/'vendor/experiment1-review/manifests/split_subjects.csv'
OUTER_SEED = 5101
SEEDS = [0, 1, 2]
ROSTERS = [2024, 4101, 4102]
BUDGETS = {'n4': 4, 'n9': 9}
require, sha, read_csv = linear.require, linear.sha, linear.read_csv
write_json, write_csv, now = linear.write_json, linear.write_csv, linear.now


def outer_assignment(subjects):
    subjects = sorted(subjects)
    require(len(subjects) == len(set(subjects)) == 54, 'Need original train54')
    order = np.random.Generator(np.random.PCG64(OUTER_SEED)).permutation(subjects)
    return {str(s): i % 3 for i, s in enumerate(order)}


def fold_people(assignment, fold):
    require(fold in range(3), 'Invalid outer fold')
    train = sorted(s for s, f in assignment.items() if f != fold)
    held = sorted(s for s, f in assignment.items() if f == fold)
    require(len(train) == 36 and len(held) == 18 and set(train).isdisjoint(held), 'Outer leakage')
    return train, held


def roster_orders(train, original_orders):
    allowed = set(train)
    orders = {seed: [s for s in original_orders[seed] if s in allowed] for seed in ROSTERS}
    require(all(len(v) == len(set(v)) == 36 and set(v) == allowed for v in orders.values()), 'Invalid fold roster')
    return orders


def plan():
    return [dict(run_id=f'CX_SH_f{f}_{source}_{budget}_r{roster}_v01', fold=f,
                 source_key=source, adaptation_seed=None if source == 'S0' else int(source[-1]),
                 roster_seed=roster, budget=budget, labeled_subjects=count)
            for f in range(3) for roster in ROSTERS for budget, count in BUDGETS.items()
            for source in ['S0', 'S1_aseed0', 'S1_aseed1', 'S1_aseed2']]


def inputs():
    rows = read_csv(INDEX)
    splits = read_csv(SPLIT)
    train, _ = ref.validate_split(splits)
    assignment = outer_assignment([r['subject_id'] for r in train])
    selected = [dict(r, source_pool_position=i) for i, r in enumerate(rows) if r['split'] == 'train']
    require(len(selected) == 132938 and {r['subject_id'] for r in selected} == set(assignment), 'Train index mismatch')
    require(len({r['sample_id'] for r in selected}) == len(selected), 'Duplicate epoch')
    return selected, train, assignment


def prepare():
    require(not REPORT.exists(), 'Create a new version instead of overwriting')
    rows, train, assignment = inputs()
    original = linear.make_orders(sorted(assignment), read_csv(linear.REFERENCE))
    profiles, rosters, inner = [], [], []
    for fold in range(3):
        training, held = fold_people(assignment, fold)
        orders = roster_orders(training, original)
        for seed, order in orders.items():
            rosters.extend(dict(fold=fold, roster_seed=seed, subject_id=s, sampling_rank=i+1,
                                n4=int(i < 4), n9=int(i < 9)) for i, s in enumerate(order))
            for budget, size in BUDGETS.items():
                labeled = order[:size]; mapping = linear.inner_folds(labeled)
                selected = [r for r in rows if r['subject_id'] in labeled]
                for f in range(3):
                    y = [int(r['y_true']) for r in selected if mapping[r['subject_id']] != f]
                    require((np.bincount(y, minlength=5) > 0).all(), 'Preselected inner train lacks a class; no redraw')
                inner.extend(dict(fold=fold, roster_seed=seed, budget=budget, subject_id=s,
                                  validation_fold=mapping[s]) for s in labeled)
                counts = np.bincount([int(r['y_true']) for r in selected], minlength=5)
                profiles.append(dict(fold=fold, roster_seed=seed, budget=budget, labeled_subjects=size,
                                     fraction_of_train36=size/36, epochs=len(selected),
                                     **dict(zip(['W','N1','N2','N3','REM'], counts.tolist()))))
    REPORT.mkdir()
    write_csv(REPORT/'outer_subjects.csv', [dict(subject_id=s, heldout_fold=f) for s, f in sorted(assignment.items())])
    write_csv(REPORT/'label_subsets.csv', rosters)
    write_csv(REPORT/'inner_subjects.csv', inner)
    write_csv(REPORT/'budget_profiles.csv', profiles)
    write_csv(REPORT/'run_plan.csv', plan())
    cfg = dict(version='subject_holdout_20261101_v01', registered_at=now(), outer_seed=OUTER_SEED,
        outer_assignment='PCG64 permutation of sorted original train54, rank modulo3; no label stratification',
        outer_folds=3, train_subjects_per_fold=36, heldout_subjects_per_fold=18,
        adaptation_seeds=SEEDS, roster_seeds=ROSTERS, roster_policy='filter prior fixed54 orders to outer train36',
        absolute_label_budgets=BUDGETS, actual_label_fractions=[4/36,9/36],
        new_adaptations=9, steps_per_adaptation=10000, parent_numerical_config=PARENT.relative_to(ROOT).as_posix(),
        adaptation_changes='only allowed subject pool, seed, output paths; all other numerical settings inherited',
        selection='new EMA at10000; no outer/inner score selects backbone', trainable_parameters=527233,
        classifier='float64 converged logistic, mean CE+lambda/2*W_squared, unregularized bias',
        lambda_grid=linear.LAMBDAS, lambda_selection='S0-only 3fold subject OOF within current labeled outer training subjects',
        one_lambda_shared_with_three_S1=True, tie_rule='larger lambda', gradient_inf_tolerance=linear.GTOL,
        solver_passes=[dict(maxiter=10000,maxfun=25000,ftol=1e-15),dict(maxiter=10000,maxfun=25000,ftol=0.)],
        inner_fits=162, final_models=72, normalization='only current inner/final labeled training people; float64; std_floor1e-6',
        EEG_only=True, EOG_read=False, original_dev8_read=False, original_test16_read=False,
        features='t1 gate std correction1 20layers; FP32 forward -> FP16 rounding -> FP32 time-mean2560',
        extraction_batch_size=4, extraction_subjects='original train54, frozen inference only; outer heldout excluded from learning',
        outer_prediction_embargo='all9 adaptations,9 caches,162 inner fits,72 final models complete before outer predictions',
        primary_metric='pooled out-of-fold Macro-F1 delta S1-S0, by roster and adaptation seed',
        useful_effect_reference_pp=.5, reference_threshold_is_not_clinical_standard=True,
        uncertainty='subject-paired bootstrap54,1000 resamples; shared models/folds imply residual dependence',
        interpretation='new foldwise local adaptation exclusion; previously used development corpus, not independent external data; S0 pretraining overlap unaudited',
        no_hyperparameter_changes_based_on_outer_results=True)
    write_json(REPORT/'config.json', cfg)
    (REPORT/'实验方案.md').write_text('''# 按受试者留出验证：执行锁定方案

原54名训练者随机按人分3折，每折36人适配、18人留出；一个人的所有夜晚始终同折。PCG64 seed5101固定排序模3，不参考标签或前轮分数。原8名开发验证者及16名封存测试者完全排除。

每折从原S0独立启动适配seed0/1/2，共9份新EMA，末2层及out_layer共527233参数、10000步、batch2、lr1e-5、无标签v_prediction、EMA0.999均沿用已验证数值协议。旧S1看过全部54人，绝不复用其权重或特征。初始化器只构造过原54人的未读取provider，新程序在任何next/read前用36人白名单provider替换；白名单reader会拒绝外层留出者，训练只允许signal.npy和epoch_index.npy。保留实际访问日志并核查全部36人覆盖。每次诊断和正式训练均重新初始化权重、优化器、EMA与随机状态。

标注者排序采用前轮2024/4101/4102的54人固定顺序过滤本折训练者，前4/9人嵌套；相对36人为11.11%/25%，不将4人冒称精确10%，也不沿用旧5/14人的百分比。内层3折按标注者排序模3，仅用S0 OOF Macro-F1选择lambda0.01/0.001/0.0001，同分取强正则；同一折/名单/预算的4个表征共用lambda。S0内层选参不使用S1或外层标签。逐折/最终标准化仅拟合各自有标签训练者。

54人S0缓存复用前校验SHA。S1只重提这54人，冻结eval前向时才允许读取留出者信号，模型不接收标签；FP32特征经FP16舍入后转FP32按30个时间窗均值池化，与旧缓存数值流程一致；直接保存2560维池化，避免逐窗大缓存。每份选定EMA新建kernel缓存并通过缓存前后等值探针。

9份主干和缓存完成后做162次训练内选参拟合与72个最终模型。梯度无穷范数≤1e-5，最多同目标2遍各10000步，不因外层结果改名单、步数、lambda或读出方式。所有拟合完成才统一放开外层预测。每个名单×适配seed×预算合并3折OOF（54人各预测一次），用相同折S0作配对；同时报告逐折、逐人差值、N1精度/召回/F1与1000次按人配对bootstrap，不把共享折/主干/名单当完全独立重复。预设+0.5个百分点Macro-F1为参考有用幅度，非临床阈值。保留全部负向与未达收敛结果。

该数据此前参与过开发；这是新的按折隔离实验，并非全新外部验证。初始S0的预训练数据重叠未审计，结论限于本次本地适配阶段排除了留出者。按人bootstrap未完全刻画共享模型的训练不确定性。分折10000步访问预算相同，但训练池从54变36，不能把差异全归结为划分。

硬件中断仅允许从本版本哈希匹配的最近恢复点恢复；非有限值或来源/隔离失败停止并保存，不自动调参。本轮无GitHub上传或对其他成员发送消息。
''', encoding='utf-8')
    write_json(REPORT/'summary.json', dict(status='prepared', stage='unit_tests_and_preflight_pending',
        completed_adaptations=0, expected_adaptations=9, completed_caches=0, expected_caches=9,
        completed_cv_fits=0, expected_cv_fits=162, completed_final_models=0, expected_final_models=72,
        outer_evaluations=0, test_used=False))


def bind():
    require(not (REPORT/'runtime_binding.json').exists(), 'Cannot replace binding')
    tests=(REPORT/'runtime_tests.log').read_text()
    require(tests.count('\nOK\n')==2 and 'Ran 12 tests' in tests and 'Ran 9 tests' in tests, 'Unit tests missing')
    rows, train, assignment = inputs()
    records = sorted({r['record_id'] for r in rows})
    # Hash opaque NPZ bytes; no label array is loaded by this operation.
    data = {r:sha(ref.DATA/f'{r}.npz') for r in records}
    write_json(REPORT/'source_signal_sha256.json', data)
    files = [Path(__file__), ROOT/'tests/test_subject_holdout.py', Path(linear.__file__),
        ROOT/'tests/test_eeg_multisubset_v02.py', Path(adapt.__file__), Path(ref.__file__),
        ROOT/'scripts/validate_adapt_lock.py', ROOT/'scripts/extract_features.py',
        ROOT/'scripts/extract_adapted_features.py', PARENT, PARENT.with_suffix('.lock.json'),
        INDEX, S0_AUDIT, SPLIT, linear.REFERENCE, linear.EVALUATOR,
        ROOT/'features/eeg_cond14_v01/feature_config.json']
    files += [REPORT/name for name in ['config.json','outer_subjects.csv','label_subsets.csv',
        'inner_subjects.csv','budget_profiles.csv','run_plan.csv','实验方案.md','runtime_tests.log','source_signal_sha256.json','launch_pipeline.py']]
    lock = json.loads(PARENT.with_suffix('.lock.json').read_text())
    files += [ROOT/p for p in lock['input_sha256']]
    files += [ROOT/'vendor/EEGDM/model/classifier.py', ROOT/'vendor/EEGDM/src/util.py']
    write_json(REPORT/'runtime_binding.json', dict(bound_at=now(), status='bound_before_diagnostic_or_formal_updates',
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files}, test_used=False))


def check_binding():
    for p, value in json.loads((REPORT/'runtime_binding.json').read_text())['input_sha256'].items():
        require(sha(ROOT/p) == value, 'Bound input changed: '+p)


class FoldSignalReader(ref.SignalOnlyReader):
    def __init__(self, directory, record_split, allowed_records, expected_sha):
        super().__init__(directory, record_split)
        self.allowed_records = frozenset(allowed_records)
        self.expected_sha = expected_sha
        self.verified_records = set()

    def read_member(self, record, member):
        require(record in self.allowed_records, 'Outer heldout/unknown record forbidden during adaptation')
        require(member in ref.ALLOWED_MEMBERS, 'Label member forbidden')
        if record not in self.verified_records:
            require(sha(self.directory/f'{record}.npz') == self.expected_sha[record], 'Signal source changed')
            self.verified_records.add(record)
        return super().read_member(record, member)


def restrict_provider(original, training, directory, record_split, expected_sha, seed):
    require(not original.reader.accesses and not original.examples, 'Initializer already read signals')
    records = {r for row in training for r in row['record_ids'].split(';')}
    require(len(training) == 36 and len({r['subject_id'] for r in training}) == 36, 'Need exactly36 adaptation people')
    reader = FoldSignalReader(directory, record_split, records, expected_sha)
    return ref.TrainingSignalBatches(reader, training, 2, seed)


def initialize_fold(fold, seed):
    require(seed in SEEDS, 'Invalid adaptation seed')
    cfg = yaml.safe_load(PARENT.read_text())
    cfg = copy.deepcopy(cfg)
    cfg['sampling'].update(adaptation_seed=seed, data_seed=seed, noise_seed=seed)
    # Reuse the original exact model/optimizer/EMA constructor, which does ZERO data reads.
    runtime = adapt.initialize(cfg, dict(config_sha256=sha(PARENT)))
    model, optimizer, ema, scheduler, original, parameters, initial = runtime
    rows, people, assignment = inputs()
    train, held = fold_people(assignment, fold)
    selected = [r for r in people if r['subject_id'] in train]
    _, partitions = ref.validate_split(read_csv(SPLIT))
    provider = restrict_provider(original, selected, ref.DATA, partitions,
        json.loads((REPORT/'source_signal_sha256.json').read_text()), seed)
    del original
    cfg['data'].update(n_subjects=36,n_records=len(provider.reader.allowed_records),
        actual_allowed_subjects=train,actual_allowed_records=sorted(provider.reader.allowed_records))
    cfg['run_id']=f'CX_SH_fold{fold}_aseed{seed}_steps10000_v01'
    initial.update(outer_fold=fold, adaptation_seed=seed, adaptation_subjects=train,
                   outer_heldout_subjects=held, initializer_data_reads=0, original_provider_replaced_before_any_read=True,
                   initialization_contract_sha256=sha(PARENT),study_config_sha256=sha(REPORT/'config.json'))
    return cfg, (model, optimizer, ema, scheduler, provider, parameters, initial)


def adaptation_step(runtime, cfg, step):
    model, optimizer, ema, scheduler, provider, parameters, initial = runtime
    start = time.perf_counter(); batch = provider.next(step)
    data_seconds = time.perf_counter()-start
    clean = torch.from_numpy(batch).cuda()
    noise = torch.randn_like(clean)
    times = torch.randint(0, cfg['diffusion']['num_train_timesteps'], (len(clean),), device='cuda')
    cond = torch.full((len(clean),1), 14, dtype=torch.long, device='cuda')
    target = ref.diffusion_target(scheduler, cfg['diffusion']['prediction_type'], clean, noise, times)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(scheduler.add_noise(clean,noise,times), times[:,None], cond)
    require(prediction.shape == target.shape, 'Target shape mismatch')
    loss = F.mse_loss(prediction,target)
    require(bool(torch.isfinite(loss)), 'Nonfinite diffusion loss')
    loss.backward()
    require(all(p.grad is not None for _,p in parameters), 'Missing trainable gradient')
    norm = torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,error_if_nonfinite=True)
    optimizer.step(); ema.update(model)
    require(ema.updates == step, 'EMA count mismatch')
    adapt.assert_trainable_finite(parameters,ema)
    torch.cuda.synchronize()
    return dict(step=step, loss=float(loss.detach()), gradient_norm_before_clip=float(norm),
                data_seconds=data_seconds, step_seconds=time.perf_counter()-start, ema_updates=ema.updates)


def adaptation_run(fold, seed, resume=False, diagnostic=None):
    check_binding()
    cfg, runtime = initialize_fold(fold,seed)
    model, optimizer, ema, scheduler, provider, parameters, initial = runtime
    folder = REPORT/(diagnostic or f'adaptations/fold{fold}_seed{seed}')
    folder.mkdir(parents=True,exist_ok=resume)
    write_json(folder/'initialization.json',initial)
    numerical_keys=['initialization','data','input','sampling','model','diffusion','training','ema','environment','stopping_and_selection']
    actual={k:cfg[k] for k in numerical_keys}
    actual.update(run_id=cfg['run_id'],study_config_sha256=sha(REPORT/'config.json'),
        downstream_protocol='study config.json, converged logistic; original parent downstream section is not used',
        output_directory=folder.relative_to(ROOT).as_posix())
    write_json(folder/'actual_adaptation_config.json',actual)
    history=[];start_step=0;binding_sha=sha(REPORT/'runtime_binding.json')
    parent_sha=sha(PARENT)
    if resume and (folder/'latest_recovery.json').exists():
        receipt=json.loads((folder/'latest_recovery.json').read_text())
        require(sha(folder/'recovery.pt')==receipt['sha256'],'Recovery SHA mismatch')
        payload=torch.load(folder/'recovery.pt',map_location='cpu',weights_only=False)
        require(payload['fold']==fold and payload['seed']==seed,'Recovery fold/seed differs')
        adapt.restore_recovery(payload,model,optimizer,ema,provider,parent_sha,binding_sha)
        history=payload['history'];start_step=payload['completed_updates']
    steps=1 if diagnostic=='preflight_gradient' else 100 if diagnostic else 10000
    status=dict(status='running',fold=fold,seed=seed,pid=os.getpid(),completed_updates=start_step,
        target_updates=steps,started_at=now(),adaptation_subjects=initial['adaptation_subjects'],
        outer_heldout_subjects=initial['outer_heldout_subjects'],outer_heldout_signals_read=False,
        stage_labels_read=False,original_dev8_read=False,test_used=False)
    write_json(folder/'summary.json',status)
    before={n:p.detach().cpu().clone() for n,p in parameters} if diagnostic=='preflight_gradient' else None
    beginning=time.perf_counter()
    try:
        for step in range(start_step+1,steps+1):
            row=adaptation_step(runtime,cfg,step);history.append(row)
            if step==1 and before is not None:
                evidence=ref.parameter_evidence(model,before)
                write_csv(folder/'gradient_evidence.csv',evidence)
            if step==1 or step%100==0 or step==steps:
                adapt.assert_frozen(model,ema,cfg['model']['frozen_parameter_sha256'])
                require(all(r['record_id'] in provider.reader.allowed_records for r in provider.reader.accesses),'Forbidden access')
                status.update(completed_updates=step,last_loss=row['loss'],updated_at=now(),
                    mean_seconds_per_step=(time.perf_counter()-beginning)/(step-start_step),
                    sampled_subjects=len({r['subject_id'] for r in provider.examples}))
                write_json(folder/'summary.json',status)
                write_csv(folder/'history.csv',history)
                print(json.dumps(dict(stage='adaptation_progress',**{k:v for k,v in status.items() if k!='stage'})),flush=True)
            if step%1000==0 and not diagnostic:
                check_binding()
                payload=adapt.make_recovery(model,optimizer,ema,provider,step,history,parent_sha,binding_sha,0)
                payload.update(fold=fold,seed=seed)
                receipt=adapt.atomic_save(folder/'recovery.pt',payload)
                write_json(folder/'latest_recovery.json',receipt)
        if not diagnostic or steps>=36:
            require({r['subject_id'] for r in provider.examples}==set(initial['adaptation_subjects']),'Incomplete36-person coverage')
        write_csv(folder/'signal_batches.csv',provider.examples)
        write_csv(folder/'signal_access_audit.csv',provider.reader.accesses)
        if not diagnostic:
            metadata=dict(state_dict=adapt.cpu_tree(ema.state),completed_updates=10000,ema_updates=10000,
                weight_role='selected_new_ema',selected_by='fixed_step_10000_no_outer_validation',fold=fold,
                seed=seed,adaptation_subjects=initial['adaptation_subjects'],outer_heldout_subjects=initial['outer_heldout_subjects'],
                original_checkpoint_sha256=ref.CHECKPOINT_SHA,binding_sha256=binding_sha)
            receipt=adapt.atomic_save(folder/'selected_ema_step10000.pt',metadata)
            status.update(selected_weight=receipt,selected_parameter_sha256=adapt.parameter_digest(ema.state,[n for n,_ in model.named_parameters()]))
        status.update(status='complete',completed_at=now(),frozen_parameters_unchanged=True,
            optimizer_updates=steps,ema_updates=ema.updates,sample_occurrences=len(provider.examples))
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:write_json(folder/'summary.json',status)


def pool_like_original(raw):
    require(raw.ndim==4 and raw.shape[1:]==(20,30,128),'Raw shape mismatch')
    rounded=raw.astype(np.float16)
    require(np.isfinite(rounded).all(),'FP16 overflow')
    return rounded.astype(np.float32).mean(axis=2,dtype=np.float32).reshape(len(raw),2560)


def array_digest(values):
    digest=hashlib.sha256()
    for start in range(0,len(values),128):digest.update(np.ascontiguousarray(values[start:start+128]).tobytes())
    return digest.hexdigest()


def extraction_run(fold,seed,resume=False):
    check_binding()
    import extract_adapted_features as extraction
    source=REPORT/f'adaptations/fold{fold}_seed{seed}'
    status=json.loads((source/'summary.json').read_text())
    require(status['status']=='complete' and status['completed_updates']==status['ema_updates']==10000,'Incomplete adaptation')
    weight=source/status['selected_weight']['path']
    require(sha(weight)==status['selected_weight']['sha256'],'Selected EMA checksum mismatch')
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    rows,people,assignment=inputs();training,held=fold_people(assignment,fold)
    require(checkpoint['adaptation_subjects']==training and checkpoint['outer_heldout_subjects']==held,'EMA subject mismatch')
    require(checkpoint['binding_sha256']==sha(REPORT/'runtime_binding.json'),'EMA binding mismatch')
    torch.set_num_threads(4);torch.manual_seed(0)
    base=torch.load(ROOT/'checkpoints/backbone.ckpt',map_location='cpu',weights_only=False)
    model=extraction.legacy.Wavenet(**dict(base['hyper_parameters']['model_kwargs']))
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    names=set((ROOT/'manifests/week2_c_adaptation_v01/trainable_parameters.txt').read_text().splitlines())
    for name,p in model.named_parameters():p.requires_grad_(name in names)
    parent=yaml.safe_load(PARENT.read_text())
    require(ref.state_digest(model,True)==parent['model']['frozen_parameter_sha256'],'Frozen EMA changed')
    require(ref.state_digest(model)==status['selected_parameter_sha256'],'EMA parameter digest differs')
    model.requires_grad_(False).eval()
    extractor=extraction.AdaptedExtractor(model,14)
    folder=REPORT/f'features/fold{fold}_seed{seed}';folder.mkdir(parents=True,exist_ok=resume)
    grouped=defaultdict(list)
    for i,r in enumerate(rows):grouped[r['record_id']].append((i,r))
    partitions={r['record_id']:'train' for r in rows}
    signal_sha=json.loads((REPORT/'source_signal_sha256.json').read_text())
    first=next(iter(grouped));probe_rows=[r for _,r in grouped[first]]
    require(sha(ref.DATA/f'{first}.npz')==signal_sha[first],'Probe signal source changed')
    signal=extraction.load_signal_record(ref.DATA,first,partitions,probe_rows)[:4]
    uncached=pool_like_original(extractor(signal));extractor.cache_frozen_kernels()
    cached=pool_like_original(extractor(signal))
    np.testing.assert_array_equal(uncached,cached)
    baseline=np.load(S0,mmap_mode='r',allow_pickle=False)
    positions=[int(r['source_pool_position']) for _,r in grouped[first]][:4]
    np.testing.assert_array_equal(cached.reshape(-1,20,128)[:,:18],np.asarray(baseline[positions]).reshape(-1,20,128)[:,:18])
    del baseline
    write_json(folder/'probe.json',dict(status='passed',cached_uncached_bitwise_equal=True,
        frozen_first18_layers_equal_S0=True,weight_sha256=sha(weight),fold=fold,seed=seed))
    partial=folder/'pooled.partial.npy';target=folder/'pooled.npy'
    require(not target.exists(),'Completed pooled array already exists; verify via pipeline')
    dest=np.lib.format.open_memmap(partial,mode='r+' if resume and partial.exists() else 'w+',dtype=np.float32,shape=(len(rows),2560))
    progress=dict(status='running',fold=fold,seed=seed,records=0,samples=0,expected_records=len(grouped),expected_samples=len(rows),
        weight_sha256=sha(weight),test_used=False,original_dev8_read=False,started_at=now())
    write_json(folder/'summary.json',progress)
    try:
        for rec,entries in grouped.items():
            ids=[i for i,_ in entries];expected=[r for _,r in entries]
            require(ids==list(range(ids[0],ids[-1]+1)),'Record rows must be contiguous')
            section=slice(ids[0],ids[-1]+1);marker=folder/f'{rec}.done.json'
            if marker.exists():
                receipt=json.loads(marker.read_text())
                require(receipt['weight_sha256']==progress['weight_sha256'] and array_digest(dest[section])==receipt['pooled_sha256'],'Record receipt mismatch')
            else:
                require(sha(ref.DATA/f'{rec}.npz')==signal_sha[rec],'Signal input changed')
                signals=extraction.load_signal_record(ref.DATA,rec,partitions,expected)
                start=time.perf_counter()
                for j in range(0,len(ids),4):dest[ids[j:j+4]]=pool_like_original(extractor(signals[j:j+4]))
                dest.flush();del signals
                write_json(marker,dict(record_id=rec,fold=fold,seed=seed,samples=len(ids),
                    outer_role='heldout_frozen_inference_only' if expected[0]['subject_id'] in held else 'outer_train',
                    seconds=time.perf_counter()-start,pooled_sha256=array_digest(dest[section]),weight_sha256=progress['weight_sha256']))
            progress.update(records=progress['records']+1,samples=progress['samples']+len(ids),updated_at=now())
            write_json(folder/'summary.json',progress)
            print(json.dumps(dict(stage='pool_record',**progress)),flush=True)
        dest.flush();del dest;partial.replace(target)
        write_csv(folder/'feature_index.csv',[dict(r,outer_role='heldout' if r['subject_id'] in held else 'train') for r in rows])
        progress.update(status='complete',completed_at=now(),pooled_sha256=sha(target),NPZ_labels_read=False)
    except BaseException:
        progress.update(status='failed',error=traceback.format_exc());raise
    finally:write_json(folder/'summary.json',progress)


def fitted(folder,x,y,mean,std,value,cfg,status):
    if folder.exists():
        hashes=json.loads((folder/'artifact_sha256.json').read_text())
        require(all(sha(folder/name)==value for name,value in hashes.items()),'Saved fit changed')
        require(json.loads((folder/'config.json').read_text())==cfg,'Saved fit metadata changed')
        info=json.loads((folder/'optimizer_summary.json').read_text())
        require(info['converged'] and info['gradient_inf']<=linear.GTOL,'Unconverged saved fit')
        with np.load(folder/'model.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(mean,saved['mean']);np.testing.assert_array_equal(std,saved['std'])
            require(float(saved['regularization'])==value,'Saved regularization mismatch')
            theta=saved['theta'].copy()
        loss,grad=linear.loss_gradient(theta,x,y,value)
        require(float(np.max(np.abs(grad)))<=linear.GTOL,'Saved model no longer converges on exact data')
        return theta,info
    return linear.save_fit(folder,x,y,mean,std,value,cfg,status)


def audit_assets():
    rows,people,assignment=inputs();evidence=[];caches=[]
    for fold in range(3):
        training,held=fold_people(assignment,fold)
        allowed={r for p in people if p['subject_id'] in training for r in p['record_ids'].split(';')}
        for seed in SEEDS:
            folder=REPORT/f'adaptations/fold{fold}_seed{seed}'
            info=json.loads((folder/'summary.json').read_text());init=json.loads((folder/'initialization.json').read_text())
            require(info['status']=='complete' and info['completed_updates']==info['ema_updates']==10000,'Incomplete EMA trajectory')
            require(init['optimizer_updates']==init['ema_updates']==0 and init['adaptation_subjects']==training
                    and init['outer_heldout_subjects']==held and init['initializer_data_reads']==0,'Incorrect reset or subject pool')
            require(init['initial_parameter_sha256']==init['new_ema_initial_parameter_sha256'], 'Initial online/EMA mismatch')
            batches=read_csv(folder/'signal_batches.csv');accesses=read_csv(folder/'signal_access_audit.csv')
            history=read_csv(folder/'history.csv')
            require(len(history)==10000 and [int(r['step']) for r in history]==list(range(1,10001)),'Missing update history')
            require(all(np.isfinite(float(r['loss'])) and np.isfinite(float(r['gradient_norm_before_clip'])) for r in history),'Nonfinite history')
            require(len(batches)==20000 and {r['subject_id'] for r in batches}==set(training),'Sample coverage/count mismatch')
            require(all(r['record_id'] in allowed and r['subject_id'] in training and int(r['y_hidden'])==-1 for r in batches),'Outer person entered adaptation')
            require(len(accesses)==20000 and all(r['record_id'] in allowed and r['member'] in ref.ALLOWED_MEMBERS for r in accesses),'Forbidden member or record')
            weight=folder/info['selected_weight']['path'];require(sha(weight)==info['selected_weight']['sha256'],'EMA SHA mismatch')
            evidence.append(dict(fold=fold,seed=seed,train_people=36,heldout_people=18,optimizer_updates=10000,
                ema_updates=10000,sample_occurrences=20000,signal_member_reads=len(accesses),heldout_accesses=0,NPZ_label_accesses=0,
                selected_weight_sha256=info['selected_weight']['sha256']))
            cache=REPORT/f'features/fold{fold}_seed{seed}';summary=json.loads((cache/'summary.json').read_text())
            require(summary['status']=='complete' and summary['samples']==132938 and summary['weight_sha256']==info['selected_weight']['sha256'],'Cache provenance mismatch')
            require(sha(cache/'pooled.npy')==summary['pooled_sha256'],'Cache SHA mismatch')
            cache_rows=read_csv(cache/'feature_index.csv')
            require([r['sample_id'] for r in cache_rows]==[r['sample_id'] for r in rows],'Cache identity/order mismatch')
            probe=json.loads((cache/'probe.json').read_text())
            require(probe['status']=='passed' and probe['cached_uncached_bitwise_equal'] and probe['frozen_first18_layers_equal_S0'],'Extraction probe failed')
            caches.append(dict(fold=fold,seed=seed,samples=132938,pooled_sha256=summary['pooled_sha256'],probe_passed=True))
    write_json(REPORT/'adaptation_isolation_audit.json',dict(status='passed',checks=evidence,original_dev8_read=False,test_used=False))
    write_json(REPORT/'feature_cache_audit.json',dict(status='passed',checks=caches,FP16_rounding_before_pooling=True))


def classifiers(status):
    linear.REPORT=REPORT
    official=linear.official_module();rows,people,assignment=inputs()
    original=linear.make_orders(sorted(assignment),read_csv(linear.REFERENCE))
    all_s0=np.load(S0,mmap_mode='r',allow_pickle=False)
    original_positions=np.array([int(r['source_pool_position']) for r in rows])
    s0=np.asarray(all_s0[original_positions]);del all_s0
    source_sha=json.loads(S0_AUDIT.read_text())['pooled_sha256']
    lambdas={};registry=[]
    for fold in range(3):
        train,held=fold_people(assignment,fold);orders=roster_orders(train,original)
        for roster in ROSTERS:
            for budget,count in BUDGETS.items():
                selected=orders[roster][:count];rank={s:i for i,s in enumerate(selected)}
                ids=np.array([i for i,r in enumerate(rows) if r['subject_id'] in selected])
                order=np.argsort([rank[rows[i]['subject_id']] for i in ids],kind='stable');ids=ids[order]
                x=np.asarray(s0[ids]);y=np.array([int(rows[i]['y_true']) for i in ids],dtype=np.int64)
                subjects=np.array([rows[i]['subject_id'] for i in ids])
                def fit(tx,ty,mean,std,value,inner,training,validation):
                    key=f'f{fold}_r{roster}_{budget}_inner{inner}_lambda{value:g}'
                    cfg=dict(fit_id=key,role='train_only_S0_inner',fold=fold,roster_seed=roster,budget=budget,
                        regularization=value,training_subjects=training,inner_validation_subjects=validation,
                        outer_heldout_subjects=held,outer_labels_used=False,binding_sha256=sha(REPORT/'runtime_binding.json'),source_pool_sha256=source_sha)
                    status.update(current_fit=key);write_json(REPORT/'summary.json',status)
                    theta,info=fitted(REPORT/'cv_fits'/key,tx,ty,mean,std,value,cfg,status)
                    registry.append(dict(fit_id=key,role='inner',fold=fold,roster_seed=roster,budget=budget,
                        regularization=value,gradient_inf=info['gradient_inf'],seconds=info['seconds'],iterations=info['iterations']))
                    status['completed_cv_fits']+=1;write_json(REPORT/'summary.json',status)
                    write_csv(REPORT/'optimizer_fit_registry.csv',registry)
                    return theta,info
                chosen,scores,records,oof=linear.regularization_search(x,y,subjects,linear.LAMBDAS,fit,official.compute_metrics)
                lambdas[(fold,roster,budget)]=chosen
                write_json(REPORT/f'cv_f{fold}_r{roster}_{budget}.json',dict(selected_lambda=chosen,OOF_macro_f1={str(k):v for k,v in scores.items()},outer_labels_used=False))
                write_csv(REPORT/f'cv_f{fold}_r{roster}_{budget}_folds.csv',records)
                predictions=[]
                mapping=linear.inner_folds(selected)
                for j,i in enumerate(ids):
                    r=rows[i]
                    predictions.append(dict(sample_id=r['sample_id'],subject_id=r['subject_id'],y_true=r['y_true'],
                        inner_fold=mapping[r['subject_id']],**{f'pred_lambda_{v:g}':int(oof[v][j]) for v in linear.LAMBDAS}))
                write_csv(REPORT/f'cv_f{fold}_r{roster}_{budget}_OOF.csv',predictions)
                del x,y,subjects
    require(status['completed_cv_fits']==162,'Missing inner fits')
    write_csv(REPORT/'selected_regularization.csv',[dict(fold=f,roster_seed=r,budget=b,regularization=v) for (f,r,b),v in lambdas.items()])
    status['stage']='fit72_final_models_outer_predictions_embargoed';write_json(REPORT/'summary.json',status)
    for entry in plan():
        fold=entry['fold'];training,held=fold_people(assignment,fold)
        orders=roster_orders(training,original);selected=orders[entry['roster_seed']][:entry['labeled_subjects']]
        ids=np.array([i for i,r in enumerate(rows) if r['subject_id'] in selected])
        folder=REPORT/f"features/fold{fold}_seed{entry['adaptation_seed']}"
        pool=s0 if entry['source_key']=='S0' else np.load(folder/'pooled.npy',mmap_mode='r',allow_pickle=False)
        pool_sha=source_sha if entry['source_key']=='S0' else json.loads((folder/'summary.json').read_text())['pooled_sha256']
        x,mean,std=linear.normalize_training(pool[ids]);y=np.array([int(rows[i]['y_true']) for i in ids],dtype=np.int64)
        value=lambdas[(fold,entry['roster_seed'],entry['budget'])];key=entry['run_id']
        cfg=dict(entry,fit_id=key,role='outer_training_final',training_subjects=sorted(selected),
            training_samples=len(ids),outer_heldout_subjects=held,normalization_fitted_subjects=sorted(selected),
            outer_labels_used=False,regularization=value,binding_sha256=sha(REPORT/'runtime_binding.json'),source_pool_sha256=pool_sha)
        status.update(current_fit=key);write_json(REPORT/'summary.json',status)
        theta,info=fitted(REPORT/'final_models'/key,x,y,mean,std,value,cfg,status)
        registry.append(dict(fit_id=key,role='final',fold=fold,roster_seed=entry['roster_seed'],budget=entry['budget'],
            regularization=value,gradient_inf=info['gradient_inf'],seconds=info['seconds'],iterations=info['iterations']))
        status['completed_final_models']+=1;write_json(REPORT/'summary.json',status)
        write_csv(REPORT/'optimizer_fit_registry.csv',registry);del x,y,theta,pool
    del s0;gc.collect()
    require(status['completed_final_models']==72,'Missing final models')
    check_binding()
    write_json(REPORT/'outer_prediction_release.json',dict(released_at=now(),inner_fits=162,final_models=72,
        all_fits_converged=True,outer_evaluations_before_release=0,original_dev8_read=False,test_used=False))
    evaluate_and_analyze(rows,assignment,official,status)


def evaluate_and_analyze(rows,assignment,official,status):
    import pandas as pd
    status.update(stage='outer_frozen_predictions_and_analysis',current_fit=None)
    predictions=REPORT/'predictions';predictions.mkdir(exist_ok=True)
    results=[]
    for entry in plan():
        folder=REPORT/'final_models'/entry['run_id'];cfg=json.loads((folder/'config.json').read_text())
        hashes=json.loads((folder/'artifact_sha256.json').read_text())
        require(all(sha(folder/k)==v for k,v in hashes.items()),'Model archive changed before prediction')
        ids=np.array([i for i,r in enumerate(rows) if assignment[r['subject_id']]==entry['fold']])
        require(set(cfg['training_subjects']).isdisjoint({rows[i]['subject_id'] for i in ids}),'Prediction leakage')
        with np.load(folder/'model.npz',allow_pickle=False) as saved:
            theta=saved['theta'];mean=saved['mean'];std=saved['std']
        if entry['source_key']=='S0':
            pool=np.load(S0,mmap_mode='r',allow_pickle=False)
            values=pool[[int(rows[i]['source_pool_position']) for i in ids]]
        else:
            pool=np.load(REPORT/f"features/fold{entry['fold']}_seed{entry['adaptation_seed']}/pooled.npy",mmap_mode='r',allow_pickle=False)
            values=pool[ids]
        x=linear.apply_normalization(values,mean,std);prob=linear.predict_probabilities(theta,x)
        output=[]
        for i,p in zip(ids,prob):
            r=rows[i];item={k:r[k] for k in ['sample_id','subject_id','record_id','epoch_index','y_true']}
            item.update(y_pred=int(p.argmax()),**dict(zip(official.PROB_COLS,map(float,p))));output.append(item)
        target=predictions/(entry['run_id']+'.csv');write_csv(target,output)
        frame=official.load_predictions(target)
        require(len(frame)==len(ids) and frame.subject_id.nunique()==18,'Fold coverage mismatch')
        metrics=official.compute_metrics(frame.y_true.to_numpy(),frame.y_pred.to_numpy())
        write_json(predictions/(entry['run_id']+'_metrics.json'),metrics)
        results.append(dict(entry,macro_f1=metrics['macro_f1'],n1_f1=metrics['n1_f1'],n1_precision=metrics['n1_precision'],
            n1_recall=metrics['n1_recall'],accuracy=metrics['accuracy'],predictions_sha256=sha(target)))
        status['outer_evaluations']+=1;write_json(REPORT/'summary.json',status)
        del pool,values,x
    write_csv(REPORT/'fold_run_registry.csv',results)
    merged=REPORT/'OOF54';merged.mkdir(exist_ok=True);pairs=[];persons=[];fold_pairs=[]
    for roster in ROSTERS:
        for budget in BUDGETS:
            frames={}
            for source in ['S0','S1_aseed0','S1_aseed1','S1_aseed2']:
                group=[r for r in results if r['roster_seed']==roster and r['budget']==budget and r['source_key']==source]
                frame=pd.concat([official.load_predictions(predictions/(r['run_id']+'.csv')) for r in group],ignore_index=True)
                frame=frame.sort_values('sample_id').reset_index(drop=True)
                require(len(frame)==len(rows) and frame.sample_id.nunique()==len(rows) and frame.subject_id.nunique()==54,'OOF54 mismatch')
                frames[source]=frame
                path=merged/f'{source}_{budget}_r{roster}.csv';frame.to_csv(path,index=False)
                write_json(path.with_suffix('.metrics.json'),official.evaluate(path,n_boot=1000,seed=0))
            for seed in SEEDS:
                source=f'S1_aseed{seed}';a=frames[source];b=frames['S0']
                interval=official.paired_bootstrap_delta(a,b,n_boot=1000,seed=0)
                write_json(merged/f'delta_seed{seed}_{budget}_r{roster}.json',interval)
                am=official.compute_metrics(a.y_true.to_numpy(),a.y_pred.to_numpy())
                bm=official.compute_metrics(b.y_true.to_numpy(),b.y_pred.to_numpy())
                common=dict(roster_seed=roster,budget=budget,adaptation_seed=seed)
                pairs.append(dict(common,S0_macro_f1=bm['macro_f1'],S1_macro_f1=am['macro_f1'],
                    delta_macro_f1_pp=100*(am['macro_f1']-bm['macro_f1']),delta_n1_f1_pp=100*(am['n1_f1']-bm['n1_f1']),
                    delta_n1_precision_pp=100*(am['n1_precision']-bm['n1_precision']),delta_n1_recall_pp=100*(am['n1_recall']-bm['n1_recall']),
                    macro_ci_lower_pp=100*interval['macro_f1']['lo'],macro_ci_upper_pp=100*interval['macro_f1']['hi']))
                for subject in sorted(assignment):
                    af=a[a.subject_id==subject];bf=b[b.subject_id==subject]
                    ap=official.compute_metrics(af.y_true.to_numpy(),af.y_pred.to_numpy());bp=official.compute_metrics(bf.y_true.to_numpy(),bf.y_pred.to_numpy())
                    persons.append(dict(common,fold=assignment[subject],subject_id=subject,
                        delta_macro_f1_pp=100*(ap['macro_f1']-bp['macro_f1']),delta_n1_f1_pp=100*(ap['n1_f1']-bp['n1_f1'])))
                for fold in range(3):
                    ac=next(r for r in results if r['fold']==fold and r['roster_seed']==roster and r['budget']==budget and r['source_key']==source)
                    bc=next(r for r in results if r['fold']==fold and r['roster_seed']==roster and r['budget']==budget and r['source_key']=='S0')
                    fold_pairs.append(dict(common,fold=fold,delta_macro_f1_pp=100*(ac['macro_f1']-bc['macro_f1']),delta_n1_f1_pp=100*(ac['n1_f1']-bc['n1_f1'])))
    write_csv(REPORT/'OOF54_paired_deltas.csv',pairs);write_csv(REPORT/'per_subject_pairs.csv',persons);write_csv(REPORT/'fold_pairs.csv',fold_pairs)
    require(len(pairs)==18 and len(persons)==972 and len(fold_pairs)==54,'Incomplete paired matrix')
    fit_rows=read_csv(REPORT/'optimizer_fit_registry.csv')
    require(len(fit_rows)==234 and all(float(r['gradient_inf'])<=linear.GTOL for r in fit_rows),'Incomplete convergence matrix')
    write_json(REPORT/'completion_audit.json',dict(status='passed',new_fold_adaptations=9,inner_fits=162,final_models=72,
        OOF54_contrasts=18,subject_pairs=972,fold_pairs=54,outer_heldout_excluded_from_local_adaptation=True,
        original_dev8_read=False,test_used=False,all_fits_converged=True))
    lines=['# 按受试者留出验证：完成报告','',f'完成于 {now()}。3折×3适配种子，共9个新主干、162个内层拟合、72个最终模型通过审计。',
        '', '| 名单 | 预算人数 | 适配seed | S0 Macro-F1 | S1 Macro-F1 | 差值pp | 配对95%区间pp | N1 F1差值pp |',
        '| --- | --- | --- | ---: | ---: | ---: | --- | ---: |']
    for r in pairs:lines.append(f"| {r['roster_seed']} | {r['budget']} | {r['adaptation_seed']} | {100*r['S0_macro_f1']:.3f}% | {100*r['S1_macro_f1']:.3f}% | {r['delta_macro_f1_pp']:+.3f} | [{r['macro_ci_lower_pp']:+.3f}, {r['macro_ci_upper_pp']:+.3f}] | {r['delta_n1_f1_pp']:+.3f} |")
    lines+=['','每个OOF结果包含原54名训练者各一次外层留出预测。4/9人标注预算相对每折36人为11.11%/25%。各折模型、名单及同一人的重复对照相关，不能当成独立样本增大样本量；按人区间未完整包含训练随机性。',
        '', '结论仅适用于本次按折排除受试者的本地继续适配。此前该54人已用于开发，S0预训练重叠未审计，本轮不是全新外部数据验证。原8人开发验证与16人封存测试均未使用。详见逐折、逐人配对和锁定方案。结果尚未上传GitHub。']
    (REPORT/'按受试者留出_完成报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(resume=False):
    check_binding()
    for name in ['preflight_gradient','preflight_benchmark100']:
        require(json.loads((REPORT/name/'summary.json').read_text())['status']=='complete','Preflight not passed')
    require(sha(S0)==json.loads(S0_AUDIT.read_text())['pooled_sha256'],'S0 cache SHA mismatch')
    require(shutil.disk_usage(REPORT).free>20*1024**3,'Need at least20GiB free before starting nine new caches')
    with (REPORT/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume:require(json.loads((REPORT/'summary.json').read_text())['outer_evaluations']==0,'No automatic rerun after outer unsealing')
        status=dict(status='running',stage='foldwise_adaptation',pid=os.getpid(),started_at=now(),
            completed_adaptations=0,expected_adaptations=9,completed_caches=0,expected_caches=9,
            completed_cv_fits=0,expected_cv_fits=162,completed_final_models=0,expected_final_models=72,outer_evaluations=0,test_used=False)
        try:
            for fold in range(3):
                for seed in SEEDS:
                    for mode,subdir,counter in [('adapt','adaptations','completed_adaptations'),('extract','features','completed_caches')]:
                        folder=REPORT/f'{subdir}/fold{fold}_seed{seed}'
                        summary=folder/'summary.json'
                        status.update(stage='foldwise_'+mode,current_fold=fold,current_seed=seed,current_progress=summary.relative_to(ROOT).as_posix())
                        write_json(REPORT/'summary.json',status)
                        old=json.loads(summary.read_text()) if summary.exists() else {}
                        if old.get('status')=='complete':
                            require(resume,'Unexpected pre-existing completed worker')
                            p=folder/'pooled.npy' if mode=='extract' else folder/old['selected_weight']['path']
                            require(sha(p)==(old['pooled_sha256'] if mode=='extract' else old['selected_weight']['sha256']),'Completed worker changed')
                        else:
                            if old.get('status')=='failed':raise ValueError('Failed worker requires diagnosis; no automatic continuation')
                            args=[sys.executable,'-u',str(Path(__file__)),mode,'--fold',str(fold),'--seed',str(seed)]
                            if resume and folder.exists():args+=['--resume']
                            subprocess.run(args,check=True)
                        status[counter]+=1;write_json(REPORT/'summary.json',status)
            status.update(stage='audit_nine_fold_models_and_caches',current_progress=None)
            write_json(REPORT/'summary.json',status)
            audit_assets()
            status.update(stage='162_train_only_S0_inner_fits')
            write_json(REPORT/'summary.json',status)
            with threadpool_limits(limits=4):classifiers(status)
            check_binding()
            status.update(status='complete',stage='subject_holdout_analysis_complete',completed_at=now())
        except BaseException:
            status.update(status='failed',failed_at=now(),error=traceback.format_exc());raise
        finally:write_json(REPORT/'summary.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','bind','gradient','benchmark','adapt','extract','run'])
    p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='bind':bind()
    elif args.mode=='gradient':adaptation_run(0,0,diagnostic='preflight_gradient')
    elif args.mode=='benchmark':
        require(json.loads((REPORT/'preflight_gradient/summary.json').read_text())['status']=='complete','Gradient gate missing')
        adaptation_run(0,0,diagnostic='preflight_benchmark100')
    elif args.mode=='adapt':adaptation_run(args.fold,args.seed,args.resume)
    elif args.mode=='extract':extraction_run(args.fold,args.seed,args.resume)
    else:run(args.resume)


if __name__=='__main__':main()
