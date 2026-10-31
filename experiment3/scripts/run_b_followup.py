"""Audit B's delivery, run 3 matched label budgets and write official comparisons."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'vendor/experiment2-review-6e7df99'
A=ROOT.parent
OUT=ROOT/'reports/b_followup_20261023'
BASE=ROOT/'runs/E3_EEG_label100_seed0_provisional_v01'
COMMIT='6e7df9978c48c84bbef4d2f7e9d66e9ccd93b923'

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('official',B/'src/evaluation/evaluate.py')
    ev=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    split=pd.read_csv(B/'manifests/split_subjects.csv').sort_values('subject_id').reset_index(drop=True)
    original=pd.read_csv(A/'manifests/split_subjects.csv').sort_values('subject_id').reset_index(drop=True)
    assert split.equals(original)
    idx=pd.read_csv(ROOT/'features/eeg_cond14_v01/feature_index.csv')
    tr=idx[idx.split=='train']
    val=idx[idx.split=='val'].sort_values('sample_id').reset_index(drop=True)
    assert len(tr)==132938 and len(val)==21186
    assert set(idx.split)=={'train','val'}
    basecfg=json.loads((BASE/'run_config.json').read_text())
    assert set(basecfg['training_subjects'])==set(tr.subject_id)
    assert set(basecfg['validation_subjects'])==set(val.subject_id)
    lists={}
    for seed in (2024,2025,2026):
        filename='label_subsets_v1.csv' if seed==2024 else f'label_subsets_v1-s{seed}.csv'
        frame=pd.read_csv(B/'manifests'/filename)
        assert len(frame)==54 and frame.subject_id.is_unique and set(frame.subject_id)==set(tr.subject_id)
        assert set(frame.sampling_seed)=={seed}
        for col in ('label25','label50','label100'):
            assert set(frame[col]) <= {0,1}
        selected=set(frame.loc[frame.label25==1,'subject_id'])
        assert len(selected)==14 and selected <= set(frame.loc[frame.label50==1,'subject_id']) <= set(frame.loc[frame.label100==1,'subject_id'])
        data=tr[tr.subject_id.isin(selected)]
        for row in frame.itertuples():
            actual=tr[tr.subject_id==row.subject_id]
            assert len(actual)==row.n_epochs
            assert set(actual.record_id)==set(row.record_ids.split(';'))
        lists[seed]=dict(subjects=sorted(selected),records=int(data.record_id.nunique()),samples=len(data),
            class_counts={str(i):int((data.y_true==i).sum()) for i in range(5)},
            manifest=filename,sha256=hashlib.sha256((B/'manifests'/filename).read_bytes()).hexdigest())
    audit=dict(status='passed',b_branch='feat/b-evaluation',b_commit=COMMIT,
        split_matches_A=True,label100_matches_previous_E3=True,subsets=lists,
        evaluator_sha256=hashlib.sha256((B/'src/evaluation/evaluate.py').read_bytes()).hexdigest(),test_used=False)
    dump(OUT/'b_delivery_audit.json',audit)
    print('B delivery audit passed',flush=True)

    def checked_frame(path):
        df=ev.load_predictions(path).sort_values('sample_id').reset_index(drop=True)
        assert np.isfinite(df[ev.PROB_COLS].to_numpy()).all()
        for key in ('sample_id','subject_id','record_id','epoch_index','y_true'):
            assert df[key].equals(val[key])
        return df

    def evaluate_run(path,outname):
        df=checked_frame(path)
        metrics=ev.evaluate(path,n_boot=1000,seed=0)
        metrics['b_commit']=COMMIT
        dump(OUT/outname,metrics)
        return df,metrics

    acsv=A/'交付包/实验①交付包_20261018/3_验证集预测_给D/predictions_val_EEG.csv'
    af,am=evaluate_run(acsv,'metrics_official_CNN100.json')
    cf,cm=evaluate_run(BASE/'predictions_val.csv','metrics_official_E3_100.json')
    old=json.loads((BASE/'metrics_provisional.json').read_text())
    assert abs(old['macro_f1']-cm['macro_f1'])<1e-12
    assert old['confusion_matrix']==cm['confusion_matrix']
    flat=ROOT/'runs/E3_EEG_label100_seed0_timeflatten_v01/predictions_val.csv'
    if flat.exists():
        evaluate_run(flat,'metrics_official_E3_timeflatten100.json')
    dump(OUT/'paired_E3_100_vs_CNN100.json',ev.paired_bootstrap_delta(cf,af,n_boot=1000,seed=0))
    all_results=[]
    comparisons={}
    for seed in (2024,2025,2026):
        run=ROOT/'runs'/f'E3_EEG_label25_sseed{seed}_seed0_v01'
        if not (run/'summary.json').exists():
            print(f'Start E3 25% sampling_seed={seed}',flush=True)
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/train_linear_budget.py'),'--sampling-seed',str(seed)],check=True)
        ff,fm=evaluate_run(run/'predictions_val.csv',f'metrics_official_E3_25_sseed{seed}.json')
        bcsv=B/'reports/predictions_E2'/f'predictions_val_E2_EEG_label25_sseed{seed}_seed0_v01.csv'
        bf,bm=evaluate_run(bcsv,f'metrics_official_CNN25_sseed{seed}.json')
        b_saved=json.loads((B/'reports/runs_E2'/f'metrics_E2_EEG_label25_sseed{seed}_seed0_v01.json').read_text())
        assert abs(bm['macro_f1']-b_saved['macro_f1'])<1e-12
        delta=ev.paired_bootstrap_delta(ff,bf,n_boot=1000,seed=0)
        decline=ev.paired_bootstrap_delta(ff,cf,n_boot=1000,seed=0)
        comparisons[seed]=dict(E3_25_minus_CNN25=delta,E3_25_minus_E3_100=decline)
        dump(OUT/f'paired_sseed{seed}.json',comparisons[seed])
        all_results.append(dict(sampling_seed=seed,train_subjects=14,train_samples=lists[seed]['samples'],
            E3_macro_f1=fm['macro_f1'],CNN_macro_f1=bm['macro_f1'],E3_n1_f1=fm['n1_f1'],CNN_n1_f1=bm['n1_f1'],
            E3_accuracy=fm['accuracy'],CNN_accuracy=bm['accuracy'],
            E3_drop_pp=100*(cm['macro_f1']-fm['macro_f1']),CNN_drop_pp=100*(am['macro_f1']-bm['macro_f1']),
            E3_minus_CNN_pp=100*(fm['macro_f1']-bm['macro_f1'])))
        print(json.dumps(all_results[-1]),flush=True)
    # Verify all three new heads directly from the shared original validation features.
    # Read each validation feature block once, applying each saved normalizer/head.
    print('Replaying all validation predictions from original features for all 3 new heads',flush=True)
    torch.set_num_threads(4)
    heads={}
    for seed in (2024,2025,2026):
        run=ROOT/'runs'/f'E3_EEG_label25_sseed{seed}_seed0_v01'
        ckpt=torch.load(run/'best_head.pt',map_location='cpu',weights_only=True)
        model=torch.nn.Linear(2560,5).eval()
        model.load_state_dict(ckpt['model'])
        saved=checked_frame(run/'predictions_val.csv')
        heads[seed]=(model,ckpt,saved,np.empty((len(val),5),dtype=np.float32))
    with torch.inference_mode():
        for name, group in val.groupby('feature_path',sort=True):
            data=np.load(ROOT/'features/eeg_cond14_v01'/name,mmap_mode='r',allow_pickle=False)
            for start in range(0,len(group),64):
                block=group.iloc[start:start+64]
                raw=data[block.row_index.to_numpy()].astype(np.float32)
                assert np.isfinite(raw).all()
                value=torch.from_numpy(raw.mean(axis=2).reshape(len(block),2560))
                for model,ckpt,saved,predictions in heads.values():
                    predictions[block.index]=model((value-ckpt['mean'])/ckpt['std']).softmax(1).numpy()
    replay={}
    for seed,(model,ckpt,saved,predictions) in heads.items():
        actual=saved[ev.PROB_COLS].to_numpy()
        assert np.allclose(predictions,actual,atol=2e-5,rtol=1e-5)
        assert np.array_equal(predictions.argmax(1),saved.y_pred.to_numpy())
        replay[seed]=dict(status='passed',samples=len(val),max_probability_error=float(abs(predictions-actual).max()))
    dump(OUT/'new_heads_replay_audit.json',replay)
    pd.DataFrame(all_results).to_csv(OUT/'comparison_summary.csv',index=False)
    mean_c=float(np.mean([r['E3_macro_f1'] for r in all_results]))
    std_c=float(np.std([r['E3_macro_f1'] for r in all_results],ddof=1))
    mean_b=float(np.mean([r['CNN_macro_f1'] for r in all_results]))
    std_b=float(np.std([r['CNN_macro_f1'] for r in all_results],ddof=1))
    lines=['# 实验3：B材料接收后的少标签实验与统一评估','',
        f'日期2026-10-23。B分支feat/b-evaluation，commit {COMMIT}。主分支尚未包含该交付；本地使用固定版本快照。',
        '', '## 材料验收与设置','',
        'B的划分与原A划分一致；100%名单与已完成E3训练一致。三份25%名单均为54名训练受试者的14人子集，与验证/测试互斥；实际记录、样本与类别计数已逐一核对。主名单种子2024，2025/2026为重复名单。',
        '', '复用已冻结的条件14 EEGDM特征，时间均值后2560维，单层Linear(2560,5)。训练seed=0、AdamW、lr=0.001、weight_decay=0.0001、batch=128、最多50轮、验证Macro-F1早停patience=10，与E3 100%方案一致。标准化仅用各组有标签训练受试者；未标注受试者特征和标签不参与分类头训练或标准化。',
        '', 'B的官方evaluate.py v1假数据8项检查通过；已有100%和时间展开预测均已用该版本复算。主方案100%指标及混淆矩阵与原临时指标一致，因此无需重训100%主方案。',
        '', '## 验证集主指标','', '| 方案 | 标签预算/抽样种子 | 训练段数 | Macro-F1 | N1 F1 | Accuracy |','|---|---|---:|---:|---:|---:|',
        f"| CNN (A) | 100% | 132938 | {am['macro_f1']*100:.2f}% | {am['n1_f1']*100:.2f}% | {am['accuracy']*100:.2f}% |",
        f"| EEGDM+线性头 (C) | 100% | 132938 | {cm['macro_f1']*100:.2f}% | {cm['n1_f1']*100:.2f}% | {cm['accuracy']*100:.2f}% |"]
    for r in all_results:
        lines += [f"| CNN (B) | 25% / {r['sampling_seed']} | {r['train_samples']} | {r['CNN_macro_f1']*100:.2f}% | {r['CNN_n1_f1']*100:.2f}% | {r['CNN_accuracy']*100:.2f}% |",
                  f"| EEGDM+线性头 (C) | 25% / {r['sampling_seed']} | {r['train_samples']} | {r['E3_macro_f1']*100:.2f}% | {r['E3_n1_f1']*100:.2f}% | {r['E3_accuracy']*100:.2f}% |"]
    primary_c=json.loads((OUT/'metrics_official_E3_25_sseed2024.json').read_text())
    primary_b=json.loads((OUT/'metrics_official_CNN25_sseed2024.json').read_text())
    lines += ['', '## 主名单各类别指标（抽样种子2024）','', '| 指标 | CNN100% | CNN25% | EEGDM100% | EEGDM25% |','|---|---:|---:|---:|---:|']
    for stage in ev.STAGE_NAMES:
        vals=[m['per_class_f1'][stage] for m in (am,primary_b,cm,primary_c)]
        lines.append('| '+stage+' F1 | '+' | '.join(f'{v*100:.2f}%' for v in vals)+' |')
    for key,title in [('n1_precision','N1 Precision'),('n1_recall','N1 Recall')]:
        lines.append('| '+title+' | '+' | '.join(f'{m[key]*100:.2f}%' for m in (am,primary_b,cm,primary_c))+' |')
    lines += ['', '## 标签减少时的变化','', '| 抽样种子 | EEGDM下降/百分点 | CNN下降/百分点 | 25%时EEGDM−CNN/百分点 [配对95%区间] |','|---|---:|---:|---|']
    for r in all_results:
        d=comparisons[r['sampling_seed']]['E3_25_minus_CNN25']['macro_f1']
        lines.append(f"| {r['sampling_seed']} | {r['E3_drop_pp']:.2f} | {r['CNN_drop_pp']:.2f} | {r['E3_minus_CNN_pp']:+.2f} [{d['lo']*100:+.2f}, {d['hi']*100:+.2f}] |")
    lines += ['', f'三份25%名单的Macro-F1均值±样本标准差：EEGDM {mean_c*100:.2f}% ± {std_c*100:.2f}个百分点；CNN {mean_b*100:.2f}% ± {std_b*100:.2f}个百分点。n=3仅供观察名单波动，不作为置信区间。',
        '', f"本轮三份名单中，EEGDM的Macro-F1在{sum(r['E3_macro_f1']>r['CNN_macro_f1'] for r in all_results)}/3份高于CNN。以上按预先固定主名单及全部重复结果报告，不选择最好名单作为主结果。应同时查看平均差、标签减少时的下降与配对区间，不能仅凭某一份名单或N1单项改善声称整体优势。",
        '', '## 解释边界','',
        '所有比较使用同一8名验证受试者、21186段，预测逐条对齐。三组新模型均从原始特征及保存的最佳头重放全部验证样本，类别预测一致。配对区间按受试者bootstrap 1000次、seed=0；不按片段重采样。验证集用于早停和此前池化配置选择，区间未校正模型选择，不能替代独立测试或统计显著性结论。',
        '', 'EEGDM与CNN的模型结构、外部预训练、输入变换、训练设备不同；CNN 100%为CUDA、25%为B的MPS，E3分类头为CPU。CNN最多60轮、E3最多50轮；优化器也不同。这是已报告各自方案的少标签对照，不是只改变主干的严格消融。',
        '', '14/54=25.93%的训练受试者（25%预算按人数取整）；各组片段比例不同，不能写成严格25%的片段。每份名单只训练seed0一次，名单波动与训练随机性影响无法分离。未训练50%或使用测试集。',
        '', '## 给D的新产物','',
        'runs/E3_EEG_label25_sseed{2024,2025,2026}_seed0_v01中的predictions_val.csv、best_head.pt、run_config.json、history.json、metrics_official.json、summary.json、training_subjects_25_from_B.txt。新预测沿用原sample_id与五类概率列；特征目录与索引完全复用，无需重新传23.67GB特征。正式重算的100%指标位于本报告目录metrics_official_E3_100.json。',
        '', '原桌面20261020交付包保留为当时快照，尚未包含本轮新增结果。']
    (OUT/'实验3_少标签统一评估报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(OUT/'summary.json',dict(status='complete',b_commit=COMMIT,test_used=False,
        label100_revalidated=True,completed_sampling_seeds=[2024,2025,2026],results=all_results,
        E3_25_mean_macro_f1=mean_c,E3_25_std_macro_f1=std_c))
    print('ALL EXPERIMENTS AND OFFICIAL COMPARISONS COMPLETE',flush=True)

if __name__=='__main__':
    main()
