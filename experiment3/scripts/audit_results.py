"""Independent validation of E3 outputs and paired comparison with delivered A EEG."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score

ROOT=Path(__file__).resolve().parents[1]
A=ROOT.parent
DELIVERY=A/'交付包/实验①交付包_20261018'
RUN=ROOT/'runs/E3_EEG_label100_seed0_provisional_v01'
FEATURE=ROOT/'features/eeg_cond14_v01'
OUT=ROOT/'reports/result_audit'
NAMES=['W','N1','N2','N3','REM']

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def metrics(y,p):
    pr,re,f,s=precision_recall_fscore_support(y,p,labels=range(5),zero_division=0)
    return dict(accuracy=float(accuracy_score(y,p)),macro_f1=float(f.mean()),
        per_class_precision=dict(zip(NAMES,pr.tolist())),per_class_recall=dict(zip(NAMES,re.tolist())),
        per_class_f1=dict(zip(NAMES,f.tolist())),per_class_support=dict(zip(NAMES,s.tolist())),
        confusion_matrix=confusion_matrix(y,p,labels=range(5)).tolist())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    split=read(A/'manifests/split_subjects.csv')
    subjects={s:{r['subject_id'] for r in split if r['split']==s} for s in ('train','val','test')}
    assert not (subjects['train']&subjects['val'] or subjects['train']&subjects['test'] or subjects['val']&subjects['test'])
    cfg=json.loads((RUN/'run_config.json').read_text())
    assert set(cfg['training_subjects'])==subjects['train']
    assert set(cfg['validation_subjects'])==subjects['val']
    idx=read(FEATURE/'feature_index.csv')
    assert len(idx)==154124 and len({r['sample_id'] for r in idx})==len(idx)
    expected={}
    shapes=0
    for r in split:
        if r['split']=='test':
            continue
        for rec in r['record_ids'].split(';'):
            with np.load(DELIVERY/f'2_预处理数据与脚本_给B和C/data/processed/{rec}.npz',allow_pickle=False) as z:
                labels,epochs=z['labels'],z['epoch_index']
                f=np.load(FEATURE/f'{rec}.npy',mmap_mode='r',allow_pickle=False)
                assert f.shape==(len(labels),20,30,128) and f.dtype==np.float16
                shapes+=1
                for j,(label,epoch) in enumerate(zip(labels,epochs)):
                    expected[f'{rec}_{int(epoch):04d}']=(r['subject_id'],rec,int(epoch),int(label),r['split'],j)
    assert set(expected)=={r['sample_id'] for r in idx}
    for r in idx:
        assert (r['subject_id'],r['record_id'],int(r['epoch_index']),int(r['y_true']),r['split'],int(r['row_index']))==expected[r['sample_id']]
        assert r['feature_path']==r['record_id']+'.npy'
    a=read(DELIVERY/'3_验证集预测_给D/predictions_val_EEG.csv')
    c=read(RUN/'predictions_val.csv')
    amap={r['sample_id']:r for r in a}
    assert len(a)==len(amap)==len(c)==21186
    assert set(amap)=={r['sample_id'] for r in c}=={k for k,v in expected.items() if v[4]=='val'}
    a=[amap[r['sample_id']] for r in c]
    probability_checks={}
    predictions={}
    y=np.array([int(r['y_true']) for r in c])
    for name,rows in [('A',a),('E3',c)]:
        for r in rows:
            assert (r['subject_id'],r['record_id'],int(r['epoch_index']),int(r['y_true']))==expected[r['sample_id']][:4]
        probs=np.array([[float(r['p_'+n]) for n in NAMES] for r in rows])
        pred=np.array([int(r['y_pred']) for r in rows])
        assert np.isfinite(probs).all() and probs.min()>=0 and probs.max()<=1
        assert np.allclose(probs.sum(1),1,atol=1e-6)
        assert np.array_equal(probs.argmax(1),pred)
        probability_checks[name]=dict(max_sum_error=float(abs(probs.sum(1)-1).max()),argmax_matches=True)
        predictions[name]=pred
    ma,mc=metrics(y,predictions['A']),metrics(y,predictions['E3'])
    for calculated,path in [(ma,A/'reports/metrics_E1_EEG_label100_seed0_v01.json'),(mc,RUN/'metrics_provisional.json')]:
        saved=json.loads(path.read_text())
        for key,value in calculated.items():
            if isinstance(value,dict):
                assert all(np.isclose(v,saved[key][k],rtol=0,atol=1e-12) for k,v in value.items())
            else:
                assert np.allclose(value,saved[key],rtol=0,atol=1e-12)
    # Reproduce all validation probabilities from saved features and saved best head.
    ckpt=torch.load(RUN/'best_head.pt',map_location='cpu',weights_only=True)
    model=torch.nn.Linear(2560,5).eval()
    model.load_state_dict(ckpt['model'])
    assert torch.isfinite(ckpt['mean']).all() and torch.isfinite(ckpt['std']).all() and (ckpt['std']>0).all()
    locations={r['sample_id']:r for r in idx}
    reproduced=np.empty((len(c),5),np.float32)
    groups={}
    for i,r in enumerate(c):
        loc=locations[r['sample_id']]
        groups.setdefault(loc['feature_path'],[]).append((i,int(loc['row_index'])))
    with torch.inference_mode():
        for filename,entries in groups.items():
            features=np.load(FEATURE/filename,mmap_mode='r',allow_pickle=False)
            for start in range(0,len(entries),64):
                block=entries[start:start+64]
                data=features[[j for _,j in block]].astype(np.float32)
                assert np.isfinite(data).all()
                x=torch.from_numpy(data.mean(axis=2).reshape(len(block),2560))
                reproduced[[i for i,_ in block]]=model((x-ckpt['mean'])/ckpt['std']).softmax(1).numpy()
    saved_probs=np.array([[float(r['p_'+n]) for n in NAMES] for r in c])
    max_error=float(abs(reproduced-saved_probs).max())
    assert np.allclose(reproduced,saved_probs,atol=2e-5,rtol=1e-5),max_error
    assert np.array_equal(reproduced.argmax(1),predictions['E3'])
    history=json.loads((RUN/'history.json').read_text())
    best=max(history,key=lambda r:r['val_macro_f1'])
    assert best['epoch']==ckpt['epoch']==37
    assert np.isclose(best['val_macro_f1'],mc['macro_f1'])
    per_subject=[]
    for sub in sorted(subjects['val']):
        mask=np.array([r['subject_id']==sub for r in c])
        sa=f1_score(y[mask],predictions['A'][mask],labels=range(5),average='macro',zero_division=0)
        sc=f1_score(y[mask],predictions['E3'][mask],labels=range(5),average='macro',zero_division=0)
        per_subject.append(dict(subject=sub,n=int(mask.sum()),A_macro_f1=sa,E3_macro_f1=sc,delta_pp=100*(sc-sa)))
    audit=dict(status='passed',index_rows=len(idx),feature_shapes_checked=shapes,validation_rows=len(c),
        split_subjects={k:len(v) for k,v in subjects.items()},probabilities=probability_checks,
        all_validation_predictions_reproduced=True,max_probability_reproduction_error=max_error,
        best_epoch=37,metrics_recalculated_match=True,test_arrays_read=False,
        scope='Checks local delivered artifacts and E3 validation replay; does not retrain A or independently re-extract backbone features.')
    dump(OUT/'audit.json',audit)
    dump(OUT/'comparison.json',dict(A=ma,E3=mc,per_subject=per_subject))
    lines=['# 实验3结果核验与A初步对比','',
        '核验通过：154124条特征索引与A原始标签/epoch索引完全对应；122个特征文件形状和dtype正确。训练54人、验证8人、测试16人互斥，E3使用全部54名训练受试者。',
        '', '两份验证预测均为相同的21186段，sample_id无重复、无缺失，标签一致；概率有限、范围及行和正常，argmax与预测一致。独立重算指标与保存值一致。',
        '',f'从保存的特征、标准化参数和最佳分类头重新推理全部验证样本，预测类别完全一致；概率最大绝对误差{max_error:.3g}。最佳第37轮与训练历史一致。未读取测试数组。',
        '', '## 指标对比（百分比；差值为百分点）','', '| 指标 | A：监督CNN | E3：冻结EEGDM+线性头 | E3−A |','|---|---:|---:|---:|']
    table=[('Macro-F1',ma['macro_f1'],mc['macro_f1']),('Accuracy',ma['accuracy'],mc['accuracy'])]
    table += [(n+' F1',ma['per_class_f1'][n],mc['per_class_f1'][n]) for n in NAMES]
    table += [('N1 Precision',ma['per_class_precision']['N1'],mc['per_class_precision']['N1']),('N1 Recall',ma['per_class_recall']['N1'],mc['per_class_recall']['N1'])]
    lines += [f'| {name} | {aa*100:.2f} | {cc*100:.2f} | {(cc-aa)*100:+.2f} |' for name,aa,cc in table]
    lines += ['', '## 误差来源','',
        'E3的N1召回提高，但精确率下降：正确识别N1从1165增加到1415段，同时更多W、N2、REM被判为N1。N1 F1小幅改善，不能只凭召回提高判断整体更好。',
        '', 'REM是最大单类F1差距：REM→N1从234增加到523段，正确REM从1798减少到1497段；N2→N1从592增加到842段，W→N1从494增加到658段。以上是观察到的混淆变化，不能据此确定因果机制。',
        '', '## 按验证受试者拆分','', '| 受试者 | 段数 | A Macro-F1 | E3 Macro-F1 | 差值/百分点 |','|---|---:|---:|---:|---:|']
    lines += [f"| {r['subject']} | {r['n']} | {r['A_macro_f1']*100:.2f} | {r['E3_macro_f1']*100:.2f} | {r['delta_pp']:+.2f} |" for r in per_subject]
    lines += ['', '每位受试者均使用固定五类、zero_division=0；不能将该表简单平均视为上面的合并样本Macro-F1。',
        '', '## 结论与边界','',
        '当前单次运行中，E3整体低于A，N1 F1略高；结果不是EEGDM整体能力的最终判断。两者使用相同验证样本与标签、同一指标定义，适合做初步配对对比。',
        '', '模型和输入管线不同：E3使用外部预训练冻结主干、代理条件14、200Hz及幅度压缩、时间均值池化和线性头；A为监督CNN。训练优化及早停设置也不完全相同，因此这不是只改变一个因素的严格消融。',
        '', '验证集用于早停选模型，分数是模型选择结果，不是独立测试表现；只有一次seed=0运行、8名验证受试者，未做显著性检验，不声称统计显著优劣。B正式名单和统一评估仍待复核。',
        '', '本次核验覆盖本地交付文件、全部索引、预测与验证集分类头重放；没有重训A、重提全部主干特征或审计外部预训练数据重叠。训练特征有限值检查在原训练加载阶段完成，本次重扫的是全部验证特征。',
        '', '建议先保留当前结果作为v01基线并向D交接。若继续尝试其他通道/池化，应作为独立配置记录，不覆盖v01；25%实验必须等待B名单。']
    (OUT/'核验与初步对比.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    print(json.dumps(per_subject),flush=True)

if __name__=='__main__':
    main()
