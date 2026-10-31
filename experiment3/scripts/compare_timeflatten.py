"""Wait for the current experiment, verify raw-feature predictions and write comparison."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/E3_EEG_label100_seed0_timeflatten_v01'
BASE=ROOT/'runs/E3_EEG_label100_seed0_provisional_v01'
FEATURE=ROOT/'features/eeg_cond14_v01'
OUT=ROOT/'reports/time_pooling_comparison'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def main():
    deadline=time.monotonic()+21600
    while not (RUN/'summary.json').exists():
        if time.monotonic()>deadline:
            raise RuntimeError('No completed training result within six hours')
        time.sleep(30)
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    baseline=read(BASE/'predictions_val.csv')
    rows=read(RUN/'predictions_val.csv')
    assert len(rows)==len(baseline)==21186
    assert len({r['sample_id'] for r in rows})==len(rows)
    keys=['sample_id','subject_id','record_id','epoch_index','y_true']
    assert [[r[k] for k in keys] for r in rows]==[[r[k] for k in keys] for r in baseline]
    names=['W','N1','N2','N3','REM']
    probs=np.array([[float(r['p_'+n]) for n in names] for r in rows])
    y=np.array([int(r['y_true']) for r in rows])
    pred=np.array([int(r['y_pred']) for r in rows])
    assert np.isfinite(probs).all() and probs.min()>=0 and probs.max()<=1
    assert np.allclose(probs.sum(1),1,rtol=0,atol=1e-6)
    assert np.array_equal(probs.argmax(1),pred)
    config=json.loads((RUN/'run_config.json').read_text())
    basecfg=json.loads((BASE/'run_config.json').read_text())
    same=['seed','training_subjects','validation_subjects','optimizer','lr','weight_decay','batch_size','max_epochs','patience','selection','feature_config']
    assert all(config[k]==basecfg[k] for k in same)
    index={r['sample_id']:r for r in read(FEATURE/'feature_index.csv')}
    ckpt=torch.load(RUN/'best_head.pt',map_location='cpu',weights_only=True)
    model=torch.nn.Linear(76800,5).eval()
    model.load_state_dict(ckpt['model'])
    grouped={}
    for i,r in enumerate(rows):
        loc=index[r['sample_id']]
        assert loc['split']=='val' and loc['y_true']==r['y_true']
        grouped.setdefault(loc['feature_path'],[]).append((i,int(loc['row_index'])))
    reproduced=np.empty_like(probs)
    with torch.inference_mode():
        for name,entries in grouped.items():
            f=np.load(FEATURE/name,mmap_mode='r',allow_pickle=False)
            for start in range(0,len(entries),64):
                block=entries[start:start+64]
                x=torch.from_numpy(f[[j for _,j in block]].astype(np.float32).reshape(len(block),76800))
                reproduced[[i for i,_ in block]]=model((x-ckpt['mean'])/ckpt['std']).softmax(1).numpy()
    assert np.allclose(reproduced,probs,atol=3e-5,rtol=1e-5)
    assert np.array_equal(reproduced.argmax(1),pred)
    m=json.loads((RUN/'metrics_provisional.json').read_text())
    b=json.loads((BASE/'metrics_provisional.json').read_text())
    a=json.loads((ROOT.parent/'reports/metrics_E1_EEG_label100_seed0_v01.json').read_text())
    assert abs(f1_score(y,pred,labels=range(5),average='macro',zero_division=0)-m['macro_f1'])<1e-12
    assert abs(accuracy_score(y,pred)-m['accuracy'])<1e-12
    perclass=f1_score(y,pred,labels=range(5),average=None,zero_division=0)
    assert all(abs(perclass[i]-m['per_class_f1'][n])<1e-12 for i,n in enumerate(names))
    result=dict(status='passed',samples=len(rows),raw_feature_replay=True,
        max_probability_error=float(abs(reproduced-probs).max()),same_configuration_fields=same,
        mean_macro_f1=b['macro_f1'],flatten_macro_f1=m['macro_f1'],A_macro_f1=a['macro_f1'],
        flatten_minus_mean_pp=100*(m['macro_f1']-b['macro_f1']),test_used=False)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    summary=json.loads((RUN/'summary.json').read_text())
    lines=['# 时间池化方式对比','',
        '固定同一批冻结EEGDM条件14特征、同一训练/验证划分和100%标签。原基线文件保持不变。本次仅使用训练、验证数据；正式评估仍等待B。',
        '', '| 项目 | 时间均值基线 | 保留时间位置 |','|---|---|---|',
        '| 输入 | 20×128=2560维 | 20×30×128=76800维 |',
        '| 分类头 | 单层线性 | 单层线性 |','| 参数量 | 12805 | 384005 |',
        '| 标准化 | 训练集每层/隐藏维 | 训练集每层/时间位置/隐藏维 |',
        '| seed / lr / batch | 0 / 0.001 / 128 | 相同 |',
        '| 优化器 / weight decay | AdamW / 0.0001 | 相同 |',
        '| 最多轮数 / 早停patience | 50 / 10 | 相同 |',
        '', '## 验证集结果','', '| 指标 | 时间均值 | 保留时间 | 差值/百分点 | A CNN参考 |','|---|---:|---:|---:|---:|']
    metrics=[('Macro-F1',b['macro_f1'],m['macro_f1'],a['macro_f1']),('Accuracy',b['accuracy'],m['accuracy'],a['accuracy'])]
    metrics += [(n+' F1',b['per_class_f1'][n],m['per_class_f1'][n],a['per_class_f1'][n]) for n in names]
    metrics += [('N1 Precision',b['n1_precision'],m['n1_precision'],a['n1_precision']),('N1 Recall',b['n1_recall'],m['n1_recall'],a['n1_recall'])]
    lines += [f'| {n} | {bb*100:.2f} | {mm*100:.2f} | {(mm-bb)*100:+.2f} | {aa*100:.2f} |' for n,bb,mm,aa in metrics]
    lines += ['', f"本次运行{summary['epochs']}轮，最佳第{summary['best_epoch']}轮，训练及逐轮验证耗时{summary['training_seconds']/60:.1f}分钟（不含准备磁盘缓存与拟合标准化）。",
        '', '## 核验与解释边界','',
        '全部21186条预测与原基线逐条对齐；概率、标签和重算指标检查通过。直接从原始特征文件及保存的最佳分类头重放全部验证样本，分类结果一致，排除了磁盘缓存重排或展平轴错误导致的预测不一致。',
        '', '参数量增加约30倍，同时标准化从时间均值的各维变为每个时间位置的各维；结果反映这一整套表示改变，不是固定模型容量下只改变时间信息的实验。相同学习率未必分别最适合两种表示，本次未额外搜索超参数。',
        '', '固定seed不意味着不同维度模型的初始参数或具体打乱顺序完全相同。两者都按同一随机化规则训练，未重复多个种子。验证集用于早停和本次配置比较，不是独立测试结果，不声称统计显著性或因果结论。',
        '', '保存位置：runs/E3_EEG_label100_seed0_timeflatten_v01；原基线：runs/E3_EEG_label100_seed0_provisional_v01。Linux磁盘缓存为~/.cache/eegdm/timeflatten_v01，约23.67GB，可重建；完整实验结果保存在D盘。']
    (OUT/'时间池化对比报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
