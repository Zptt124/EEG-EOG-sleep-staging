"""Disk-streamed time-preserving linear probe, paired with mean-pooling v01."""
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
FEATURE=ROOT/'features/eeg_cond14_v01'
BASE=ROOT/'runs/E3_EEG_label100_seed0_provisional_v01'
OUT=ROOT/'runs/E3_EEG_label100_seed0_timeflatten_v01'
CACHE=Path.home()/'.cache/eegdm/timeflatten_v01'
DIM=20*30*128

def dump(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)

def log(**kw):
    print(json.dumps(kw,ensure_ascii=False),flush=True)
    dump(OUT/'progress.json',kw)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():
        raise RuntimeError('Completed run exists; do not overwrite')
    CACHE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    with (FEATURE/'feature_index.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    summary=json.loads((FEATURE/'summary.json').read_text())
    assert summary['split_counts']=={'train':132938,'val':21186}
    assert len(rows)==154124 and len({r['sample_id'] for r in rows})==len(rows)
    assert {r['split'] for r in rows}=={'train','val'}
    parts={s:[r for r in rows if r['split']==s] for s in ('train','val')}
    basecfg=json.loads((BASE/'run_config.json').read_text())
    for s,k in [('train','training_subjects'),('val','validation_subjects')]:
        assert sorted({r['subject_id'] for r in parts[s]})==basecfg[k]
    cfg=dict(basecfg)
    cfg.update(pooling='none; flatten [layer,time_window,hidden] in C order',input_dim=DIM,
        classifier='single affine Linear(76800,5), no hidden layer',parameter_count=DIM*5+5,
        baseline_parameter_count=2560*5+5,device='cpu',storage='float16 raw Linux mmap; float32 batch arithmetic',
        comparison='Same seed, optimizer, LR, weight decay, batch size, max epochs and patience as mean baseline; dimension and per-position normalization differ')
    dump(OUT/'run_config.json',cfg)
    signature=dict(index_sha256=hashlib.sha256((FEATURE/'feature_index.csv').read_bytes()).hexdigest(),
        feature_config=json.loads((FEATURE/'feature_config.json').read_text()),dim=DIM)
    if (CACHE/'complete.json').exists():
        assert json.loads((CACHE/'complete.json').read_text())==signature
    else:
        log(stage='prepare_disk_cache',samples=0,total=len(rows))
        count=0
        for partition,rr in parts.items():
            dest=np.lib.format.open_memmap(CACHE/f'{partition}.npy',mode='w+',dtype=np.float16,shape=(len(rr),DIM))
            grouped={}
            for i,r in enumerate(rr):
                grouped.setdefault(r['feature_path'],[]).append((i,int(r['row_index'])))
            for name,entries in grouped.items():
                src=np.load(FEATURE/name,mmap_mode='r',allow_pickle=False)
                assert src.shape[1:]==(20,30,128) and src.dtype==np.float16
                for offset in range(0,len(entries),64):
                    block=entries[offset:offset+64]
                    value=src[[j for _,j in block]]
                    assert np.isfinite(value).all()
                    dest[[i for i,_ in block]]=value.reshape(len(block),DIM)
                count+=len(entries)
                log(stage='prepare_disk_cache',samples=count,total=len(rows))
                del src
            dest.flush()
            del dest
        dump(CACHE/'complete.json',signature)
    arrays={s:np.load(CACHE/f'{s}.npy',mmap_mode='r',allow_pickle=False) for s in parts}
    labels={s:np.array([int(r['y_true']) for r in rr],dtype=np.int64) for s,rr in parts.items()}
    for s in parts:
        assert arrays[s].shape==(len(parts[s]),DIM)
        assert set(labels[s])==set(range(5))
    if not (CACHE/'normalizer.npz').exists():
        log(stage='fit_training_normalizer')
        # Merge per-block moments in float64: no validation information is used.
        mean=np.zeros(DIM,np.float64)
        m2=np.zeros(DIM,np.float64)
        count=0
        for start in range(0,len(arrays['train']),128):
            data=arrays['train'][start:start+128].astype(np.float64)
            n=len(data)
            bm=data.mean(0)
            delta=bm-mean
            m2+=((data-bm)**2).sum(0)+delta**2*count*n/(count+n)
            mean+=delta*n/(count+n)
            count+=n
        std=np.maximum(np.sqrt(m2/count).astype(np.float32),1e-6)
        mean=mean.astype(np.float32)
        np.savez(CACHE/'normalizer.npz',mean=mean,std=std)
    with np.load(CACHE/'normalizer.npz') as z:
        mean=torch.from_numpy(z['mean'].copy())
        std=torch.from_numpy(z['std'].copy())
    assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>0).all()

    def batch(partition,ids):
        value=np.asarray(arrays[partition][ids],dtype=np.float32)
        return (torch.from_numpy(value)-mean)/std

    def evaluate(model):
        with torch.inference_mode():
            return torch.cat([model(batch('val',slice(i,i+128))).softmax(1)
                              for i in range(0,len(parts['val']),128)]).numpy()

    model=torch.nn.Linear(DIM,5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    y=torch.from_numpy(labels['train'])
    history=[]
    best=-1.
    stale=0
    epoch0=1
    if (OUT/'last_state.pt').exists():
        state=torch.load(OUT/'last_state.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
        history,best,stale=state['history'],state['best'],state['stale']
        epoch0=state['epoch']+1
    for epoch in range(epoch0,51):
        if stale>=10:
            break
        start=time.perf_counter()
        model.train()
        order=torch.randperm(len(y))
        total=0.
        for ids in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch('train',ids.numpy()))
            loss=torch.nn.functional.cross_entropy(logits,y[ids])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total+=loss.item()*len(ids)
        model.eval()
        probs=evaluate(model)
        score=float(f1_score(labels['val'],probs.argmax(1),labels=range(5),average='macro',zero_division=0))
        item=dict(epoch=epoch,train_loss=total/len(y),val_macro_f1=score,seconds=time.perf_counter()-start)
        history.append(item)
        if score>best:
            best,stale=score,0
            torch.save(dict(model=model.state_dict(),mean=mean,std=std,epoch=epoch,config=cfg),OUT/'best_head.pt')
        else:
            stale+=1
        dump(OUT/'history.json',history)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
            epoch=epoch,history=history,best=best,stale=stale),OUT/'last_state.tmp.pt')
        (OUT/'last_state.tmp.pt').replace(OUT/'last_state.pt')
        log(stage='training',**item,best_macro_f1=best,patience_used=stale)
    ckpt=torch.load(OUT/'best_head.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    probs=evaluate(model)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-6)
    pred=probs.argmax(1)
    spec=importlib.util.spec_from_file_location('a_metrics',ROOT.parent/'scripts/metrics.py')
    metric=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    result=metric.compute_metrics(labels['val'],pred,names=['W','N1','N2','N3','REM'])
    assert abs(result['macro_f1']-best)<1e-12
    dump(OUT/'metrics_provisional.json',result)
    with (OUT/'predictions_val.csv').open('w',encoding='utf-8',newline='') as f:
        keys=['sample_id','subject_id','record_id','epoch_index','y_true']
        pkeys=['p_W','p_N1','p_N2','p_N3','p_REM']
        writer=csv.DictWriter(f,fieldnames=keys+['y_pred']+pkeys)
        writer.writeheader()
        for i,r in enumerate(parts['val']):
            item={k:r[k] for k in keys}
            item.update(y_pred=int(pred[i]))
            item.update(dict(zip(pkeys,map(float,probs[i]))))
            writer.writerow(item)
    dump(OUT/'summary.json',dict(status='complete',provisional=True,test_used=False,
        train_samples=len(y),val_samples=len(pred),best_epoch=ckpt['epoch'],val_macro_f1=best,
        epochs=len(history),training_seconds=sum(r['seconds'] for r in history),parameter_count=DIM*5+5))
    log(stage='complete',best_epoch=ckpt['epoch'],val_macro_f1=best)

if __name__=='__main__':
    main()
