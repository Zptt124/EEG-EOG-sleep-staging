"""Training-only engineering checks, no classifier or validation-based selection."""
import json
import time
import numpy as np
import torch
from extract_features import ROOT, DATA, Extractor, record_splits, transform, config, write_csv

out=ROOT/'reports/feature_pilot'
out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(0)
rng=np.random.default_rng(0)
signals=[]
metadata=[]
train=[r for r,s in sorted(record_splits().items()) if s=='train']
# Two nights maximum per subject; deliberately select six distinct subjects.
chosen=[]
for rec in train:
    if rec[:5] not in [r[:5] for r in chosen]:
        chosen.append(rec)
    if len(chosen)==6:
        break
for rec in chosen:
    with np.load(DATA/f'{rec}.npz',allow_pickle=False) as z:
        x,y,idx=z['signal'][:,0],z['labels'],z['epoch_index']
        for label in range(5):
            candidates=np.flatnonzero(y==label)
            selection=rng.choice(candidates,min(10,len(candidates)),replace=False)
            for j in sorted(selection):
                signals.append(x[j])
                metadata.append(dict(sample_id=f'{rec}_{int(idx[j]):04d}',subject_id=rec[:5],
                                     record_id=rec,epoch_index=int(idx[j]),y_true=int(y[j]),split='train'))
signals=np.stack(signals)
write_csv(out/'pilot_samples.csv',metadata)
prepared=transform(signals)
profiles=[]
for condition in (14,18,10):
    model=Extractor(condition)
    warm=model(signals[:1])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start=time.perf_counter()
    features=np.concatenate([model(signals[b:b+4]) for b in range(0,len(signals),4)])
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    native=features[:2]
    explicit=model(signals[:2],mode='explicit')
    repeated=model(signals[:2])
    assert np.allclose(native,explicit,atol=1e-5,rtol=1e-4)
    assert np.allclose(native,repeated,atol=1e-5,rtol=1e-4)
    restored=features.astype(np.float16).astype(np.float32)
    assert np.isfinite(restored).all() and features.std()>0
    assert np.all(np.std(features.reshape(len(features),-1),axis=1)>0)
    np.save(out/f'features_cond{condition}.npy',features.astype(np.float16))
    profile=dict(condition=condition,n_samples=len(signals),batch_size=4,
        seconds=elapsed,samples_per_second=len(signals)/elapsed,
        peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20,
        features_finite=bool(np.isfinite(features).all()),feature_std=float(features.std()),
        native_vs_explicit_max_abs=float(np.max(np.abs(native-explicit))),
        repeat_max_abs=float(np.max(np.abs(native-repeated))),
        float16_max_abs_error=float(np.max(np.abs(features-restored))),
        float16_relative_l2_error=float(np.linalg.norm(features-restored)/np.linalg.norm(features)),
        feature_bytes_per_sample=20*30*128*2,
        estimated_full_hours=154124/(len(signals)/elapsed)/3600,
        estimated_full_gb=154124*20*30*128*2/1e9)
    profiles.append(profile)
    print(json.dumps(profile),flush=True)
    del model,features,restored,native,explicit,repeated
    torch.cuda.empty_cache()
write_csv(out/'resource_profile.csv',profiles)
summary=dict(status='passed',subjects=chosen,n_samples=len(signals),
    class_counts={str(i):sum(r['y_true']==i for r in metadata) for i in range(5)},
    input_shape=list(signals.shape),transformed_shape=list(prepared.shape),
    original_unit='uV',transformed_min=float(prepared.min()),transformed_max=float(prepared.max()),
    conditions_tested=[14,18,10],primary_condition=14,
    selection_basis='Preselected engineering default FP1-F3; no claim of best accuracy',
    test_used=False,validation_used=False,config=config('eeg',14))
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,1,figsize=(12,5),layout='constrained')
ax[0].plot(np.arange(3000)/100,signals[0],lw=.5)
ax[0].set(ylabel='EEG (uV)',title=metadata[0]['sample_id']+' | raw 100 Hz')
ax[1].plot(np.arange(6000)/200,prepared[0,0],lw=.5)
ax[1].set(xlabel='Time (s)',ylabel='Model input',title='200 Hz; /100; staged mu-law')
fig.savefig(out/'conversion_check.png',dpi=150)
plt.close(fig)
