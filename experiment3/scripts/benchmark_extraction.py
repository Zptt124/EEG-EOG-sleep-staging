import json,time
import numpy as np
import torch
from extract_features import ROOT,DATA,Extractor
with np.load(DATA/'SC4001.npz',allow_pickle=False) as z:
    x=z['signal'][:64,0]
m=Extractor(14)
baseline=m(x[:4])
m.cache_frozen_kernels()
cached=m(x[:4])
assert np.allclose(baseline,cached,atol=1e-6,rtol=1e-5)
rows=[]
for bs in [4,16,32]:
    try:
        m(x[:bs])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start=time.perf_counter()
        feats=np.concatenate([m(x[i:i+bs]) for i in range(0,len(x),bs)])
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        assert np.allclose(baseline,feats[:4],atol=1e-5,rtol=1e-4)
        row=dict(batch_size=bs,seconds=elapsed,samples_per_second=len(x)/elapsed,
                 peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20,
                 estimated_full_hours=154124/(len(x)/elapsed)/3600,
                 difference_from_uncached=float(np.max(np.abs(baseline-feats[:4]))))
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        row=dict(batch_size=bs,error='OOM')
    rows.append(row)
    print(json.dumps(row),flush=True)
(ROOT/'reports/feature_pilot/optimized_benchmark.json').write_text(json.dumps(rows,indent=2))
