"""Package the existing unstandardized original EEG/EOG pools for B.

NPZ keys are exactly sample_id (Unicode) and x (float32). No feature recompute,
normalization, raw signal access or model fitting is performed.
"""
import csv
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliveries/B_pooled_features_20261031_v01'
REPORT=ROOT/'reports/B_pooled_features_20261031_v01'
SOURCES=[
    dict(modality='EEG',representation='S0_original_frozen',
         pool='reports/week2_eeg_comparison_20261028/execution/S0_pooled.npy',
         audit='reports/week2_eeg_comparison_20261028/execution/S0_feature_audit.json',
         features='features/eeg_cond14_v01',
         filename='EEG_S0_pooled_154124x2560_unstandardized_v01.npz'),
    dict(modality='EOG',representation='S0_original_frozen_C_local_reconstruction',
         pool='reports/week2_fusion_20261029/execution/EOG_audit/S0_pooled.npy',
         audit='reports/week2_fusion_20261029/execution/EOG_audit/S0_feature_audit.json',
         features='features/eog_cond14_local_Dprotocol_v01',
         filename='EOG_C_rebuilt_S0_pooled_154124x2560_unstandardized_v01.npz')]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    REPORT.mkdir(parents=True,exist_ok=False)
    status=dict(status='packaging',completed_files=0,expected_files=2,normalization_applied=False,test_used=False)
    write_json(REPORT/'status.json',status)
    manifest=dict(schema_version=1,release_tag='c-pooled-features-v01-20261031',
        requested_by='B via user',EEG_is_original_S0_not_adapted_S1=True,
        shape=[154124,2560],dtype='float32',keys=['sample_id','x'],normalization_applied=False,
        feature_normalization='none; downstream statistics must be fitted on permitted labeled training subjects only',
        pooling='raw float16 gate [N,20,30,128] converted to float32, mean over 30 time windows, flatten layer-major',
        train_samples=132938,validation_samples=21186,test_samples=0,train_subjects=54,validation_subjects=8,
        original_checkpoint_sha256='0fe2d385d074834f3c22d9458798da49b2e5a58bd817284f130aae8814f5c1ef',
        EOG_source='C local regeneration using D original frozen reference protocol',
        D_full_cache_received=False,D_raw_file_byte_equivalence_unverified=True,
        B_receipt_or_acceptance_confirmed=False,packaging_script_sha256=sha(Path(__file__)),files=[])
    canonical=None
    for source in SOURCES:
        began=time.perf_counter()
        audit=json.loads((ROOT/source['audit']).read_text(encoding='utf-8'))
        assert audit['status']=='passed' and audit['raw_all_finite'] and not audit['test_used']
        folder=ROOT/source['features'];pool=ROOT/source['pool']
        assert sha(pool)==audit['pooled_sha256'], 'Source pool SHA mismatch'
        assert sha(folder/'feature_index.csv')==audit['source_index_sha256']
        assert sha(folder/'feature_config.json')==audit['source_config_sha256']
        with (folder/'feature_index.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        identity=[tuple(r[k] for k in ['sample_id','subject_id','record_id','epoch_index','split','y_true']) for r in rows]
        if canonical is None:
            canonical=identity
            with (OUT/'sample_index.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.writer(f);writer.writerow(['sample_id','subject_id','record_id','epoch_index','split','y_true']);writer.writerows(identity)
        else:
            assert identity==canonical,'EEG and EOG sample identities differ'
        ids=np.asarray([r['sample_id'] for r in rows])
        assert ids.dtype.kind=='U' and ids.shape==(154124,) and len(np.unique(ids))==154124
        assert {r['split'] for r in rows}=={'train','val'}
        assert sum(r['split']=='train' for r in rows)==132938
        assert sum(r['split']=='val' for r in rows)==21186
        x=np.load(pool,mmap_mode='r',allow_pickle=False)
        assert x.shape==(154124,2560) and x.dtype==np.float32 and x.flags.c_contiguous
        status.update(current_modality=source['modality'],stage='write_npz',rows_written=0)
        write_json(REPORT/'status.json',status)
        partial=OUT/(source['filename']+'.partial')
        with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            with archive.open('sample_id.npy','w',force_zip64=True) as f:
                np.lib.format.write_array(f,ids,allow_pickle=False)
            with archive.open('x.npy','w',force_zip64=True) as f:
                np.lib.format.write_array_header_2_0(f,dict(descr=np.lib.format.dtype_to_descr(x.dtype),fortran_order=False,shape=x.shape))
                for start in range(0,len(x),4096):
                    block=x[start:start+4096]
                    assert np.isfinite(block).all(),'Nonfinite pool values'
                    f.write(memoryview(block).cast('B'))
                    status['rows_written']=min(start+4096,len(x));write_json(REPORT/'status.json',status)
                    print(json.dumps(dict(stage='write_npz',modality=source['modality'],rows=status['rows_written'],total=len(x))),flush=True)
        target=OUT/source['filename'];partial.replace(target)
        status['stage']='reload_and_bitwise_verify';write_json(REPORT/'status.json',status)
        with np.load(target,allow_pickle=False) as packed:
            assert set(packed.files)=={'sample_id','x'}
            actual_ids=packed['sample_id'];actual_x=packed['x']
            assert actual_x.shape==(154124,2560) and actual_x.dtype==np.float32
            assert actual_ids.dtype.kind=='U' and np.array_equal(actual_ids,ids)
            for start in range(0,len(x),4096):
                assert np.array_equal(actual_x[start:start+4096],x[start:start+4096]),'Packed x differs from unstandardized source'
            del actual_x
        assert target.stat().st_size<2*1024**3,'Release asset exceeds GitHub 2GiB bound'
        record=dict(source,bytes=target.stat().st_size,sha256=sha(target),shape=list(x.shape),dtype=str(x.dtype),
                    sample_id_dtype=str(ids.dtype),unique_sample_ids=154124,
                    source_pooled_sha256=audit['pooled_sha256'],source_index_sha256=audit['source_index_sha256'],
                    source_config_sha256=audit['source_config_sha256'],
                    roundtrip_x_bitwise_equal=True,roundtrip_sample_id_order_exact=True,
                    finite=True,normalization_applied=False,packaging_seconds=time.perf_counter()-began)
        manifest['files'].append(record)
        write_json(OUT/'manifest.json',manifest)
        status['completed_files']+=1;write_json(REPORT/'status.json',status)
        print(json.dumps(dict(stage='packed_verified',modality=source['modality'],bytes=record['bytes'],sha256=record['sha256'])),flush=True)
        del x
    manifest['sample_index_sha256']=sha(OUT/'sample_index.csv')
    write_json(OUT/'manifest.json',manifest)
    (OUT/'SHA256SUMS.txt').write_text(''.join(f"{sha(p)}  {p.name}\n" for p in sorted(OUT.iterdir()) if p.is_file()),encoding='utf-8')
    status.update(status='packaged_and_verified',stage='ready_for_release_upload',completed_files=2)
    write_json(REPORT/'status.json',status)


if __name__=='__main__':main()
