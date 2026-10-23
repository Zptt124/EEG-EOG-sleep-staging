"""Verify the published five prediction tables against B's official evaluator."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluator',type=Path,default=ROOT/'vendor/experiment2-review-6e7df99/src/evaluation/evaluate.py')
    args=parser.parse_args()
    if not args.evaluator.exists():
        raise RuntimeError('Run python experiment3/scripts/prepare_sources.py first')
    spec=importlib.util.spec_from_file_location('b_official',args.evaluator)
    ev=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    index=pd.read_csv(ROOT/'manifests/eeg_cond14_v01/feature_index.csv')
    expected=index[index.split=='val'].sort_values('sample_id').reset_index(drop=True)
    assert len(index)==154124 and index.sample_id.is_unique and len(expected)==21186
    result=[]
    for file in sorted((ROOT/'reports/predictions_E3').glob('*.csv')):
        run=file.stem.removeprefix('predictions_val_')
        df=ev.load_predictions(file).sort_values('sample_id').reset_index(drop=True)
        assert np.isfinite(df[ev.PROB_COLS].to_numpy()).all()
        for column in ('sample_id','subject_id','record_id','epoch_index','y_true'):
            assert df[column].equals(expected[column]),(run,column)
        m=ev.compute_metrics(df.y_true,df.y_pred)
        saved=json.loads((ROOT/'reports/runs_E3'/run/'metrics.json').read_text())
        assert np.isclose(m['macro_f1'],saved['macro_f1'],rtol=0,atol=1e-12)
        assert m['confusion_matrix']==saved['confusion_matrix']
        for key in ('per_class_f1','per_class_precision','per_class_recall','per_class_support'):
            assert all(np.isclose(v,saved[key][name],rtol=0,atol=1e-12) for name,v in m[key].items())
        cfg=json.loads((ROOT/'reports/runs_E3'/run/'run_config.json').read_text())
        assert set(cfg['validation_subjects'])==set(expected.subject_id)
        assert not set(cfg['training_subjects']) & set(expected.subject_id)
        if cfg['label_fraction']==.25:
            seed=cfg['sampling_seed']
            name='label_subsets_v1.csv' if seed==2024 else f'label_subsets_v1-s{seed}.csv'
            subset=pd.read_csv(args.evaluator.parents[2]/'manifests'/name)
            assert set(cfg['training_subjects'])==set(subset.loc[subset.label25==1,'subject_id'])
        result.append(dict(run_id=run,status='passed',samples=len(df),macro_f1=m['macro_f1']))
    assert len(result)==5
    print(json.dumps(dict(status='passed',runs=result,test_used=False),indent=2))

if __name__=='__main__':
    main()
