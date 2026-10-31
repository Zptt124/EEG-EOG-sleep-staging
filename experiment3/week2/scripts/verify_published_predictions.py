"""Verify the published C W2 predictions with the byte-identical B v1 evaluator.

Uses no model, original signal, feature array, test data or new model selection.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SHA = '4168c650b154f2c2a2c4ff26e8acd8466a41d3f6823dbaeabfa6eb07ab7f7605'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root=ROOT):
    evaluator = root/'evaluation_reference/evaluate.py'
    require(sha(evaluator) == EVALUATOR_SHA, 'B evaluator SHA mismatch')
    spec = importlib.util.spec_from_file_location('official_b_w2_publication', evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    report = root/'reports/week2_final_C_20261029'
    registry = list(csv.DictReader((report/'run_registry.csv').open(encoding='utf-8-sig', newline='')))
    require(len(registry) == 80 and len({r['run_id'] for r in registry}) == 80, 'Expected 80 unique runs')
    index = pd.read_csv(root/'features/eeg_cond14_v01/feature_index.csv')
    identity = ['sample_id', 'subject_id', 'record_id', 'epoch_index', 'y_true']
    canonical = index.loc[index['split']=='val', identity].reset_index(drop=True)
    require(len(canonical) == 21186 and canonical.subject_id.nunique() == 8, 'Wrong validation set')
    checked, positions = [], set()
    for row in registry:
        run = row['run_id']
        require(Path(run).name == run and run.startswith('W2_C_'), 'Unsafe run ID')
        folder = root/'runs'/run
        artifacts = json.loads((folder/'artifact_sha256.json').read_text(encoding='utf-8'))
        verified = 0
        for name, expected in artifacts.items():
            require(Path(name).name == name, 'Unsafe artifact path')
            if name.endswith(('.pt', '.ckpt')):
                continue  # Model files are explicitly external to this publication.
            require(sha(folder/name) == expected, f'Artifact SHA mismatch: {run}/{name}')
            verified += 1
        predictions = official.load_predictions(folder/'predictions_val.csv')
        require(predictions[identity].equals(canonical), f'Validation identity mismatch: {run}')
        require(np.isfinite(predictions[official.PROB_COLS].to_numpy()).all(), f'Non-finite probability: {run}')
        measured = official.compute_metrics(predictions.y_true.to_numpy(), predictions.y_pred.to_numpy())
        stored = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        require(all(stored[key] == value for key, value in measured.items()), f'Official point metrics differ: {run}')
        require(measured['macro_f1'] == float(row['macro_f1']) and measured['n1_f1'] == float(row['n1_f1']), f'Registry metrics differ: {run}')
        position = (row['modality'], row['budget'], row['adaptation_seed'], row['downstream_seed'])
        require(position not in positions, 'Duplicate matrix cell')
        positions.add(position)
        checked.append(dict(run_id=run, uploaded_artifact_SHA_verified=verified,
                            validation_samples=len(predictions), official_point_metrics_match=True))
    expected = {(m,b,a,str(s)) for m in ('EEG','EEG_EOG') for b in ('label10','label25')
                for a in ('','0','1','2') for s in range(5)}
    require(positions == expected, 'Incomplete experiment matrix')
    return dict(status='passed', scope='uploaded_predictions_and_text_artifacts_only',
                runs=80, validation_samples_per_run=21186, validation_subjects=8,
                independent_adaptation_seeds=[0,1,2], models_loaded=False,
                raw_signals_loaded=False, test_used=False, B_evaluator_sha256=EVALUATOR_SHA,
                team_acceptance_complete=False, checks=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new audit JSON; existing files are never overwritten')
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError('Output already exists; choose a new file')
    result = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
