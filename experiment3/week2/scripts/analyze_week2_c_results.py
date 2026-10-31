"""Read-only W2 C validation audit, N1 error analysis and reproducible delivery.

No fitting, new model selection, test signals or external messages. Predictions
from all three preregistered backbones and both modalities are retained.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import run_s0_s1_eeg_comparison as heads

ROOT = heads.ROOT
OUT = ROOT/'reports/week2_final_C_20261029'
SOURCES = [heads.REPORT/'execution', ROOT/'reports/week2_seed1_features_heads_20261028_v04/execution',
           ROOT/'reports/week2_seed2_features_heads_20261029/execution', ROOT/'reports/week2_fusion_20261029/execution']
require, sha, write_json = heads.require, heads.sha, heads.write_json


def changes(y, baseline, adapted):
    y, b, a = map(np.asarray, (y, baseline, adapted))
    require(y.shape == b.shape == a.shape and y.ndim == 1, 'Prediction alignment shape differs')
    require(all(np.isin(v, range(5)).all() for v in (y, b, a)), 'Stage outside 0..4')
    n1 = y == 1
    return dict(samples=len(y), N1_support=int(n1.sum()),
                recovered_N1=int((n1 & (b != 1) & (a == 1)).sum()),
                lost_N1=int((n1 & (b == 1) & (a != 1)).sum()),
                N1_false_positive_removed=int((~n1 & (b == 1) & (a != 1)).sum()),
                N1_false_positive_added=int((~n1 & (b != 1) & (a == 1)).sum()),
                exact_stage_recovered=int(((b != y) & (a == y)).sum()),
                exact_stage_lost=int(((b == y) & (a != y)).sum()),
                changed_predictions=int((b != a).sum()))


def context_groups(rows):
    lookup = {(r['record_id'], int(r['epoch_index'])): int(r['y_true']) for r in rows}
    require(len(lookup) == len(rows), 'Duplicate record/epoch in context metadata')
    result = []
    for r in rows:
        rec, epoch, label = r['record_id'], int(r['epoch_index']), int(r['y_true'])
        prior, following = lookup.get((rec, epoch-1)), lookup.get((rec, epoch+1))
        if prior is None or following is None:
            result.append('gap_or_record_edge')
        elif prior != label or following != label:
            result.append('label_transition')
        else:
            result.append('stable_neighbors')
    return np.array(result)


def output_csv(name, rows):
    if rows:
        heads.write_csv(OUT/name, rows)


def input_registry():
    result = {}
    for source in SOURCES:
        for row in heads.read_csv(source/'run_registry.csv'):
            result[row['run_id']] = row
    require(len(result) == 80, 'Expected 80 unique W2 classifier runs')
    return result


def analyze():
    OUT.mkdir(parents=True, exist_ok=False)
    official = heads.official_module()
    canonical = [r for r in heads.read_csv(heads.SOURCES['S0']/'feature_index.csv') if r['split'] == 'val']
    expected = [(r['sample_id'], r['subject_id'], r['record_id'], int(r['epoch_index']), int(r['y_true'])) for r in canonical]
    y = np.array([int(r['y_true']) for r in canonical], dtype=np.int8)
    subjects = np.array([r['subject_id'] for r in canonical])
    contexts = context_groups(canonical)
    subsets = heads.read_csv(heads.SUBSETS)
    all_index = heads.read_csv(heads.SOURCES['S0']/'feature_index.csv')
    cache, run_rows, subject_rows, checks = {}, [], [], []
    registry = input_registry()
    for run_id, listed in registry.items():
        folder = ROOT/'runs'/run_id
        hashes = json.loads((folder/'artifact_sha256.json').read_text())
        require(all(Path(name).name == name and sha(folder/name) == digest for name, digest in hashes.items()),
                f'Run artifact checksum mismatch: {run_id}')
        cfg = yaml.safe_load((folder/'config.yaml').read_text())
        summary = json.loads((folder/'summary.json').read_text())
        require(summary['status'] == 'complete' and not cfg['test_used'], 'Incomplete/forbidden run')
        budget, seed, rep = cfg['budget'], int(cfg['downstream_seed']), cfg['representation']
        modality, adaptation = cfg['modality'], cfg['adaptation_seed']
        require(modality in ('EEG', 'EEG_EOG') and rep in ('S0', 'S1'), 'Unexpected run mode')
        require((rep == 'S0' and adaptation is None) or (rep == 'S1' and adaptation in (0,1,2)), 'Wrong backbone identity')
        selected, train, val = heads.selected_positions(all_index, subsets, budget)
        require(set(cfg['training_subjects']) == selected and cfg['training_samples'] == len(train), 'Budget metadata differs')
        require(cfg['batch_size'] == 128 and cfg['max_epochs'] == 50 and cfg['patience'] == 10
                and cfg['learning_rate'] == .001 and cfg['weight_decay'] == .0001, 'Classifier settings differ')
        checkpoint = torch.load(folder/'best_head.pt', map_location='cpu', weights_only=True)
        history = json.loads((folder/'history.json').read_text())
        require(checkpoint['config'] == cfg and checkpoint['epoch'] == summary['best_epoch'], 'Checkpoint configuration/epoch mismatch')
        require(checkpoint['epoch'] == max(history, key=lambda h: h['val_macro_f1'])['epoch'], 'Strict best-epoch selection differs')
        dimension = 2560 if modality == 'EEG' else 5120
        require(tuple(checkpoint['model']['weight'].shape) == (5, dimension), 'Classifier dimension mismatch')
        require(torch.isfinite(checkpoint['mean']).all() and torch.isfinite(checkpoint['std']).all()
                and (checkpoint['std'] >= 1e-6).all(), 'Invalid saved normalization')
        frame = official.load_predictions(folder/'predictions_val.csv')
        actual = list(frame[['sample_id','subject_id','record_id','epoch_index','y_true']].itertuples(index=False,name=None))
        require(actual == expected and np.isfinite(frame[official.PROB_COLS].to_numpy()).all(), 'Prediction identity/probability mismatch')
        prediction = frame['y_pred'].to_numpy(dtype=np.int8)
        metrics = official.compute_metrics(y, prediction)
        stored = json.loads((folder/'metrics.json').read_text())
        require(metrics['confusion_matrix'] == stored['confusion_matrix'] and metrics['macro_f1'] == float(listed['val_macro_f1']),
                'Official point metrics differ')
        key = (modality, budget, adaptation, seed)
        require(key not in cache, 'Duplicate experimental position')
        metadata = dict(run_id=run_id, modality=modality, budget=budget, representation=rep,
                        adaptation_seed=adaptation, downstream_seed=seed)
        row = dict(metadata, macro_f1=metrics['macro_f1'], n1_precision=metrics['n1_precision'],
                   n1_recall=metrics['n1_recall'], n1_f1=metrics['n1_f1'], rem_f1=metrics['per_class_f1']['REM'],
                   accuracy=metrics['accuracy'], best_epoch=checkpoint['epoch'], epochs=len(history),
                   head_seconds=summary['seconds'], train_subjects=len(selected), train_samples=len(train),
                   validation_samples=len(y), local_unlabeled_adaptation_subjects=54 if rep=='S1' else 0,
                   test_used=False, predictions_path=(folder/'predictions_val.csv').relative_to(ROOT).as_posix())
        run_rows.append(row)
        cache[key] = dict(metadata=metadata, metrics=row, pred=prediction, pN1=frame['p_N1'].to_numpy(dtype=np.float32))
        for subject in sorted(set(subjects)):
            mask = subjects == subject
            m = official.compute_metrics(y[mask], prediction[mask])
            subject_rows.append(dict(metadata, subject_id=subject, samples=int(mask.sum()), macro_f1=m['macro_f1'],
                                     n1_f1=m['n1_f1'], n1_support=m['per_class_support']['N1'], rem_f1=m['per_class_f1']['REM']))
        checks.append(dict(run_id=run_id, artifact_files_verified=len(hashes), config_and_checkpoint_match=True,
                           strict_selection_verified=True, predictions_match_A_validation=True, official_metrics_recomputed=True))
        print(json.dumps(dict(stage='audit_predictions', completed=len(checks), total=80, run_id=run_id)), flush=True)
    require(set(cache) == {(m,b,a,s) for m in ('EEG','EEG_EOG') for b in ('label10','label25')
                           for a in (None,0,1,2) for s in range(5)}, 'Experimental matrix incomplete')
    output_csv('run_registry.csv', run_rows)
    output_csv('subject_metrics.csv', subject_rows)
    pairs, subject_pairs, directions, context_rows, changed_rows, comparisons = [], [], [], [], [], []
    for modality in ('EEG', 'EEG_EOG'):
        for budget in ('label10','label25'):
            for adaptation in range(3):
                for seed in range(5):
                    base, adapted = cache[(modality,budget,None,seed)], cache[(modality,budget,adaptation,seed)]
                    b, a = base['pred'], adapted['pred']
                    metadata = dict(modality=modality, budget=budget, adaptation_seed=adaptation, downstream_seed=seed,
                                    S0_run_id=base['metadata']['run_id'], S1_run_id=adapted['metadata']['run_id'])
                    stats = changes(y,b,a)
                    delta = {f'delta_{k}_pp': 100*(adapted['metrics'][k]-base['metrics'][k])
                             for k in ('macro_f1','n1_f1','rem_f1')}
                    pair = dict(metadata, **delta, S0_best_epoch=base['metrics']['best_epoch'],
                                S1_best_epoch=adapted['metrics']['best_epoch'], **stats)
                    pairs.append(pair)
                    for subject in sorted(set(subjects)):
                        mask = subjects == subject
                        m0, m1 = official.compute_metrics(y[mask],b[mask]), official.compute_metrics(y[mask],a[mask])
                        subject_pairs.append(dict(metadata, subject_id=subject,
                            delta_macro_f1_pp=100*(m1['macro_f1']-m0['macro_f1']),
                            delta_n1_f1_pp=100*(m1['n1_f1']-m0['n1_f1']), **changes(y[mask],b[mask],a[mask])))
                    for stage in (0,2,3,4):
                        for kind, mask, old, new in [('N1_false_negative_destination', y==1, b==stage, a==stage),
                                                     ('N1_false_positive_true_stage', y==stage, b==1, a==1)]:
                            old_count, new_count = int((mask&old).sum()), int((mask&new).sum())
                            directions.append(dict(metadata, direction=kind, other_stage=official.STAGE_NAMES[stage],
                                                   S0_count=old_count, S1_count=new_count, delta_count=new_count-old_count,
                                                   denominator=int(mask.sum())))
                    for category in sorted(set(contexts)):
                        mask = contexts == category
                        context_rows.append(dict(metadata, context=category, **changes(y[mask],b[mask],a[mask])))
                    labels = np.full(len(y), '', dtype=object)
                    labels[(y==1)&(b!=1)&(a==1)] = 'recovered_N1'
                    labels[(y==1)&(b==1)&(a!=1)] = 'lost_N1'
                    labels[(y!=1)&(b==1)&(a!=1)] = 'N1_false_positive_removed'
                    labels[(y!=1)&(b!=1)&(a==1)] = 'N1_false_positive_added'
                    for i in np.flatnonzero(labels != ''):
                        r = canonical[i]
                        changed_rows.append(dict(metadata, sample_id=r['sample_id'], subject_id=r['subject_id'],
                            record_id=r['record_id'], epoch_index=int(r['epoch_index']), y_true=int(y[i]),
                            S0_y_pred=int(b[i]), S1_y_pred=int(a[i]), S0_p_N1=float(base['pN1'][i]),
                            S1_p_N1=float(adapted['pN1'][i]), change_type=labels[i], context=contexts[i]))
                group = [r for r in run_rows if r['modality']==modality and r['budget']==budget]
                s0 = [r for r in group if r['adaptation_seed'] is None]
                s1 = [r for r in group if r['adaptation_seed']==adaptation]
                value = dict(modality=modality,budget=budget,adaptation_seed=adaptation,downstream_seeds=5)
                for metric in ('macro_f1','n1_f1','n1_precision','n1_recall','rem_f1'):
                    v0,v1=[r[metric] for r in s0],[r[metric] for r in s1]
                    value.update({f'S0_{metric}_mean':float(np.mean(v0)),f'S1_{metric}_mean':float(np.mean(v1)),
                                  f'S0_{metric}_head_sd':float(np.std(v0,ddof=1)),f'S1_{metric}_head_sd':float(np.std(v1,ddof=1)),
                                  f'delta_{metric}_pp':float(100*(np.mean(v1)-np.mean(v0)))})
                comparisons.append(value)
    output_csv('paired_run_changes.csv',pairs)
    output_csv('per_subject_paired.csv',subject_pairs)
    output_csv('n1_error_direction.csv',directions)
    output_csv('error_by_context.csv',context_rows)
    output_csv('changed_N1_epochs.csv',changed_rows)
    output_csv('adapt_comparison.csv',comparisons)
    exemplars = {}
    for row in sorted(changed_rows,key=lambda r:(r['downstream_seed'],r['sample_id'])):
        key=(row['modality'],row['budget'],row['adaptation_seed'],row['change_type'])
        exemplars.setdefault(key,row)
    output_csv('deterministic_case_examples.csv',list(exemplars.values()))
    worst=min([p for p in pairs if p['modality']=='EEG_EOG'],key=lambda p:p['delta_n1_f1_pp'])
    worst_subjects=sorted([r for r in subject_pairs if r['S1_run_id']==worst['S1_run_id']],key=lambda r:r['delta_n1_f1_pp'])
    worst_directions=[r for r in directions if r['S1_run_id']==worst['S1_run_id']]
    write_json(OUT/'key_findings.json',dict(worst_primary_N1_pair=worst,its_subject_deltas=worst_subjects,
                                           its_N1_directions=worst_directions,case_count=len(changed_rows),
                                           stable_case_selection='lowest downstream seed, then sample_id per modality/budget/adaptation/direction'))
    resources=[]
    for seed in range(3):
        run=ROOT/f'runs/W2_C_EEG_S1_aseed{seed}_steps10000_v01'
        summary=json.loads((run/'summary.json').read_text())
        resources.append(dict(adaptation_seed=seed,completed_updates=summary['completed_updates'],
            started_at=summary['started_at_Asia_Shanghai'],completed_at=summary['completed_at_Asia_Shanghai'],
            session_clock_seconds=summary['session_elapsed_seconds'],peak_allocated_mib=summary['peak_allocated_mib'],
            selected_checkpoint=(run/summary['selected_weight']['path']).relative_to(ROOT).as_posix(),
            selected_checkpoint_sha256=summary['selected_weight']['sha256']))
    output_csv('adaptation_resources_and_weights.csv',resources)
    make_plots(comparisons,directions)
    completion=dict(status='passed',verified_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),unique_runs=80,
                    primary_fusion_runs=40,EEG_diagnostic_runs=40,independent_adaptation_seeds=[0,1,2],
                    paired_contrasts=60,validation_subjects=8,validation_samples=21186,test_used=False,
                    D_full_cache_received=False,EOG_source='local_regeneration_using_D_reference_protocol',
                    all_artifact_SHA_and_point_metric_checks=checks,analysis_source_sha256=sha(__file__),
                    analysis_fits_models=False,analysis_selects_new_checkpoints=False)
    write_json(OUT/'completion_audit.json',completion)
    print(json.dumps(dict(status='complete',output=str(OUT),audited_runs=80,paired_contrasts=60,
                         changed_N1_rows=len(changed_rows),worst_pair=worst),ensure_ascii=False),flush=True)


def make_plots(comparisons,directions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for row,modality in enumerate(('EEG','EEG_EOG')):
        for col,metric in enumerate(('macro_f1','n1_f1')):
            ax=axes[row,col]
            for offset,budget,color in [(-.18,'label10','#236da1'),(.18,'label25','#d57a22')]:
                values=[next(r[f'delta_{metric}_pp'] for r in comparisons if r['modality']==modality
                             and r['budget']==budget and r['adaptation_seed']==seed) for seed in range(3)]
                ax.bar(np.arange(3)+offset,values,width=.33,label='5 labeled subjects' if budget=='label10' else '14 labeled subjects',color=color)
            ax.axhline(0,color='#333333',linewidth=1)
            ax.set_xticks(range(3),['Adapt seed 0','Adapt seed 1','Adapt seed 2'])
            ax.set_title(f'{modality}: {"Macro-F1" if metric=="macro_f1" else "N1 F1"}')
            ax.set_ylabel('S1 - S0 (percentage points)')
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Each bar averages 5 head seeds; 3 independently adapted backbones. Development validation only.',fontsize=11)
    fig.savefig(OUT/'adaptation_deltas.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,budget in zip(axes,('label10','label25')):
        selected=[r for r in directions if r['modality']=='EEG_EOG' and r['budget']==budget
                  and r['direction']=='N1_false_negative_destination']
        names=['W','N2','N3','REM']
        before=[np.mean([r['S0_count'] for r in selected if r['other_stage']==n]) for n in names]
        after=[np.mean([r['S1_count'] for r in selected if r['other_stage']==n]) for n in names]
        ax.bar(np.arange(4)-.18,before,.35,label='S0',color='#697b8d')
        ax.bar(np.arange(4)+.18,after,.35,label='S1 (all repeats)',color='#ac594a')
        ax.set_xticks(range(4),names);ax.set_title('5 labeled subjects' if budget=='label10' else '14 labeled subjects')
        ax.set_ylabel('True N1 epochs misclassified (mean per head)')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend()
    fig.suptitle('Primary EEG+EOG: N1 false-negative destinations; all preregistered repeats retained.',fontsize=11)
    fig.savefig(OUT/'primary_N1_error_destinations.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    analyze()
