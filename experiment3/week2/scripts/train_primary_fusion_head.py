"""Locked primary fusion head; pilot math with registered 5120-dimensional input."""
import json
import time
import numpy as np
import torch
import yaml
from replication_feature_head_runtime import (ROOT, sha, require, write_json, write_csv,
    new_head, predict, selection_update)


def train_head(plan, rows, train_x, train_y, val_x, val_y, mean, std, context, official, adaptation_seed):
    output = ROOT/'runs'/plan['run_id']
    output.mkdir(parents=True, exist_ok=False)
    seed = int(plan['downstream_seed'])
    model, optimizer = new_head(seed, dimension=5120)
    cfg = dict(context, run_id=plan['run_id'], representation=plan['representation'],
               adaptation_seed=adaptation_seed if plan['representation'] == 'S1' else None,
               downstream_seed=seed, sampling_seed=2024, budget=plan['budget'], modality='EEG_EOG',
               role='primary', input_dim=5120, classifier='Linear(5120,5)',
               optimizer='AdamW', learning_rate=0.001, weight_decay=0.0001,
               batch_size=128, max_epochs=50, patience=10, loss='unweighted_cross_entropy',
               selection='strictly highest development validation macro-F1',
               normalization='fit only current budget labeled train subjects, independently for S0/S1',
               std_floor=1e-6, backbone_frozen=True, test_used=False, device='cpu',
               cpu_threads=4, deterministic=True, N1_REM_from_same_selected_head=True)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    history, best, stale = [], -1., 0
    start_time = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        total = 0.
        for ids in torch.randperm(len(train_x)).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(train_x[ids]), train_y[ids])
            require(bool(torch.isfinite(loss)), 'Nonfinite classifier loss')
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                    'Nonfinite/missing classifier gradient')
            optimizer.step()
            total += loss.item()*len(ids)
        model.eval()
        pred = predict(model, val_x)
        score = official.compute_metrics(val_y, pred)['macro_f1']
        history.append(dict(epoch=epoch, train_loss=total/len(train_x), val_macro_f1=score))
        write_json(output/'history.json', history)
        improved, best, stale = selection_update(score, best, stale)
        if improved:
            torch.save(dict(model=model.state_dict(), mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                            epoch=epoch, config=cfg), output/'best_head.pt')
        print(json.dumps(dict(stage='train_head', run_id=plan['run_id'], **history[-1])), flush=True)
        if stale >= 10:
            break
    checkpoint = torch.load(output/'best_head.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    probs = predict(model, val_x, True)
    require(np.isfinite(probs).all(), 'Nonfinite validation probabilities')
    predictions = []
    for row, probability in zip(rows, probs):
        result = {k: row[k] for k in ('sample_id', 'subject_id', 'record_id', 'epoch_index', 'y_true')}
        result['y_pred'] = int(probability.argmax())
        result.update(zip(official.PROB_COLS, map(float, probability)))
        predictions.append(result)
    write_csv(output/'predictions_val.csv', predictions)
    frame = official.load_predictions(output/'predictions_val.csv')
    require(len(frame) == 21186 and frame['subject_id'].nunique() == 8, 'Validation prediction coverage differs')
    metrics = official.evaluate(output/'predictions_val.csv', n_boot=1000, seed=0)
    write_json(output/'metrics.json', metrics)
    official.per_subject_table(frame).to_csv(output/'per_subject_metrics.csv', index=False)
    write_csv(output/'confusion_matrix.csv', [dict(true_stage=stage, **dict(zip(official.STAGE_NAMES, counts)))
              for stage, counts in zip(official.STAGE_NAMES, metrics['confusion_matrix'])])
    elapsed = time.perf_counter()-start_time
    write_csv(output/'resource_profile.csv', [dict(device='cpu', threads=4, seconds=elapsed,
              epochs=len(history), train_samples=len(train_x), val_samples=len(val_x), test_used=False)])
    result = dict(status='complete', run_id=plan['run_id'], representation=plan['representation'], budget=plan['budget'],
                  seed=seed, epochs=len(history), best_epoch=checkpoint['epoch'], val_macro_f1=metrics['macro_f1'],
                  n1_f1=metrics['n1_f1'], rem_f1=metrics['per_class_f1']['REM'], seconds=elapsed, test_used=False)
    (output/'run_summary.md').write_text(
        f"# {plan['run_id']}\n\nEEG+EOG primary comparison; adaptation seed {adaptation_seed}. "
        f"Best epoch {checkpoint['epoch']}; validation Macro-F1 {metrics['macro_f1']:.8f}, "
        f"N1 F1 {metrics['n1_f1']:.8f}, REM F1 {metrics['per_class_f1']['REM']:.8f}. "
        "All scores use the same selected head. Test unused.\n", encoding='utf-8')
    write_json(output/'artifact_sha256.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'summary.json', result)
    return result
