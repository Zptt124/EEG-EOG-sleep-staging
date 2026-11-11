"""第 10 步：整理第三周实验 D 的 PR 交付包（在 D: 上生成 pr_package/experiment4/week3/）。

- 复制源码、预训练运行记录（不含权重）、测速、B 协议下 63 个 run 的小文件（predictions/metrics/config/inner_cv/by_subject）、
  分析表、日志；oof_train.csv、final.ckpt 与预训练权重放 Release（由 submit_pr.sh 打包）。
- 文本文件里的本机路径替换为占位符（home 目录 -> ~，D 盘项目根 -> <D_ROOT>），最后检查无残留。
- 生成 B 统一评估用登记表（predictions 为相对仓库根目录的路径）与资源记录。
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

W31 = Path(__file__).resolve().parents[2]
EXP = W31 / 'exp_d_pretrain'
OUT = W31 / 'pr_package/experiment4/week3'
REPO_PREFIX = 'experiment4/week3'
USER = 'lan' + 'gt'   # 拆开写，避免本文件自身含本机用户名
REPL = [('/mnt/' + 'd/03_Study/Research/sleepdm-eeg-eog', '<D_ROOT>'), ('/mnt/' + 'd/', '<D_DRIVE>/'),
        ('/home/' + USER, '~')]
TEXT = {'.py', '.sh', '.json', '.csv', '.md', '.log', '.yaml', '.txt'}
SMALL = ['predictions_val.csv', 'metrics.json', 'config.json', 'inner_cv.csv', 'by_subject.csv']


def fail(msg):
    print(f'ERROR: {msg}', flush=True)
    sys.exit(1)


def copy_text(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in TEXT:
        s = src.read_text(encoding='utf-8', errors='strict')
        for a, b in REPL:
            s = s.replace(a, b)
        dst.write_text(s, encoding='utf-8', newline='')
    else:
        shutil.copy2(src, dst)


def main():
    if OUT.exists():
        fail(f'{OUT} exists; move it away first')
    # 源码
    for f in sorted((EXP / 'src').iterdir()):
        if f.is_file() and f.suffix in ('.py', '.sh'):
            copy_text(f, OUT / 'src' / f.name)
    # 名单副本与预训练样本清单
    for f in ('pretrain_cache_manifest.json', 'pretrain_epoch_index.csv'):
        copy_text(EXP / 'manifests' / f, OUT / 'manifests' / f)
    # 预训练运行记录（不含权重；旧保底 probe 另放）
    for run in sorted((EXP / 'runs').iterdir()):
        for f in sorted(run.iterdir()):
            if f.is_file():
                copy_text(f, OUT / 'pretrain' / run.name / f.name)
        if (run / 'probe').exists():
            for f in sorted((run / 'probe').rglob('*')):
                if f.is_file():
                    copy_text(f, OUT / 'superseded_fallback_probe' / 'runs' / f.relative_to(run / 'probe'))
    for d in sorted((EXP / 'analysis').glob('compare_*')):
        for f in sorted(d.iterdir()):
            copy_text(f, OUT / 'superseded_fallback_probe' / 'analysis' / d.name / f.name)
    # 测速
    sp = json.loads((EXP / 'analysis/speed_test/speed_m64x12.json').read_text(encoding='utf-8'))
    copy_text(EXP / 'analysis/speed_test/speed_m64x12.json', OUT / 'speed_test/speed_m64x12.json')
    rows = [sp['row'], dict(config='m64x16', d_model=64, d_state=64, n_layer=16, d_cond=256, n_params=1257921,
                            batch_size=108, timed_steps=0,
                            note='did not finish 30 warmup steps in ~8 min (>15 s/step): GPU memory overflow into '
                                 'shared system memory on 8 GB RTX 4060 Laptop; m96x16 and official_ref not run')]
    pd.DataFrame(rows).to_csv(OUT / 'speed_test/speed_test_summary.csv', index=False)
    # B 协议 runs（小文件）
    runs = sorted(p for p in (EXP / 'runs_bprotocol').iterdir() if p.is_dir())
    if len(runs) != 63:
        fail(f'expected 63 probe runs, found {len(runs)}')
    for r in runs:
        for f in SMALL:
            if not (r / f).exists():
                fail(f'missing {r / f}')
            copy_text(r / f, OUT / 'runs' / r.name / f)
    # 分析表
    for sub in ('bprotocol', 'bprotocol_seed1'):
        for f in sorted((EXP / 'analysis' / sub).iterdir()):
            copy_text(f, OUT / 'analysis' / sub / f.name)
    # 日志
    for f in sorted((W31 / 'logs').iterdir()):
        copy_text(f, OUT / 'logs' / f.name)
    # 登记表（给 B 的 29_b_w3_evaluate.py --extra）
    regs = []
    for f in (EXP / 'analysis/bprotocol/d_eval_registry_w3.csv', EXP / 'analysis/bprotocol_seed1/d_eval_registry_seed1.csv'):
        regs.append(pd.read_csv(f))
    reg = pd.concat(regs).drop_duplicates('run_id').sort_values('run_id')
    reg['predictions'] = [f'{REPO_PREFIX}/runs/{r}/predictions_val.csv' for r in reg['run_id']]
    if len(reg) != 63:
        fail(f'registry should have 63 rows, got {len(reg)}')
    (OUT / 'registry').mkdir(parents=True, exist_ok=True)
    reg.to_csv(OUT / 'registry/d_eval_registry_w3.csv', index=False)
    # 资源记录（预训练、提特征、探针）
    res = []
    for run in sorted((EXP / 'runs').iterdir()):
        h = pd.read_csv(run / 'history.csv')
        cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        active_h = float((h['sec_per_step'] * h['step'].diff().fillna(h['step'].iloc[0])).sum() / 3600)
        res.append(dict(run_id=run.name, phase='pretrain', device='cuda RTX 4060 Laptop 8GB', precision='float32 + TF32',
                        trainable_parameters=992961, actual_updates=int(h['step'].max()),
                        actual_epochs=round(int(h['step'].max()) * cfg['batch_size'] / 1595256, 3),
                        elapsed_seconds=round(active_h * 3600), peak_memory_mib=int(h['peak_gpu_mib'].max()),
                        examples_per_second=round(cfg['batch_size'] / h['sec_per_step'].median(), 1),
                        notes='5-s windows x 2 modalities; active time from history (excludes ~4 h laptop sleep in seed0)'))
        for fj in sorted(run.glob('features_*.json')):
            fi = json.loads(fj.read_text(encoding='utf-8'))
            res.append(dict(run_id=f'{run.name}:{fj.stem}', phase='feature_extraction', device='cuda RTX 4060 Laptop 8GB',
                            precision='float32 + TF32', trainable_parameters=0, actual_updates=0, actual_epochs=0,
                            elapsed_seconds=round(fi['seconds']), peak_memory_mib=round(fi['peak_gpu_mib']),
                            examples_per_second=round(fi['epochs_per_second'], 1),
                            notes=f'{fi["n_samples"]} 30-s epochs (14 labeled train + 8 dev), eeg+eog'))
    for r in runs:
        c = json.loads((r / 'config.json').read_text(encoding='utf-8'))
        res.append(dict(run_id=r.name, phase='linear_probe', device='cpu (4 threads)', precision='float32',
                        trainable_parameters=c['feature_dim'] * 5 + 5, actual_updates=c['final_updates'],
                        actual_epochs=c['selected_epoch'], elapsed_seconds=c['elapsed_seconds'], peak_memory_mib='',
                        examples_per_second='', notes='B week-3 protocol (5-fold subject CV x 50 epochs + retrain)'))
    pd.DataFrame(res).to_csv(OUT / 'registry/d_resource_profile_w3.csv', index=False)
    # 路径泄露检查
    bad = [str(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in TEXT
           and (USER in p.read_text(encoding='utf-8') or '/mnt/' + 'd/' in p.read_text(encoding='utf-8'))]
    if bad:
        fail(f'local paths remain in {bad[:5]}')
    n = sum(1 for p in OUT.rglob('*') if p.is_file())
    size = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) / 2**20
    print(f'package files={n} size={size:.1f} MiB -> {OUT}', flush=True)
    print('PACKAGE DONE', flush=True)


if __name__ == '__main__':
    main()
