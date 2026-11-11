"""第 9 步：backbone seed 1 轻量复现（30k 步）的探针与对比（CPU，约 10–15 分钟）。

与 08 完全相同的 B 第三周协议（直接复用 08 里的 run_probe 与 B 的函数）。
臂（看结果前固定）：
  s1_rand（seed1 step0）、s1_pre030k（seed1 step30000）：label25 与 label10，head seed 0-2
  d_pre030k（seed0 step30000）补 label10（label25 已在 08 中完成）
配对：s1_pre030k:s1_rand、d_pre030k:d_rand、s1_pre030k:d_pre030k、s1_rand:d_rand、
      s1_pre030k:eegdm_official、d_pre030k:eegdm_official（label25 与 label10）
输出：runs_bprotocol/ 新 run；analysis/bprotocol_seed1/*.csv
"""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import w3d_common as C

RUN0 = 'W3_D_pretrain_m64x12_bs108_seed0_v01'
RUN1 = 'W3_D_pretrain_m64x12_bs108_seed1_v01'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def src(run, tag):
    d = C.BIG / 'features' / run / tag
    if not (d / 'DONE').exists():
        C.fail(f'Missing features {d}')
    return dict(kind='d', dir=d, files=[d / 'eeg_std.npy', d / 'eog_std.npy'], index=d / 'index.csv',
                desc=f'D small backbone {run} {tag}, eeg+eog, std pooling')


def main():
    repo = C.HOME / 'repos/w3_main'
    here = Path(__file__).resolve().parent
    m08 = load_module(here / '08_bprotocol_probes.py', 'm08')
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    sys.path.insert(0, str(repo / 'scripts'))
    import w2_common as w
    ev = w.ev
    b27 = m08.load_module(repo / 'scripts/27_b_w3_linear_probe.py', 'b27')
    b29 = m08.load_module(repo / 'scripts/29_b_w3_evaluate.py', 'b29')
    subsets = repo / 'manifests/label_subsets_v2.csv'
    if w.sha256_file(subsets) != C.LABEL_SUBSETS_SHA:
        C.fail('label_subsets_v2.csv mismatch')
    print(f'B code commit {commit}', flush=True)

    srcs = {'s1_rand': src(RUN1, 'step0_ema_model'), 's1_pre030k': src(RUN1, 'step30000_ema_model'),
            'd_rand': src(RUN0, 'step0_ema_model'), 'd_pre030k': src(RUN0, 'step30000_ema_model'),
            'eegdm_official': m08.r1_source()}
    out_root = C.EXP_D / 'runs_bprotocol'
    ana = C.ANALYSIS / 'bprotocol_seed1'
    ana.mkdir(parents=True, exist_ok=True)
    reg = []
    t_all = time.perf_counter()
    for arm in srcs:
        for budget in ('label25', 'label10'):
            for seed in (0, 1, 2):
                t0 = time.perf_counter()
                run_id, run_dir, met = m08.run_probe(arm, srcs[arm], budget, seed, b27, w, ev, out_root, commit, subsets)
                reg.append(dict(run_id=run_id, group='D', arm=arm, budget=budget, seed=seed,
                                predictions=str(run_dir / 'predictions_val.csv')))
                print(f'{run_id}: E*={met["selected_epoch"]} dev macroF1 {met["macro_f1"]:.4f} N1-F1 {met["n1_f1"]:.3f} '
                      f'({time.perf_counter() - t0:.0f}s, total {(time.perf_counter() - t_all) / 60:.1f} min)', flush=True)
    reg = pd.DataFrame(reg, columns=b29.COLS)
    reg.to_csv(ana / 'd_eval_registry_seed1.csv', index=False)
    preds, rows = {}, []
    for r in reg.to_dict('records'):
        df, sha = b29.load(r['predictions'])
        preds[r['run_id']] = df
        rows.append(b29.run_metrics(r, df, sha))
    met = pd.DataFrame(rows)
    met.to_csv(ana / 'd_w3_metrics_seed1.csv', index=False, float_format='%.6g')
    show = ['macro_f1', 'n1_f1', 'n1_precision', 'n1_recall', 'n1_ap', 'f1_REM']
    print(met.groupby(['arm', 'budget'])[show].mean().round(4).to_string(), flush=True)
    pairs = [('s1_pre030k', 's1_rand'), ('d_pre030k', 'd_rand'), ('s1_pre030k', 'd_pre030k'), ('s1_rand', 'd_rand'),
             ('s1_pre030k', 'eegdm_official'), ('d_pre030k', 'eegdm_official')]
    prow, psub = [], []
    for comp, ref in pairs:
        r, s = b29.paired(reg, preds, comp, ref)
        prow += r
        psub += s
    pr = pd.DataFrame(prow)
    pr.to_csv(ana / 'd_w3_paired_seed1.csv', index=False, float_format='%.6g')
    pd.DataFrame(psub).to_csv(ana / 'd_w3_paired_by_subject_seed1.csv', index=False, float_format='%.6g')
    cols = ['comparison', 'reference', 'budget'] + [f'delta_{k}{x}' for k in b29.KEYS for x in ('', '_lo', '_hi')]
    print(pr[cols].round(4).to_string(index=False), flush=True)
    print('SEED1 PROBES DONE', flush=True)


if __name__ == '__main__':
    main()
