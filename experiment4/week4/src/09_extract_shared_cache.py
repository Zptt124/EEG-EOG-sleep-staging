"""新一轮第 1 周：给预算线（C/D）和时序线（A/B）共用的 384 维索引化特征缓存（GPU，全量约 1 小时）。

口径（2026-11-26 新一轮第 1 周安排）：
  W4 主实验 30k、冻结 EMA（均匀 timestep 训练、5 秒窗）；读出 t=1、不加噪、query=gate、空条件；
  只保留第 10–12 层；每 1 秒求 std 后在 30 个位置上平均；EEG、EOG 各 192 维（拼接 384 维）。
  计算方式与 04_extract_features.py 完全相同（复用它的 gate_activations / pooled），
  先算全 12 层 768 维再取第 576–768 列，所以和 W4 已发布的 30k 全层特征逐位可比（--check-against）。

范围：训练池 54 人（含 W4 held-out 8 人，只缓存）+ 固定 dev 8 人；测试集 16 人不读取。
  行序 = 训练池按 pretrain_epoch_index.csv 的行序，后接 dev 按记录号、epoch 排序；
  每份记录的 epoch 必须与索引逐个对上，否则报错退出。
  --records 只提指定记录（"连续记录小包"，用于对方先核对 sample_id / 记录号 / epoch 序号）。

输出：SLEEP_W4_BIG/shared_cache/<out-name>/
  index.csv（sample_id, subject_id, record_id, epoch_index, split, role, y_true）
  eeg_std.npy、eog_std.npy（float32，[N, 192]，与 index.csv 同行序）
  features_info.json（C 的 train.py 核验的字段：step、weights、pool_sec、dim_per_modality、n_samples、
                      index_sha256、files.<name>.sha256）、DONE
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import w4d_common as C

_spec = importlib.util.spec_from_file_location('w4_extract', Path(__file__).resolve().parent / '04_extract_features.py')
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)

LAYERS = (10, 11, 12)          # 1-based，团队约定的后三层
POOL_SEC = 1
DEFAULT_RUN = 'W4_D_main_m64x12_seed0_v01'
DEFAULT_CKPT_SHA = '21c8514ea7761ef383d70b3e26ac19ea33ad0e7ccacfe690812ad359b0118e5b'   # W4 主实验 30k（Release 已登记）
EPOCH_INDEX_SHA = C.CACHE_INDEX_SHA
VAL_REF_SHA = '0f71748d782e400a2c0c398db0d66b60fe81b59dfc7e32d439dc350e4adb66e2'
HELDOUT_SHA = '470c23f7d49b4de6a3a896243e379ffc3b7a637d64b3e70c813b1c03522dffa0'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', default=DEFAULT_RUN)
    p.add_argument('--step', type=int, default=30000)
    p.add_argument('--expect-ckpt-sha', default=DEFAULT_CKPT_SHA, help="'' to skip")
    p.add_argument('--heldout-csv', required=True, help='manifests/pretrain_split_w4_v01.csv (SHA 470c23f7)')
    p.add_argument('--val-reference', required=True, help='manifests/val_reference_v1.csv (SHA 0f71748d)')
    p.add_argument('--out-name', required=True)
    p.add_argument('--records', default='', help='comma list of record ids (small package); empty = full cache')
    p.add_argument('--check-against', default='',
                   help='W4 all-layer feature dir; overlapping samples must equal its columns 576:768 bitwise')
    p.add_argument('--batch-epochs', type=int, default=32)
    args = p.parse_args()

    epoch_csv = C.MANIFESTS / 'pretrain_epoch_index.csv'
    for path, sha in ((epoch_csv, EPOCH_INDEX_SHA), (Path(args.val_reference), VAL_REF_SHA),
                      (Path(args.heldout_csv), HELDOUT_SHA)):
        if not path.exists():
            C.fail(f'Missing manifest: {path}')
        if C.sha256_file(path) != sha:
            C.fail(f'SHA mismatch for {path.name}: expected {sha[:8]}')
    C.check_manifests()

    ckpt = C.RUNS_BIG / args.run_name / f'ckpt_step{args.step}.pt'
    if not ckpt.exists():
        C.fail(f'Checkpoint not found: {ckpt}')
    ckpt_sha = C.sha256_file(ckpt)
    if args.expect_ckpt_sha and ckpt_sha != args.expect_ckpt_sha:
        C.fail(f'Checkpoint SHA {ckpt_sha[:8]} differs from expected {args.expect_ckpt_sha[:8]}')

    # ---- 期望的行序与 epoch 列表 ----
    recs = C.split_records()                                   # record -> (subject, split)
    heldout = {r['subject_id'] for r in C.read_csv(args.heldout_csv) if r['pretrain_role'] == 'heldout_val'}
    if len(heldout) != C.N_HELDOUT:
        C.fail(f'Expected {C.N_HELDOUT} held-out subjects, got {len(heldout)}')
    order, expected = [], {}
    for r in C.read_csv(epoch_csv):                            # 训练池 54 人，沿用缓存索引的行序
        rec = r['record_id']
        if rec not in expected:
            order.append(rec)
            expected[rec] = []
        expected[rec].append(int(r['epoch_index']))
    dev_truth = {}
    dev_rows = sorted(C.read_csv(args.val_reference), key=lambda r: (r['record_id'], int(r['epoch_index'])))
    for r in dev_rows:
        rec = r['record_id']
        if rec not in expected:
            order.append(rec)
            expected[rec] = []
        expected[rec].append(int(r['epoch_index']))
        dev_truth[r['sample_id']] = int(r['y_true'])
    for rec in order:
        if rec not in recs or recs[rec][1] not in ('train', 'val'):
            C.fail(f'Record {rec} is not in the train/val split (sealed test must never be read)')
    n_train_subj = len({recs[r][0] for r in order if recs[r][1] == 'train'})
    n_dev_subj = len({recs[r][0] for r in order if recs[r][1] == 'val'})
    if (n_train_subj, n_dev_subj) != (54, 8):
        C.fail(f'Expected 54 train-pool + 8 dev subjects, got {n_train_subj} + {n_dev_subj}')

    todo = order
    if args.records:
        want = [s.strip() for s in args.records.split(',') if s.strip()]
        bad = [r for r in want if r not in expected]
        if bad:
            C.fail(f'Records not in train pool / dev: {bad}')
        todo = [r for r in order if r in set(want)]

    out = C.BIG / 'shared_cache' / args.out_name
    if (out / 'DONE').exists():
        C.fail(f'{out} already complete')
    out.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, ck, opts = X.load_model(ckpt, 'ema_model')
    if opts != dict(self_cond_layer=0, self_cond_pool='mean', cross_modal='none') or model.window_sec != 5:
        C.fail(f'Expected a plain 5 s model without switches, got {opts} window {model.window_sec}')
    model = model.cuda()
    n_layer, d = len(model.layers), model.d_model
    cols = slice((LAYERS[0] - 1) * d, LAYERS[-1] * d)
    print(f'checkpoint {ckpt.name} step {ck.get("step")} sha {ckpt_sha[:8]} layers {LAYERS} cols {cols.start}:{cols.stop} '
          f'records {len(todo)}', flush=True)

    def extract(sig_mod, mi):
        x_all = C.transform_uv(sig_mod)
        S = []
        for b in range(0, len(x_all), args.batch_epochs):
            x = torch.from_numpy(x_all[b:b + args.batch_epochs, None, :]).cuda()
            a, _ = X.gate_activations(model, x, mi, X.DIFFUSION_T)
            s, _ = X.pooled(a, POOL_SEC)
            if not torch.isfinite(s).all():
                C.fail('Non-finite features')
            S.append(s[:, cols].cpu().numpy())
        return np.concatenate(S)

    index, feats = [], {'eeg_std': [], 'eog_std': []}
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for i, rec in enumerate(todo):
        subj, sp = recs[rec]
        with np.load(C.DATA / f'{rec}.npz', allow_pickle=False) as z:
            sig, lab, ep = z['signal'], z['labels'], z['epoch_index']
        ep = [int(e) for e in ep]
        if ep != expected[rec]:
            C.fail(f'{rec}: NPZ epochs do not match the frozen index (npz {len(ep)} vs index {len(expected[rec])})')
        for m in ('eeg', 'eog'):
            mi = C.MODALITIES[m]
            feats[f'{m}_std'].append(extract(sig[:, mi, :], mi))   # 只把这一模态的通道送进模型
        role = 'dev' if sp == 'val' else ('heldout_val' if subj in heldout else 'pretrain_train')
        for e, y in zip(ep, lab):
            sid = f'{rec}_{e:04d}'
            if sp == 'val' and dev_truth[sid] != int(y):
                C.fail(f'Dev label mismatch at {sid}')
            index.append(dict(sample_id=sid, subject_id=subj, record_id=rec, epoch_index=e,
                              split=sp, role=role, y_true=int(y)))
        el = time.perf_counter() - start
        print(f'[{i + 1:3d}/{len(todo)}] {rec} {role} n={len(ep)} elapsed={el:.0f}s '
              f'rate={len(index) / el:.1f} epochs/s', flush=True)

    elapsed = time.perf_counter() - start
    arrays = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    for k, a in arrays.items():
        if a.shape != (len(index), len(LAYERS) * d):
            C.fail(f'Bad shape for {k}: {a.shape}')
    if len({r['sample_id'] for r in index}) != len(index):
        C.fail('Duplicate sample_id')
    if not args.records:
        want_n = sum(len(v) for v in expected.values())
        if len(index) != want_n:
            C.fail(f'Row count {len(index)} differs from frozen train+dev index {want_n}')

    check = None
    if args.check_against:
        src = Path(args.check_against)
        ref_idx = {r['sample_id']: i for i, r in enumerate(C.read_csv(src / 'index.csv'))}
        mine = [(j, ref_idx[r['sample_id']]) for j, r in enumerate(index) if r['sample_id'] in ref_idx]
        if not mine:
            C.fail('No overlapping samples for --check-against')
        a_idx = np.array([j for j, _ in mine])
        b_idx = np.array([k for _, k in mine])
        check = dict(source=src.name, overlap=len(mine))
        for k in arrays:
            ref = np.load(src / f'{k}.npy', mmap_mode='r')
            same = np.array_equal(arrays[k][a_idx], np.asarray(ref[b_idx][:, cols]))
            maxdiff = float(np.abs(arrays[k][a_idx] - np.asarray(ref[b_idx][:, cols])).max())
            check[k] = dict(bitwise_equal=bool(same), max_abs_diff=maxdiff)
            if maxdiff > 1e-5:
                C.fail(f'{k}: differs from W4 all-layer features (max diff {maxdiff})')
        print(f'check against W4 all-layer features: {check}', flush=True)

    for k, a in arrays.items():
        np.save(out / f'{k}.npy', a)
    C.write_csv(out / 'index.csv', index)
    count = lambda key, val: sum(r[key] == val for r in index)
    info = dict(
        purpose='shared 384-d cache for the new-round week 1 budget and temporal lines' if not args.records
                else 'small continuous-record package for key checking (same format as the full cache)',
        run_name=args.run_name, checkpoint=ckpt.name, checkpoint_sha256=ckpt_sha,
        step=ck.get('step'), weights='ema_model', timestep_sampling='uniform', window_sec=5,
        readout=f'diffusion_t={X.DIFFUSION_T}, no noise, query=gate, null switches; layers '
                f'{LAYERS[0]}-{LAYERS[-1]} of {n_layer} (all-layer columns {cols.start}:{cols.stop}); '
                f'per-{POOL_SEC}s std (correction=1), mean over {30 // POOL_SEC} positions',
        layers=list(LAYERS), pool_sec=POOL_SEC, dim_per_modality=len(LAYERS) * d,
        concat_order=['eeg_std', 'eog_std'], dtype='float32',
        n_samples=len(index), n_records=len(todo),
        n_subjects=len({r['subject_id'] for r in index}),
        n_by_role={k: count('role', k) for k in ('pretrain_train', 'heldout_val', 'dev')},
        records=todo if args.records else None,
        row_order='train pool in pretrain_epoch_index.csv order, then dev by record_id and epoch_index',
        sources=dict(pretrain_epoch_index=EPOCH_INDEX_SHA, val_reference_v1=VAL_REF_SHA,
                     pretrain_split_w4_v01=HELDOUT_SHA, split_subjects=C.SPLIT_SHA),
        sealed_test_read=False, check_against_w4_all_layers=check,
        seconds=elapsed, epochs_per_second=len(index) / elapsed,
        peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
        index_sha256=C.sha256_file(out / 'index.csv'),
        files={k: dict(shape=list(a.shape), sha256=C.sha256_file(out / f'{k}.npy')) for k, a in arrays.items()})
    C.write_json(out / 'features_info.json', info)
    (out / 'DONE').write_text('ok')
    print(f'samples={len(index)} records={len(todo)} seconds={elapsed:.0f} '
          f'features_info_sha256={C.sha256_file(out / "features_info.json")}', flush=True)
    print('SHARED CACHE DONE', flush=True)


if __name__ == '__main__':
    main()
