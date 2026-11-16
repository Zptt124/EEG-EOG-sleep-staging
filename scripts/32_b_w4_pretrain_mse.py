"""Step W4-B2: 小 EEGDM 预训练的固定窗分段 MSE（B → A、D，第 1 天）。

在 31 号脚本锁定的 8 名 held-out 受试者、固定窗、固定噪声、固定 t 上，计算 v-prediction 的 MSE，
按扩散步每 5 步一段（t 0–4, 5–9, …, 45–49，共 10 段）、EEG/EOG 分开报告，另给全 t 平均。
默认取 checkpoint 的 EMA 权重。多个 checkpoint 可一次评估，用来比较预训练停止点。

与训练目标一致（D 第三周 w3d_train.py / 官方 PLDiffusionModel）：
  x_t = add_noise(x0, ε, t)，目标 v = get_velocity(x0, ε, t)，逐窗 MSE = mean((net(x_t, t, modality) − v)^2)。
  扩散设置：DDPMScheduler(50 步, beta 1e-4→0.05, squaredcos_cap_v2, v_prediction)。

条件状态：默认“空条件”，即直接调用 net(x_t, t, modality)，与下游 EOG-only 推理一致。三组跨模态模型先统一按
空条件比较。D 的公共脚本若把条件做成额外参数（如 condition_eeg=None），用 --forward 指定一个适配函数：
    def forward(net, x_t, t, modality, rows):   # rows 为该 batch 的窗表行（dict 列表）
        return net(x_t, t, modality, condition_eeg=None)
带条件的 MSE 请另起 --condition-tag（如 sync / shifted），结果表单列条件状态，不与空条件混在一起。

checkpoint 格式：D 第三周 Trainer.weights() 保存的 dict（含 model_kwargs、model、ema_model、step）。
其他格式可用 --model-kwargs JSON 补充结构参数、--weights-key 指定权重键。

用法：
  python scripts/32_b_w4_pretrain_mse.py --vendor <vendor/EEGDM> \
      --ckpt step0=<ckpt_dir>/step0.pt --ckpt step30k=<ckpt_dir>/step30000.pt \
      --out reports/week4/pretrain_mse/<run_name>
  python scripts/32_b_w4_pretrain_mse.py --self-test --vendor <vendor/EEGDM>   # 随机初始化模型自检（CPU 可跑）

输出（--out 目录）：
  mse_by_segment.csv   ckpt × 条件 × 模态 × 段：n_windows、mse_mean、se_subject（8 人人均值的标准误）
  mse_by_window.csv    逐窗 MSE（行号对应 manifests/pretrain_val_windows_w4_v01.csv）
  mse_by_subject.csv   ckpt × 条件 × 模态 × 受试者（全 t 平均）
  mse_run.json         checkpoint SHA256、窗表与噪声 SHA256、扩散设置、设备、耗时
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / 'data' / 'processed'

_spec = importlib.util.spec_from_file_location('heldout', Path(__file__).with_name('31_b_w4_pretrain_heldout.py'))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

NOISE_SCH_KWARGS = dict(num_train_timesteps=50, beta_start=0.0001, beta_end=0.05,
                        beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
# D 第三周 m64x12 的结构参数，仅供 --self-test 与缺少 model_kwargs 的 checkpoint 参考
M64X12_KWARGS = dict(in_channels=1, d_model=64, d_state=64, n_layer=12, n_ssm=None, kernel_init='diag-lin',
                     kernel_mode='diag', bidirectional=True, d_cond=256, d_cond_embed=128, local_cond_ch=0,
                     n_class=2, have_null_class=False, self_gated=False)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def add_vendor(vendor):
    vendor = Path(vendor or os.environ.get('EEGDM_VENDOR', '')).expanduser()
    if not (vendor / 'model' / 'diffusion_model.py').exists():
        sys.exit(f'找不到官方 EEGDM 代码（--vendor 或 EEGDM_VENDOR）：{vendor}')
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return vendor


def load_manifest():
    meta = json.loads(H.OUT_META.read_text(encoding='utf-8'))
    for path in (H.OUT_SPLIT, H.OUT_WINDOWS):
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != meta['files'][path.name]:
            sys.exit(f'{path.name} 与 {H.OUT_META.name} 登记的 SHA256 不一致')
    rows = pd.read_csv(H.OUT_WINDOWS)
    if len(rows) != meta['windows']['n_rows'] or not (rows['row'] == np.arange(len(rows))).all():
        sys.exit('窗表行数或行号异常')
    noise = H.make_noise(len(rows))
    if hashlib.sha256(noise.numpy().tobytes()).hexdigest() != meta['noise']['sha256']:
        sys.exit('噪声 SHA256 与登记不一致（torch 版本的 CPU 随机数实现可能不同，请报告 B）')
    return meta, rows, noise


def build_x0(rows):
    """按 D 第三周 transform_uv 处理整段 30 秒，再切第 k 个 5 秒窗 → [n,1,1000] float32。"""
    from scipy.signal import resample_poly
    from src.util import staged_mu_law
    x0 = np.empty((len(rows), 1, H.WIN_200HZ), dtype=np.float32)
    for rec, g in rows.groupby('record_id', sort=True):
        with np.load(PROCESSED / f'{rec}.npz') as z:
            ei, sig = z['epoch_index'], z['signal']
        if not (ei[g['npz_row'].to_numpy()] == g['epoch_index'].to_numpy()).all():
            sys.exit(f'{rec}: npz_row 与 epoch_index 不对应，NPZ 可能不是 A 的原始版本')
        x = sig[g['npz_row'].to_numpy(), g['modality_id'].to_numpy()].astype(np.float32)   # (n, 3000) µV
        x = resample_poly(x, 2, 1, axis=-1).astype(np.float32)
        x = staged_mu_law(x / 100).astype(np.float32)
        k = g['window_k'].to_numpy()
        idx = k[:, None] * H.WIN_200HZ + np.arange(H.WIN_200HZ)[None, :]
        x0[g.index.to_numpy(), 0] = np.take_along_axis(x, idx, axis=1)
    if not np.isfinite(x0).all():
        sys.exit('x0 含非有限值')
    return x0


def load_net(spec, weights_key, model_kwargs_override, device):
    from model.diffusion_model import Wavenet
    if spec == 'SELFTEST':
        torch.manual_seed(0)
        net = Wavenet(**M64X12_KWARGS)
        return net.to(device).eval(), dict(path='SELFTEST', sha256=None, step=0, weights_key='random_init_seed0')
    path = Path(spec).expanduser()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    kw = model_kwargs_override or ck.get('model_kwargs')
    if kw is None:
        sys.exit(f'{path}: checkpoint 无 model_kwargs，请用 --model-kwargs 提供')
    key = weights_key
    if key not in ck:
        sys.exit(f'{path}: 没有权重键 {key!r}（可用：{[k for k, v in ck.items() if isinstance(v, dict)]}）')
    net = Wavenet(**kw)
    net.load_state_dict(ck[key], strict=True)
    info = dict(path=str(path), sha256=sha256_file(path), step=ck.get('step'), weights_key=key,
                cfg_name=ck.get('cfg_name'), seed=ck.get('seed'), model_kwargs=kw)
    return net.to(device).eval(), info


def load_forward(spec):
    if not spec:
        return lambda net, xt, t, m, rows: net(xt, t, m)
    path, fn = spec.rsplit(':', 1)
    s = importlib.util.spec_from_file_location('fwd_adapter', path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return getattr(mod, fn)


@torch.no_grad()
def eval_mse(net, forward, sch, x0, noise, rows, batch, device):
    t_all = torch.from_numpy(rows['t'].to_numpy().copy()).long()[:, None]
    m_all = torch.from_numpy(rows['modality_id'].to_numpy().copy()).long()[:, None]
    rec = rows.to_dict('records')
    out = np.empty(len(rows), dtype=np.float64)
    for s in range(0, len(rows), batch):
        sl = slice(s, s + batch)
        x, e = torch.from_numpy(x0[sl]).to(device), noise[sl].to(device)
        t, m = t_all[sl].to(device), m_all[sl].to(device)
        xt = sch.add_noise(x, e, t)
        v = sch.get_velocity(x, e, t)
        pred = forward(net, xt, t, m, rec[sl])
        if pred.shape != v.shape:
            sys.exit(f'模型输出形状 {tuple(pred.shape)} ≠ 目标 {tuple(v.shape)}')
        out[sl] = ((pred.float() - v.float()) ** 2).mean(dim=(1, 2)).double().cpu().numpy()
    if not np.isfinite(out).all():
        sys.exit('MSE 含非有限值')
    return out


def summarize(rows, mse, label, cond):
    d = rows[['row', 'subject_id', 'modality', 't', 't_segment']].copy()
    d['mse'] = mse
    seg_rows = []
    for (mod, seg), g in list(d.groupby(['modality', 't_segment'])) + [((m, 'all'), g) for m, g in d.groupby('modality')]:
        subj = g.groupby('subject_id')['mse'].mean()
        t_lo, t_hi = (0, H.T_STEPS - 1) if seg == 'all' else (seg * H.T_SEGMENT, seg * H.T_SEGMENT + H.T_SEGMENT - 1)
        seg_rows.append(dict(ckpt=label, condition=cond, modality=mod, t_segment=seg, t_range=f'{t_lo}-{t_hi}',
                             n_windows=len(g), n_subjects=len(subj), mse_mean=g['mse'].mean(),
                             se_subject=subj.std(ddof=1) / np.sqrt(len(subj))))
    by_subj = d.groupby(['modality', 'subject_id'])['mse'].agg(['mean', 'size']).reset_index()
    by_subj.columns = ['modality', 'subject_id', 'mse_mean', 'n_windows']
    by_subj.insert(0, 'condition', cond)
    by_subj.insert(0, 'ckpt', label)
    d.insert(0, 'condition', cond)
    d.insert(0, 'ckpt', label)
    return pd.DataFrame(seg_rows), by_subj, d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', action='append', default=[], help='label=path，可重复；label 用于结果表')
    ap.add_argument('--vendor', help='官方 EEGDM 代码目录（commit 003b5a2），也可设 EEGDM_VENDOR')
    ap.add_argument('--weights-key', default='ema_model', help='默认 EMA 权重；random-init 的 step0 两者相同')
    ap.add_argument('--model-kwargs', help='JSON 字符串，覆盖/补充 checkpoint 的 model_kwargs')
    ap.add_argument('--forward', help='path.py:fn，自定义前向（如带 condition_eeg=None 的新接口）')
    ap.add_argument('--condition-tag', default='null', help='条件状态标签；空条件为 null')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch', type=int, default=200)
    ap.add_argument('--out', help='输出目录')
    ap.add_argument('--self-test', action='store_true', help='随机初始化 m64x12 跑两遍，检查可复现性')
    args = ap.parse_args()

    vendor = add_vendor(args.vendor)
    from diffusers import DDPMScheduler
    sch = DDPMScheduler(**NOISE_SCH_KWARGS)
    meta, rows, noise = load_manifest()
    t0 = time.time()
    x0 = build_x0(rows)
    print(f'窗表 {len(rows)} 行，x0 {x0.shape}，准备 {time.time() - t0:.1f}s，设备 {args.device}')

    if args.self_test:
        specs = [('selftest_a', 'SELFTEST'), ('selftest_b', 'SELFTEST')]
    else:
        if not args.ckpt or not args.out:
            sys.exit('需要至少一个 --ckpt label=path 与 --out')
        specs = [tuple(c.split('=', 1)) if '=' in c else (Path(c).stem, c) for c in args.ckpt]
        if len({s[0] for s in specs}) != len(specs):
            sys.exit('--ckpt 的 label 重复')
    kw = json.loads(args.model_kwargs) if args.model_kwargs else None
    forward = load_forward(args.forward)

    segs, subjs, wins, infos = [], [], [], []
    for label, spec in specs:
        net, info = load_net(spec, args.weights_key, kw, args.device)
        ts = time.time()
        mse = eval_mse(net, forward, sch, x0, noise, rows, args.batch, args.device)
        info.update(label=label, seconds=round(time.time() - ts, 1))
        s, b, w = summarize(rows, mse, label, args.condition_tag)
        segs.append(s), subjs.append(b), wins.append(w), infos.append(info)
        allm = s[s.t_segment == 'all'].set_index('modality')['mse_mean']
        print(f'{label}: EEG {allm["eeg"]:.5f}  EOG {allm["eog"]:.5f}  ({info["seconds"]}s)')
    seg = pd.concat(segs, ignore_index=True)

    if args.self_test:
        a, b = wins[0]['mse'].to_numpy(), wins[1]['mse'].to_numpy()
        diff = float(np.abs(a - b).max())
        print(f'两遍逐窗 MSE 最大差 {diff:.3e}')
        print(seg[seg.ckpt == 'selftest_a'].pivot(index='t_segment', columns='modality', values='mse_mean').round(4))
        sys.exit(0 if diff < 1e-6 else 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seg.to_csv(out / 'mse_by_segment.csv', index=False, float_format='%.6g')
    pd.concat(subjs, ignore_index=True).to_csv(out / 'mse_by_subject.csv', index=False, float_format='%.6g')
    pd.concat(wins, ignore_index=True).to_csv(out / 'mse_by_window.csv', index=False, float_format='%.6g')
    run = dict(script='scripts/32_b_w4_pretrain_mse.py', manifest=meta['version'], manifest_files=meta['files'],
               noise_sha256=meta['noise']['sha256'], heldout_subjects=meta['heldout_subjects'],
               scheduler=NOISE_SCH_KWARGS, condition=args.condition_tag, forward=args.forward or 'net(x_t, t, modality)',
               weights_key=args.weights_key, device=args.device, torch=torch.__version__,
               python=platform.python_version(), vendor=str(vendor), checkpoints=infos)
    (out / 'mse_run.json').write_text(json.dumps(run, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
    wide = seg.pivot_table(index=['modality', 't_segment'], columns='ckpt', values='mse_mean', sort=False)
    print(wide.round(5).to_string())
    print(f'已写入 {out}')


if __name__ == '__main__':
    main()
