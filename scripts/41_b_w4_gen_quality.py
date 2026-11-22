"""Step W4-B11: D 主实验各 checkpoint 的生成质量（B 的第四周任务）。

用 checkpoint 的 EMA 权重从纯噪声生成 5 秒 EEG / EOG，和 8 名预训练 held-out 受试者的真实窗比较。三类指标：
  1. 分布距离：平均对数功率谱差（dB）、五个频段相对功率的 Wasserstein 距离、窗内幅度（log std）的 Wasserstein 距离、
     手工特征上的 RBF-MMD²。
  2. 真假分类器：HistGradientBoosting，手工特征（对数功率谱 + 时域统计），5 折分层 CV 的 AUC；0.5 = 分不出来。
  3. FID / KID：用一个睡眠分期 1D CNN 的倒数第二层作特征。CNN 只在预训练池 46 人的真实窗上训练（不含 held-out、开发集、测试集），
     每个模态单独训练一个，只作为评估工具，不进入任何下游模型。

参照下限：把 held-out 真实窗分成同样大小的两半 A、B（每人各一半），所有指标都报「生成 vs A」和「B vs A」，
后者就是真实数据自身的抽样波动，生成样本越接近它越好。

子命令：
  gen      生成：DDPMScheduler（50 步，与训练一致，clip_sample=False），祖先采样，空条件；
           各 checkpoint 用同一批初始噪声和随机流，结果写到 --cache/gen_step{N}.npz。
  fidnet   训练 FID 用的分期 CNN，写到 --cache/fidnet_{eeg,eog}.pt。
  metrics  计算全部指标，写到 --out。

用法：
  python scripts/41_b_w4_gen_quality.py gen --ckpt-dir <0to30k 目录> <35kto240k 目录> --steps 5000 10000 ... --device mps
  python scripts/41_b_w4_gen_quality.py fidnet --device mps
  python scripts/41_b_w4_gen_quality.py metrics --out reports/week4/gen_quality
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys

if 'metrics' in sys.argv:
    # macOS 上 torch 和 sklearn 的 OpenMP 同时多线程时，HistGradientBoosting 会段错误；指标部分改单线程
    os.environ.setdefault('OMP_NUM_THREADS', '1')
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / 'data' / 'processed'
VENDOR = BASE / 'cache' / 'vendor' / 'EEGDM'
FS, WIN, N_WIN = 200, 1000, 6
MODS = ('eeg', 'eog')
BANDS = dict(delta=(0.5, 4), theta=(4, 8), alpha=(8, 12), sigma=(12, 15), beta=(15, 30))
GEN_SEED = 20261122

_spec = importlib.util.spec_from_file_location('m36', Path(__file__).with_name('36_b_w4_check_d_pretrain.py'))
M36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M36)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def add_vendor():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def to_uv(x):
    """模型域（µV/100 再 staged mu-law）→ µV。"""
    add_vendor()
    from src.util import inv_staged_mu_law
    return inv_staged_mu_law(np.array(x, dtype=np.float64)) * 100


# ---------------------------------------------------------------- 真实窗
def real_windows():
    """held-out 8 人的真实窗：D 的 val 单元（每人每模态 128 个）。返回 {mod: (x0[N,1000] 模型域, subject[N])}。"""
    idx = M36.rebuild_index()
    tab, _, _ = M36.rebuild_val(idx, M36.heldout_subjects())
    add_vendor()
    x0 = M36.build_x0(tab)[:, 0]
    out = {}
    for mi, mod in enumerate(MODS):
        sel = (tab['modality_id'] == mi).to_numpy()
        out[mod] = (x0[sel], tab.loc[sel, 'subject_id'].to_numpy())
    return out


def split_ab(subj):
    """每名受试者的窗按固定顺序一半进 A、一半进 B。"""
    rng = np.random.default_rng(0)
    a = np.zeros(len(subj), bool)
    for s in np.unique(subj):
        i = np.nonzero(subj == s)[0]
        a[rng.permutation(i)[: len(i) // 2]] = True
    return a


# ---------------------------------------------------------------- 生成
def ckpt_table(dirs):
    rows = {}
    for d in dirs:
        for r in csv.DictReader(open(Path(d) / 'checkpoints.csv', encoding='utf-8')):
            if (Path(d) / r['file']).exists():   # 35k–240k 包的 csv 也列了 0–30k，但不含这些文件
                rows[int(r['step'])] = (Path(d) / r['file'], r['sha256'])
    return rows


@torch.no_grad()
def cmd_gen(args):
    add_vendor()
    from model.diffusion_model import Wavenet
    from diffusers import DDPMScheduler
    sch_kw = dict(num_train_timesteps=50, beta_start=0.0001, beta_end=0.05,
                  beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction', clip_sample=False)
    table = ckpt_table(args.ckpt_dir)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    for step in args.steps:
        outp = cache / f'gen_step{step}.npz'
        if outp.exists() and not args.force:
            print(f'step {step}: 已有，跳过', flush=True)
            continue
        path, sha = table[step]
        if sha256_file(path) != sha:
            sys.exit(f'{path}: SHA256 与 checkpoints.csv 不一致')
        st = torch.load(path, map_location='cpu', weights_only=False)
        net = Wavenet(**st['model_kwargs'])
        net.load_state_dict(st['ema_model'], strict=True)
        net.eval().to(dev)
        # 同一台机器上核对 MPS 与 CPU 前向一致
        if dev.type != 'cpu':
            g = torch.Generator().manual_seed(1)
            xt = torch.randn(8, 1, WIN, generator=g)
            tt, mm = torch.full((8, 1), 20), torch.zeros(8, 1, dtype=torch.long)
            ref = Wavenet(**st['model_kwargs'])
            ref.load_state_dict(st['ema_model'])
            ref.eval()
            dmax = float((net(xt.to(dev), tt.to(dev), mm.to(dev)).cpu() - ref(xt, tt, mm)).abs().max())
        else:
            dmax = 0.0
        sch = DDPMScheduler(**sch_kw)
        sch.set_timesteps(50)
        res, t0 = {}, time.time()
        for mi, mod in enumerate(MODS):
            g = torch.Generator().manual_seed(GEN_SEED + mi)       # 各 checkpoint 相同
            x = torch.randn(args.n, 1, WIN, generator=g)
            # 每一步的随机噪声预先生成，保证各 checkpoint 的随机流完全一致
            step_noise = torch.randn(len(sch.timesteps), args.n, 1, WIN, generator=g)
            x = x.to(dev)
            m = torch.full((args.n, 1), mi, dtype=torch.long, device=dev)
            for j, t in enumerate(sch.timesteps):
                v = torch.cat([net(x[s:s + args.batch], torch.full((len(x[s:s + args.batch]), 1), int(t), device=dev),
                                   m[s:s + args.batch]) for s in range(0, args.n, args.batch)])
                x = manual_step(sch, v, int(t), x, step_noise[j].to(dev))
            res[mod] = x[:, 0].cpu().numpy().astype(np.float32)
        np.savez_compressed(outp, step=step, ckpt_sha256=sha, mps_cpu_max_abs_diff=dmax, **res)
        print(f'step {step}: 生成 {args.n}×2，用时 {time.time() - t0:.0f}s，MPS/CPU 前向差 {dmax:.1e}', flush=True)


def manual_step(sch, v, t, x, noise):
    """DDPM 祖先采样一步（v-prediction，与 diffusers DDPMScheduler.step 同公式，噪声由外部给定）。"""
    ac = sch.alphas_cumprod.to(x.device)
    a_t = ac[t]
    a_prev = ac[t - 1] if t > 0 else torch.tensor(1.0, device=x.device)
    beta_t = 1 - a_t / a_prev
    x0 = a_t.sqrt() * x - (1 - a_t).sqrt() * v
    c0 = a_prev.sqrt() * beta_t / (1 - a_t)
    ct = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
    mean = c0 * x0 + ct * x
    if t == 0:
        return mean
    var = (1 - a_prev) / (1 - a_t) * beta_t
    return mean + var.clamp(min=1e-20).sqrt() * noise


# ---------------------------------------------------------------- FID 特征网络
class StageCNN(nn.Module):
    def __init__(self, n_class=5):
        super().__init__()
        ch = [1, 32, 64, 96, 128]
        layers = []
        for i in range(4):
            layers += [nn.Conv1d(ch[i], ch[i + 1], 7, padding=3), nn.BatchNorm1d(ch[i + 1]), nn.ReLU(), nn.MaxPool1d(4 if i < 3 else 2)]
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(128, n_class)

    def embed(self, x):
        return self.body(x).mean(-1)

    def forward(self, x):
        return self.fc(self.embed(x))


def pool_windows(per_epoch=1, seed=0):
    """预训练池 46 人（split=train 去掉 held-out）的真实 5 秒窗及其所在段的标签。"""
    from scipy.signal import resample_poly
    add_vendor()
    from src.util import staged_mu_law
    held = set(M36.heldout_subjects())
    split = list(csv.DictReader(open(BASE / 'manifests' / 'split_subjects.csv', encoding='utf-8')))
    recs = sorted(rec for r in split if r['split'] == 'train' and r['subject_id'] not in held
                  for rec in r['record_ids'].split(';'))
    rng = np.random.default_rng(seed)
    X, Y = {m: [] for m in MODS}, []
    for rec in recs:
        with np.load(PROCESSED / f'{rec}.npz') as z:
            sig, lab = z['signal'], z['labels']
        sel = np.nonzero((lab >= 0) & (lab <= 4))[0]
        k = rng.integers(0, N_WIN, size=len(sel))
        for mi, mod in enumerate(MODS):
            x = resample_poly(sig[sel, mi].astype(np.float32), 2, 1, axis=-1).astype(np.float32)
            x = staged_mu_law(x / 100).astype(np.float32)
            X[mod].append(np.take_along_axis(x, k[:, None] * WIN + np.arange(WIN)[None, :], axis=1))
        Y.append(lab[sel])
    return {m: np.concatenate(v) for m, v in X.items()}, np.concatenate(Y).astype(np.int64), recs


def cmd_fidnet(args):
    torch.manual_seed(0)
    dev = torch.device(args.device)
    X, Y, recs = pool_windows()
    real = real_windows()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    info = dict(train_records=len(recs), n_windows=int(len(Y)), class_counts=np.bincount(Y).tolist())
    # held-out 真实窗的标签，只用于报告 CNN 的分期准确率
    idx = M36.rebuild_index()
    tab, _, _ = M36.rebuild_val(idx, M36.heldout_subjects())
    lab = {}
    for rec, g in tab.groupby('record_id'):
        with np.load(PROCESSED / f'{rec}.npz') as z:
            lab.update(dict(zip(g.index, z['labels'][g['npz_row'].to_numpy()])))
    tab['label'] = pd.Series(lab)
    w = torch.tensor(len(Y) / (5 * np.bincount(Y)), dtype=torch.float32, device=dev)
    for mi, mod in enumerate(MODS):
        net = StageCNN().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        xt, yt = torch.from_numpy(X[mod]), torch.from_numpy(Y)
        for ep in range(args.epochs):
            net.train()
            perm = torch.randperm(len(yt))
            for s in range(0, len(perm), 256):
                b = perm[s:s + 256]
                loss = nn.functional.cross_entropy(net(xt[b, None].to(dev)), yt[b].to(dev), weight=w)
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            xr = torch.from_numpy(real[mod][0][:, None])
            pred = torch.cat([net(xr[s:s + 512].to(dev)).argmax(1).cpu() for s in range(0, len(xr), 512)]).numpy()
        yr = tab.loc[tab['modality_id'] == mi, 'label'].to_numpy().astype(int)
        from sklearn.metrics import balanced_accuracy_score, f1_score
        info[mod] = dict(heldout_balanced_acc=float(balanced_accuracy_score(yr, pred)),
                         heldout_macro_f1=float(f1_score(yr, pred, average='macro')))
        torch.save(net.cpu().state_dict(), cache / f'fidnet_{mod}.pt')
        print(mod, info[mod], flush=True)
    info.update(epochs=args.epochs, arch='4×(Conv7-BN-ReLU-Pool) 32/64/96/128 → GAP(128) → Linear(5)',
                input='model domain (µV/100 staged mu-law), 5 s @ 200 Hz', class_weight='inverse frequency')
    (cache / 'fidnet_info.json').write_text(json.dumps(info, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- 指标
def features(x_model):
    """手工特征：µV 域的对数 Welch 功率谱（0.5–40 Hz）+ log std、线长、峰度、偏度、过零率。"""
    from scipy.signal import welch
    from scipy.stats import kurtosis, skew
    x = to_uv(x_model)
    f, p = welch(x, fs=FS, nperseg=256, axis=-1)
    keep = (f >= 0.5) & (f <= 40)
    logp = np.log10(p[:, keep] + 1e-12)
    bp = np.stack([p[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in BANDS.values()], 1)
    rel = np.log10(bp / bp.sum(1, keepdims=True) + 1e-12)
    td = np.stack([np.log10(x.std(1) + 1e-9), np.log10(np.abs(np.diff(x, axis=1)).mean(1) + 1e-9),
                   kurtosis(x, axis=1), skew(x, axis=1), (np.diff(np.sign(x), axis=1) != 0).mean(1)], 1)
    return dict(logpsd=logp, relband=rel, logstd=td[:, 0], hand=np.concatenate([logp, rel, td], 1))


def mmd2_rbf(a, b):
    """无偏 MMD²，带宽取合并样本两两距离的中位数。"""
    z = np.concatenate([a, b])
    d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    h = np.median(d[np.triu_indices(len(z), 1)])
    k = np.exp(-d / h)
    n, m = len(a), len(b)
    kaa, kbb, kab = k[:n, :n], k[n:, n:], k[:n, n:]
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1)) - 2 * kab.mean())


def fid(a, b):
    from scipy.linalg import sqrtm
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs = sqrtm(s1 @ s2)
    cs = cs.real if np.iscomplexobj(cs) else cs
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


def kid(a, b, n_sub=10, size=256, seed=0):
    """KID：三次多项式核的无偏 MMD²，子集平均（Bińkowski et al. 2018）。"""
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    vals = []
    for _ in range(n_sub):
        x, y = a[rng.choice(len(a), size, replace=False)], b[rng.choice(len(b), size, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (size * (size - 1)) + (kyy.sum() - np.trace(kyy)) / (size * (size - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def real_fake_auc(fa, fb, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    X = np.concatenate([fa, fb])
    y = np.r_[np.zeros(len(fa)), np.ones(len(fb))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=seed), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed), method='predict_proba')[:, 1]
    return float(roc_auc_score(y, p))


def compare(fa, fb, ea, eb):
    from scipy.stats import wasserstein_distance
    out = dict(psd_mean_abs_db=float(10 * np.abs(fa['logpsd'].mean(0) - fb['logpsd'].mean(0)).mean()),
               logstd_w1=float(wasserstein_distance(fa['logstd'], fb['logstd'])))
    for j, name in enumerate(BANDS):
        out[f'relband_{name}_w1'] = float(wasserstein_distance(fa['relband'][:, j], fb['relband'][:, j]))
    out['relband_w1_mean'] = float(np.mean([out[f'relband_{n}_w1'] for n in BANDS]))
    mu, sd = fa['hand'].mean(0), fa['hand'].std(0) + 1e-9
    out['mmd2_hand'] = mmd2_rbf((fa['hand'] - mu) / sd, (fb['hand'] - mu) / sd)
    out['real_fake_auc'] = real_fake_auc(fa['hand'], fb['hand'])
    out['fid'] = fid(ea, eb)
    out['kid'], out['kid_std'] = kid(ea, eb)
    return out


def cmd_metrics(args):
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real = real_windows()
    nets = {}
    for mod in MODS:
        n = StageCNN()
        n.load_state_dict(torch.load(cache / f'fidnet_{mod}.pt', map_location='cpu'))
        nets[mod] = n.eval()

    @torch.no_grad()
    def emb(mod, x):
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)[:, None])
        return torch.cat([nets[mod].embed(xt[s:s + 512]) for s in range(0, len(xt), 512)]).numpy().astype(np.float64)

    gens = sorted(cache.glob('gen_step*.npz'), key=lambda p: int(p.stem[8:]))
    rows, spectra = [], []
    for mod in MODS:
        x, subj = real[mod]
        a = split_ab(subj)
        fa, fb_real = features(x[a]), features(x[~a])
        ea, eb_real = emb(mod, x[a]), emb(mod, x[~a])
        r = compare(fa, fb_real, ea, eb_real)
        rows.append(dict(modality=mod, step='real_B', n=int((~a).sum()), **r))
        spectra.append(pd.DataFrame(dict(modality=mod, source='real_A', bin=np.arange(fa['logpsd'].shape[1]),
                                         mean_log10_psd=fa['logpsd'].mean(0))))
        print(mod, 'real_B', {k: round(v, 4) for k, v in r.items()}, flush=True)
        for p in gens:
            z = np.load(p)
            g = z[mod]
            if not np.isfinite(g).all():
                rows.append(dict(modality=mod, step=int(z['step']), n=len(g), note='non-finite'))
                continue
            fg = features(g)
            r = compare(fa, fg, ea, emb(mod, g))
            rows.append(dict(modality=mod, step=int(z['step']), n=len(g), **r))
            spectra.append(pd.DataFrame(dict(modality=mod, source=f'gen_{int(z["step"])}', bin=np.arange(fg['logpsd'].shape[1]),
                                             mean_log10_psd=fg['logpsd'].mean(0))))
            print(mod, int(z['step']), {k: round(v, 4) for k, v in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'gen_quality.csv', index=False, float_format='%.6g', lineterminator='\n')
    from scipy.signal import welch
    f, _ = welch(np.zeros((1, WIN)), fs=FS, nperseg=256)
    fk = f[(f >= 0.5) & (f <= 40)]
    sp = pd.concat(spectra)
    sp['freq_hz'] = fk[sp['bin'].to_numpy()]
    sp.drop(columns='bin').to_csv(out / 'mean_log_psd.csv', index=False, float_format='%.6g', lineterminator='\n')
    run = dict(reference='held-out 8 subjects (pretrain_split_w4_v01), D val units, split A/B per subject',
               n_real_a=int(split_ab(real['eeg'][1]).sum()),
               generated=[dict(file=p.name, step=int(np.load(p)['step']), ckpt_sha256=str(np.load(p)['ckpt_sha256']),
                               mps_cpu_max_abs_diff=float(np.load(p)['mps_cpu_max_abs_diff'])) for p in gens],
               sampler='DDPM ancestral, 50 steps, v-prediction, clip_sample=False, EMA weights, no condition',
               gen_seed=GEN_SEED, fidnet=json.loads((cache / 'fidnet_info.json').read_text(encoding='utf-8')),
               torch=torch.__version__, platform=platform.platform())
    (out / 'gen_quality_run.json').write_text(json.dumps(run, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--ckpt-dir', nargs='+', required=True)
    g.add_argument('--steps', nargs='+', type=int, required=True)
    g.add_argument('--n', type=int, default=512)
    g.add_argument('--batch', type=int, default=512)
    g.add_argument('--device', default='cpu')
    g.add_argument('--force', action='store_true')
    f = sub.add_parser('fidnet')
    f.add_argument('--device', default='cpu')
    f.add_argument('--epochs', type=int, default=4)
    mt = sub.add_parser('metrics')
    mt.add_argument('--out', default=str(BASE / 'reports' / 'week4' / 'gen_quality'))
    for p in (g, f, mt):
        p.add_argument('--cache', default=str(BASE / 'cache' / 'w4_gen_quality'))
    args = ap.parse_args()
    dict(gen=cmd_gen, fidnet=cmd_fidnet, metrics=cmd_metrics)[args.cmd](args)


if __name__ == '__main__':
    main()
