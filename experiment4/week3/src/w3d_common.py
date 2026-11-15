"""Week 3-1 实验 D（从零扩散预训练）公共模块。

- 路径：小文件（脚本、清单、日志、汇总）放 D: 盘；大文件（预处理缓存、checkpoint）放 WSL 内部。
- 模型：直接复用 C 交付包里的官方 EEGDM 代码（vendor/EEGDM，commit 003b5a2），只改尺寸参数。
- 预处理：与 C 的 extract_features.py 完全一致：100->200 Hz resample_poly(2,1)，uV/100，官方 staged_mu_law。
- 预训练只读 NPZ 的 signal 和 epoch_index，从不读取 labels。
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# ---------------- 路径 ----------------
EXP_D = Path(__file__).resolve().parents[1]          # <D_DRIVE>/.../week3-1/exp_d_pretrain
WEEK31 = EXP_D.parent                                 # <D_DRIVE>/.../week3-1
LOGS = WEEK31 / 'logs'
MANIFESTS = EXP_D / 'manifests'
ANALYSIS = EXP_D / 'analysis'

HOME = Path.home()                                    # ~
DATA = HOME / 'data/sleepedf_sc_processed'            # A 的 153 个 NPZ（软链接）
VENDOR = HOME / 'week1/C_experiment3/vendor/EEGDM'    # 官方 EEGDM 代码
BIG = HOME / 'week3-1'                                # WSL 内部的大文件目录
CACHE_DIR = BIG / 'cache/pretrain_train_v01'          # 训练池预处理缓存
RUNS_BIG = BIG / 'runs'                               # checkpoint

SPLIT_CSV = MANIFESTS / 'split_subjects.csv'
LABEL_SUBSETS_CSV = MANIFESTS / 'label_subsets_v2.csv'
SPLIT_SHA = '1420db02a89907c0bf4ca66316718b711b0fcb682d61d96965ffb9a96423c10f'
LABEL_SUBSETS_SHA = 'b343f5026b7f291ad2e1f89a78ccf2876e87473bc53b0a994329f49b8cb22ed6'
TEAM_MAIN_COMMIT = 'fe0a9aaa264047575cdd5e1e15e2ada4462653d3'
UPSTREAM_COMMIT = '003b5a2a016b2f02788ea5693d8b7cd00b6bd969'

MODALITIES = {'eeg': 0, 'eog': 1}   # NPZ signal 的通道顺序 = 模态条件编号
WIN = 1000                          # 5 秒 @ 200 Hz
N_WIN = 6                           # 每个 30 秒段切 6 个 5 秒窗

# ---------------- 模型配置 ----------------
# 官方：d_model=128, d_state=128, n_layer=20, d_cond=512, d_cond_embed=128, n_class=22
# 小 backbone：保留双向 diag S4 + Wavenet 门控残差结构，主要缩小 hidden，n_class=2（0=EEG, 1=EOG）
MODEL_CONFIGS = {
    'official_ref': dict(d_model=128, d_state=128, n_layer=20, d_cond=512),  # 只用于测速参照
    'm96x16': dict(d_model=96, d_state=64, n_layer=16, d_cond=256),
    'm64x16': dict(d_model=64, d_state=64, n_layer=16, d_cond=256),
    'm64x12': dict(d_model=64, d_state=64, n_layer=12, d_cond=256),
}


def model_kwargs(name):
    if name not in MODEL_CONFIGS:
        fail(f'Unknown model config {name}; choose from {list(MODEL_CONFIGS)}')
    c = MODEL_CONFIGS[name]
    return dict(in_channels=1, d_model=c['d_model'], d_state=c['d_state'], n_layer=c['n_layer'],
                n_ssm=None, kernel_init='diag-lin', kernel_mode='diag', bidirectional=True,
                d_cond=c['d_cond'], d_cond_embed=128, local_cond_ch=0,
                n_class=(22 if name == 'official_ref' else 2), have_null_class=False, self_gated=False)


# 与官方 conf/pretrain/base.yaml 相同的扩散与优化设置
NOISE_SCH_KWARGS = dict(num_train_timesteps=50, beta_start=0.0001, beta_end=0.05,
                        beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
EMA_KWARGS = dict(beta=0.999, update_after_step=100, update_every=10)
OPT_KWARGS = dict(lr=1e-4, weight_decay=0.0)
GRAD_CLIP = 1.0


# ---------------- 工具函数 ----------------
def fail(msg):
    print(f'ERROR: {msg}', flush=True)
    sys.exit(1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    if not rows:
        fail(f'No rows to write to {path}')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def check_manifests():
    for path, sha in ((SPLIT_CSV, SPLIT_SHA), (LABEL_SUBSETS_CSV, LABEL_SUBSETS_SHA)):
        if not path.exists():
            fail(f'Missing manifest {path}')
        if sha256_file(path) != sha:
            fail(f'SHA256 mismatch for {path}')


def split_records():
    """返回 {record_id: (subject_id, split)}，并检查受试者与记录一致。"""
    check_manifests()
    out = {}
    for row in read_csv(SPLIT_CSV):
        for rec in row['record_ids'].split(';'):
            if rec in out or rec[:5] != row['subject_id']:
                fail(f'Invalid split row for {rec}')
            out[rec] = (row['subject_id'], row['split'])
    return out


def label25_subjects(sampling_seed=2024):
    """14 人预算（label25 列）名单，来自 B 的 label_subsets_v2.csv。"""
    rows = [r for r in read_csv(LABEL_SUBSETS_CSV) if int(r['sampling_seed']) == sampling_seed]
    subj = sorted({r['subject_id'] for r in rows if int(r['label25']) == 1})
    if len(subj) != 14:
        fail(f'Expected 14 labeled subjects, got {len(subj)}')
    return subj


def add_vendor_to_path():
    if not (VENDOR / 'model/diffusion_model.py').exists():
        fail(f'Official EEGDM code not found at {VENDOR}')
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def transform_uv(signal_uv):
    """(n, 3000) uV @100 Hz -> (n, 6000) float32，与 C 的提取流程完全一致。"""
    from scipy.signal import resample_poly
    add_vendor_to_path()
    from src.util import staged_mu_law
    x = np.asarray(signal_uv, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3000 or not np.isfinite(x).all():
        fail('Expected finite (n, 3000) microvolt input')
    x = resample_poly(x, 2, 1, axis=-1).astype(np.float32)
    return np.ascontiguousarray(staged_mu_law(x / 100).astype(np.float32))


def build_model(name, seed):
    """按固定 seed 构建小 backbone（官方 Wavenet 类）。"""
    import torch
    add_vendor_to_path()
    from model.diffusion_model import Wavenet
    torch.manual_seed(seed)
    return Wavenet(**model_kwargs(name))


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def make_scheduler():
    from diffusers import DDPMScheduler
    return DDPMScheduler(**NOISE_SCH_KWARGS)


def make_optimizer(model):
    add_vendor_to_path()
    from model.util import setup_optimizer   # 官方：S4 参数单独分组
    return setup_optimizer(model, **OPT_KWARGS)


class TrainPool:
    """训练池缓存：signals (N, 2, 6000) float16，已做 200 Hz + uV/100 + staged mu-law。

    一个“单元” = (段行号 row, 模态 m, 5 秒窗 k)，编号 u = (row * 2 + m) * 6 + k。
    """

    def __init__(self, load_to_ram=True):
        cfg_path = CACHE_DIR / 'cache_manifest.json'
        if not cfg_path.exists():
            fail(f'Pretrain cache not found: {cfg_path}. Run 02_build_pretrain_cache.py first')
        self.cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        arr = np.load(CACHE_DIR / 'signals_f16.npy', mmap_mode='r')
        if arr.shape != (self.cfg['n_epochs'], 2, 6000) or arr.dtype != np.float16:
            fail(f'Cache shape/dtype mismatch: {arr.shape} {arr.dtype}')
        self.signals = np.array(arr) if load_to_ram else arr   # 真正读进内存（约 3.2 GB）
        self.index = read_csv(CACHE_DIR / 'epoch_index.csv')
        if len(self.index) != arr.shape[0]:
            fail('epoch_index.csv length mismatch')
        self.subject = np.array([r['subject_id'] for r in self.index])
        self.n_units = arr.shape[0] * 2 * N_WIN

    def units_for_fraction(self, fraction, seed):
        """在每个训练受试者内部固定抽取 fraction 比例的单元（fraction=1 即全部）。"""
        all_units = np.arange(self.n_units, dtype=np.int64)
        if fraction >= 1.0:
            return all_units
        rng = np.random.default_rng(seed)
        unit_subject = np.repeat(self.subject, 2 * N_WIN)
        keep = []
        for s in sorted(set(self.subject)):
            u = all_units[unit_subject == s]
            k = int(round(len(u) * fraction))
            keep.append(np.sort(rng.choice(u, size=k, replace=False)))
        return np.concatenate(keep)

    def gather(self, units):
        """units -> (x float32 [B,1,1000], cond int64 [B,1])"""
        units = np.asarray(units, dtype=np.int64)
        k = units % N_WIN
        m = (units // N_WIN) % 2
        row = units // (2 * N_WIN)
        x = self.signals[row, m].reshape(len(units), N_WIN, WIN)[np.arange(len(units)), k]
        return x.astype(np.float32)[:, None, :], m[:, None]
