"""W4 实验 D：公共训练脚本的公共模块（全组共用，A/B/C 也在这套代码上加开关）。

与第三周 w3d_common.py 的关系：
- 模型尺寸、扩散/优化设置、预处理、缓存格式全部沿用第三周（数值完全相同），这里是自包含的拷贝，
  方便组员单独拿走使用；第三周目录里的文件不改。
- 新增：held-out 受试者名单（pretraining validation 用，训练时剔除）、可配置路径（环境变量）。

路径（都可以用环境变量覆盖，组员在自己电脑上改环境变量即可，不用改代码）：
  SLEEP_DATA            预处理 NPZ 目录（A 的 153 个 NPZ）
  EEGDM_VENDOR          官方 EEGDM 代码（C 交付包里的 vendor/EEGDM，commit 003b5a2）
  SLEEP_PRETRAIN_CACHE  训练池缓存（00_build_pretrain_cache.py 生成；与第三周缓存逐字节相同，SHA 见下）
  SLEEP_W4_BIG          大文件输出（checkpoint）
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------- 路径 ----------------
EXP_D = Path(__file__).resolve().parents[1]          # .../week4/exp_d_pretrain
MANIFESTS = EXP_D / 'manifests'
RUNS_SMALL = EXP_D / 'runs'                           # config / history / val_mse 等小文件
ANALYSIS = EXP_D / 'analysis'

HOME = Path.home()
DATA = Path(os.environ.get('SLEEP_DATA', HOME / 'data/sleepedf_sc_processed'))
VENDOR = Path(os.environ.get('EEGDM_VENDOR', HOME / 'week1/C_experiment3/vendor/EEGDM'))
CACHE_DIR = Path(os.environ.get('SLEEP_PRETRAIN_CACHE', HOME / 'week3-1/cache/pretrain_train_v01'))
BIG = Path(os.environ.get('SLEEP_W4_BIG', HOME / 'sleep_project/week4/big'))
RUNS_BIG = BIG / 'runs'                               # checkpoint

SPLIT_CSV = MANIFESTS / 'split_subjects.csv'
LABEL_SUBSETS_CSV = MANIFESTS / 'label_subsets_v2.csv'
SPLIT_SHA = '1420db02a89907c0bf4ca66316718b711b0fcb682d61d96965ffb9a96423c10f'
LABEL_SUBSETS_SHA = 'b343f5026b7f291ad2e1f89a78ccf2876e87473bc53b0a994329f49b8cb22ed6'
CACHE_SIGNALS_SHA = 'baa935f751d3483a7437eb87293d221368b716743cb4249a960ad12634c58c61'   # 第三周缓存
CACHE_INDEX_SHA = 'e3632bce543a90b64c19dc1f1884c2f56a85eed7ce13062f6ce15f70239baa24'
UPSTREAM_COMMIT = '003b5a2a016b2f02788ea5693d8b7cd00b6bd969'

MODALITIES = {'eeg': 0, 'eog': 1}   # NPZ signal 的通道顺序 = 模态条件编号
MOD_NAMES = ('eeg', 'eog')
WIN = 1000                          # 5 秒 @ 200 Hz
N_WIN = 6                           # 每个 30 秒段切 6 个 5 秒窗
WINDOW_SECS = {5: (WIN, N_WIN), 30: (6000, 1)}   # W5：预训练窗口长度 -> (窗长点数, 每段窗数)；30 秒 = 整段不切
N_HELDOUT = 8

# ---------------- 模型配置（与第三周相同） ----------------
MODEL_CONFIGS = {
    'official_ref': dict(d_model=128, d_state=128, n_layer=20, d_cond=512),
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


def append_csv(path, row):
    path = Path(path)
    new = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


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


def train_subjects():
    recs = split_records()
    return sorted({s for s, sp in recs.values() if sp == 'train'})


def label25_subjects(sampling_seed=2024):
    """14 人预算（label25 列）名单，来自 B 的 label_subsets_v2.csv。"""
    rows = [r for r in read_csv(LABEL_SUBSETS_CSV) if int(r['sampling_seed']) == sampling_seed]
    subj = sorted({r['subject_id'] for r in rows if int(r['label25']) == 1})
    if len(subj) != 14:
        fail(f'Expected 14 labeled subjects, got {len(subj)}')
    return subj


def label_budget_subjects(column, sampling_seed=2024):
    """label_subsets_v2.csv 里某一档预算（label10 / label25 / label50）的受试者。"""
    rows = [r for r in read_csv(LABEL_SUBSETS_CSV) if int(r['sampling_seed']) == sampling_seed]
    return sorted({r['subject_id'] for r in rows if int(r[column]) == 1})


HELDOUT_W4_V01_SHA = '470c23f7d49b4de6a3a896243e379ffc3b7a637d64b3e70c813b1c03522dffa0'   # B 的 pretrain_split_w4_v01.csv


def load_heldout(path):
    """读取 pretraining validation 的 held-out 名单。

    支持两种 CSV：
    - B 的正式名单 manifests/pretrain_split_w4_v01.csv（PR #21）：54 行，取 pretrain_role == heldout_val 的 8 行，
      并核对其余 46 行都是 pretrain_train；
    - 只有 subject_id 一列的 8 行名单。
    规则（W4 共同规则 1，B 于 11-16 收紧）：8 人，全部来自训练池，且都不在 label50（27 人，含 label25、label10）里，
    这样 5/14/27 三档预算者都留在预训练池里。
    返回 (subjects, info)；info 写进每个 run 的 config.json。
    """
    path = Path(path)
    if not path.exists():
        fail(f'Held-out list not found: {path}')
    rows = read_csv(path)
    if not rows or 'subject_id' not in rows[0]:
        fail(f'{path} must be a CSV with a subject_id column')
    train = set(train_subjects())
    if 'pretrain_role' in rows[0]:
        roles = {r['subject_id'].strip(): r['pretrain_role'].strip() for r in rows}
        if len(roles) != len(rows) or set(roles) != train:
            fail(f'{path.name}: expected one row per training-pool subject (54)')
        if set(roles.values()) != {'pretrain_train', 'heldout_val'}:
            fail(f'{path.name}: pretrain_role must be pretrain_train or heldout_val')
        subj = sorted(s for s, r in roles.items() if r == 'heldout_val')
        n_rows = len(subj)
    else:
        subj = sorted({r['subject_id'].strip() for r in rows})
        n_rows = len(rows)
    if len(subj) != N_HELDOUT or n_rows != N_HELDOUT:
        fail(f'Held-out list must contain exactly {N_HELDOUT} unique subjects, got {len(subj)} ({n_rows} rows)')
    bad = [s for s in subj if s not in train]
    if bad:
        fail(f'Held-out subjects not in the training pool: {bad}')
    bad = [s for s in subj if s in set(label_budget_subjects('label50'))]
    if bad:
        fail(f'Held-out subjects overlap the label50 budget (27 subjects): {bad}')
    sha = sha256_file(path)
    official = sha == HELDOUT_W4_V01_SHA
    if not official:
        print(f'WARNING: {path.name} is not the team list pretrain_split_w4_v01.csv (SHA {sha[:12]}...); '
              'use it for smoke tests only', flush=True)
    return subj, dict(path=path.name, sha256=sha, subjects=subj, official_w4_v01=official)


def add_vendor_to_path():
    if not (VENDOR / 'model/diffusion_model.py').exists():
        fail(f'Official EEGDM code not found at {VENDOR} (set EEGDM_VENDOR)')
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
    W4 起缓存仍是全部 54 人；held-out 的 8 人在这里按受试者过滤，不重建缓存。
    """

    def __init__(self, load_to_ram=True, verify_sha=False, window_sec=5):
        if window_sec not in WINDOW_SECS:
            fail(f'window_sec must be one of {sorted(WINDOW_SECS)}, got {window_sec}')
        self.window_sec = window_sec
        self.win, self.n_win = WINDOW_SECS[window_sec]   # 5 秒：u = (row*2+m)*6+k；30 秒：u = row*2+m（k 恒为 0）
        cfg_path = CACHE_DIR / 'cache_manifest.json'
        if not cfg_path.exists():
            fail(f'Pretrain cache not found: {cfg_path} (run 00_build_pretrain_cache.py or set SLEEP_PRETRAIN_CACHE)')
        self.cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        if self.cfg['sha256_signals'] != CACHE_SIGNALS_SHA or self.cfg['sha256_epoch_index'] != CACHE_INDEX_SHA:
            fail('Cache manifest SHA differs from the team reference cache')
        if verify_sha and sha256_file(CACHE_DIR / 'epoch_index.csv') != CACHE_INDEX_SHA:
            fail('epoch_index.csv SHA256 mismatch')
        arr = np.load(CACHE_DIR / 'signals_f16.npy', mmap_mode='r')
        if arr.shape != (self.cfg['n_epochs'], 2, 6000) or arr.dtype != np.float16:
            fail(f'Cache shape/dtype mismatch: {arr.shape} {arr.dtype}')
        self.signals = np.array(arr) if load_to_ram else arr   # 真正读进内存（约 3.2 GB）
        self.index = read_csv(CACHE_DIR / 'epoch_index.csv')
        if len(self.index) != arr.shape[0]:
            fail('epoch_index.csv length mismatch')
        self.subject = np.array([r['subject_id'] for r in self.index])
        self.record = np.array([r['record_id'] for r in self.index])
        self.n_rows = arr.shape[0]
        self.n_units = self.n_rows * 2 * self.n_win

    def with_window(self, window_sec):
        """同一份内存里的信号，换一种窗口长度来看（例如 30 秒训练时仍用 5 秒标准 val 集）。"""
        import copy
        v = copy.copy(self)                                   # 浅拷贝：signals 数组共用，不多占内存
        v.window_sec = window_sec
        v.win, v.n_win = WINDOW_SECS[window_sec]
        v.n_units = v.n_rows * 2 * v.n_win
        return v

    def unit_subject(self):
        return np.repeat(self.subject, 2 * self.n_win)

    def train_units(self, heldout):
        """训练用单元：剔除 held-out 受试者之后的全部单元。"""
        keep = ~np.isin(self.subject, list(heldout))
        if keep.all() or (~keep).sum() == 0:
            fail('Held-out filtering removed nothing')
        rows = np.nonzero(keep)[0]
        nw = self.n_win
        return (rows[:, None] * 2 * nw + np.arange(2 * nw)[None, :]).reshape(-1).astype(np.int64)

    def gather(self, units):
        """units -> (x float32 [B,1,win], cond int64 [B,1])；win = 1000（5 秒）或 6000（连续 30 秒）"""
        units = np.asarray(units, dtype=np.int64)
        nw = self.n_win
        k = units % nw
        m = (units // nw) % 2
        row = units // (2 * nw)
        x = self.signals[row, m].reshape(len(units), nw, self.win)[np.arange(len(units)), k]
        return x.astype(np.float32)[:, None, :], m[:, None]

    def window_start_sec(self):
        """每个 5 秒窗位置 w = row*6+k 在本记录内的起点秒数 = epoch_index*30 + 5k（按真实时间戳，含缺口）。"""
        ep = np.array([int(r['epoch_index']) for r in self.index], dtype=np.int64)
        return (ep[:, None] * 30 + np.arange(N_WIN)[None, :] * 5).reshape(-1)

    def gather_eeg_cond(self, units, mode, pairs=None):
        """为每个单元取 EEG 条件窗。返回 (x [B,1,1000] float32, valid [B] bool)。

        same_window：同一记录、起止时间完全一致的 Fpz-Cz 窗（同一 30 秒段同一 5 秒窗）。
        shifted：查错时配对表 pairs（见 02_build_shift_pairs.py）；没有候选的窗 valid=False（置空条件）。
        """
        if self.n_win != N_WIN:
            fail('Cross-modal EEG conditions are only defined for 5 s windows')
        units = np.asarray(units, dtype=np.int64)
        w = (units // (2 * N_WIN)) * N_WIN + units % N_WIN          # 窗位置（与模态无关）
        if mode == 'same_window':
            cw = w
            valid = np.ones(len(units), dtype=bool)
        elif mode == 'shifted':
            if pairs is None:
                fail('shifted mode needs the shift-pair table')
            cw = pairs[w]
            valid = cw >= 0
            cw = np.where(valid, cw, w)                                # 无候选：随便取一个窗，掩码置 0
        else:
            fail(f'Unknown cross-modal mode {mode}')
        row, k = cw // N_WIN, cw % N_WIN
        x = self.signals[row, 0].reshape(len(units), N_WIN, WIN)[np.arange(len(units)), k]
        return x.astype(np.float32)[:, None, :], valid


# ---------------- 错时配对表（C 的跨模态规则） ----------------
SHIFT_MIN_SEC, SHIFT_MAX_SEC = 60, 300
PAIR_SEED = 20261114
PAIRS_DIR = BIG / 'shift_pairs'
PAIRS_NPY = PAIRS_DIR / 'shift_pairs_60_300_v01.npy'


def build_shift_pairs(pool, lo=SHIFT_MIN_SEC, hi=SHIFT_MAX_SEC, seed=PAIR_SEED):
    """对每个 5 秒窗位置 w，在同一记录里起点间隔 |Δ| ∈ [lo, hi] 秒的窗中（两个方向都算）用固定 seed 均匀抽一个。

    返回 (pair [n_rows*6] int64，-1 = 没有候选；gap_sec [n_rows*6] int64，带符号，条件窗起点 - 目标窗起点)。
    只用时间戳和窗口是否存在（缓存里的窗都是完整有效的），不读任何睡眠标签。
    """
    start = pool.window_start_sec()
    rec_w = np.repeat(pool.record, N_WIN)
    n = len(start)
    pair = np.full(n, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    u = rng.random(n)                                                  # 每个窗一个均匀数，按全局顺序固定
    bounds = np.flatnonzero(np.r_[True, rec_w[1:] != rec_w[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        t = start[a:b]                                                 # 记录内单调递增（已核对）
        l1 = np.searchsorted(t, t - hi, 'left')
        h1 = np.searchsorted(t, t - lo, 'right')
        l2 = np.searchsorted(t, t + lo, 'left')
        h2 = np.searchsorted(t, t + hi, 'right')
        n1, n2 = h1 - l1, h2 - l2
        tot = n1 + n2
        r = np.minimum((u[a:b] * tot).astype(np.int64), np.maximum(tot - 1, 0))
        idx = np.where(r < n1, l1 + r, l2 + (r - n1))
        pair[a:b] = np.where(tot > 0, a + idx, -1)
    gap = np.where(pair >= 0, start[np.maximum(pair, 0)] - start, 0)
    ok = pair >= 0
    if ok.any():
        if not (np.all(np.abs(gap[ok]) >= lo) and np.all(np.abs(gap[ok]) <= hi)
                and np.all(rec_w[pair[ok]] == rec_w[ok])):
            fail('Shift-pair self-check failed')
    return pair, gap


def load_shift_pairs(pool, path=PAIRS_NPY):
    path = Path(path)
    if not path.exists():
        fail(f'Shift-pair table not found: {path} (run 02_build_shift_pairs.py)')
    pair = np.load(path)
    if pair.shape != (pool.n_rows * N_WIN,):
        fail('Shift-pair table does not match the cache')
    return pair, dict(path=path.name, sha256=sha256_file(path), min_sec=SHIFT_MIN_SEC, max_sec=SHIFT_MAX_SEC,
                      pair_seed=PAIR_SEED, n_no_candidate=int((pair < 0).sum()))
