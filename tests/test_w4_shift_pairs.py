"""错时 EEG 配对规则检查（需要 A 的预处理 NPZ，缺失时跳过）。"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('sp', BASE / 'scripts' / '33_b_w4_shift_pairs.py')
SP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SP)
REC = 'SC4001'
pytestmark = pytest.mark.skipif(not (BASE / 'data' / 'processed' / f'{REC}.npz').exists(), reason='缺少预处理 NPZ')


def _targets():
    e, k, _, _ = SP.record_windows(REC)
    return pd.DataFrame(dict(subject_id=REC[:5], record_id=REC, epoch_index=e, window_k=k))


def test_constraints():
    p = SP.pair_table(_targets())
    has = p.n_candidates > 0
    assert has.all()
    assert p.loc[has, 'signed_offset_sec'].abs().between(SP.MIN_GAP, SP.MAX_GAP).all()
    assert (p.loc[has, 'cond_window_id'].str[:6] == REC).all()
    cs = 30 * p.cond_epoch_index + 5 * p.cond_window_k
    ts = 30 * p.epoch_index + 5 * p.window_k
    assert (cs - ts == p.signed_offset_sec).all()


def test_subset_and_order_independent():
    t = _targets()
    full = SP.pair_table(t).set_index('target_window_id')
    sub = t.sample(300, random_state=0).reset_index(drop=True)
    SP.pair_windows.__defaults__[1].clear()
    p = SP.pair_table(sub).set_index('target_window_id')
    assert (p.cond_window_id == full.loc[p.index, 'cond_window_id']).all()


def test_seed_changes_draw():
    e, k, _, _ = SP.record_windows(REC)
    a = SP.pair_windows(REC, e, k, seed=SP.PAIR_SEED)['signed_offset_sec']
    b = SP.pair_windows(REC, e, k, seed=SP.PAIR_SEED + 1)['signed_offset_sec']
    assert (a != b).mean() > 0.9
    assert np.array_equal(a, SP.pair_windows(REC, e, k, seed=SP.PAIR_SEED)['signed_offset_sec'])
