"""第四周预训练 held-out 名单规则检查（不需要信号文件）。"""
import csv
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('h', BASE / 'scripts' / '31_b_w4_pretrain_heldout.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def test_heldout_rules():
    rows, heldout, cand = H.select_heldout()
    assert len(rows) == 54 and len(heldout) == 8 and len(cand) == 27
    held = [r for r in rows if r['pretrain_role'] == 'heldout_val']
    assert {r['subject_id'] for r in held} == set(heldout)
    assert all(r['label50'] == '0' and r['label25'] == '0' and r['label10'] == '0' for r in held)
    assert sorted(r['stratum'] for r in held) == list(range(8))


def test_committed_files_match():
    rows, heldout, _ = H.select_heldout()
    committed = list(csv.DictReader(open(H.OUT_SPLIT, encoding='utf-8')))
    assert [r['subject_id'] for r in committed if r['pretrain_role'] == 'heldout_val'] == heldout
    meta = json.loads(H.OUT_META.read_text(encoding='utf-8'))
    assert meta['heldout_subjects'] == heldout
    win = list(csv.DictReader(open(H.OUT_WINDOWS, encoding='utf-8')))
    assert len(win) == meta['windows']['n_rows'] == 4000
    assert {r['subject_id'] for r in win} == set(heldout)
