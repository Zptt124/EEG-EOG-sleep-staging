"""窗口索引规则的假数据检查:缺口不跨越、边界复制中心并记 mask、中心集合不变。"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("win", ROOT / "scripts" / "11_make_window_index.py")
win = importlib.util.module_from_spec(spec)
spec.loader.exec_module(win)


def rows_of(idx):
    return {r["epoch_index"]: r for r in win.windows_for_record("S00", "S001", "train", "label25", idx)}


def test_gap_is_not_bridged_and_edges_are_padded():
    r = rows_of([5, 6, 7, 10, 11, 20])        # 7→10 与 11→20 为缺口,20 是孤立片段
    assert r[5]["boundary_rule"] == "pad_previous" and r[5]["previous_sample_id"] == "S001_0005"
    assert r[6]["boundary_rule"] == "real" and (r[6]["previous_sample_id"], r[6]["next_sample_id"]) == ("S001_0005", "S001_0007")
    assert r[7]["boundary_rule"] == "pad_next" and r[7]["next_is_real"] == 0 and r[7]["next_sample_id"] == "S001_0007"
    assert r[10]["boundary_rule"] == "pad_previous" and r[10]["previous_is_real"] == 0
    assert r[11]["boundary_rule"] == "pad_next"
    assert r[20]["boundary_rule"] == "pad_both" and r[20]["previous_sample_id"] == r[20]["next_sample_id"] == "S001_0020"


def test_every_epoch_is_a_center_exactly_once_and_neighbors_are_real_ids():
    idx = [0, 1, 2, 4, 5]
    r = rows_of(idx)
    assert sorted(r) == idx
    ids = {f"S001_{i:04d}" for i in idx}
    for row in r.values():
        assert row["previous_sample_id"] in ids and row["next_sample_id"] in ids
        assert row["previous_is_real"] == int(row["previous_sample_id"] != row["center_sample_id"])


def test_unsorted_or_duplicated_epoch_index_is_rejected():
    for bad in ([1, 1, 2], [3, 2, 1]):
        try:
            win.windows_for_record("S00", "S001", "train", "label25", bad)
        except AssertionError:
            continue
        raise AssertionError(bad)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
