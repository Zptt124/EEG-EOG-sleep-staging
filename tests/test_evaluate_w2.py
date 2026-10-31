"""第二周追加的 evaluate.py 负面检查:重复 ID、缺样本、多样本、错标签、概率异常必须报错。

运行:python tests/test_evaluate_w2.py   (也可用 pytest)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))
import evaluate as ev  # noqa: E402
from test_evaluate import hand_case, make_df  # noqa: E402


def ref_of(df):
    return df[["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]].copy()


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def test_reference_accepts_exact_set_in_any_order():
    df = hand_case()
    sha = ev.check_against_reference(df.sample(frac=1, random_state=0), ref_of(df))
    assert sha == ev.sample_set_sha256(df["sample_id"]) and len(sha) == 64


def test_reference_rejects_missing_extra_and_wrong_labels():
    df, ref = hand_case(), ref_of(hand_case())
    assert raises(lambda: ev.check_against_reference(df.iloc[1:], ref))              # 缺样本
    assert raises(lambda: ev.check_against_reference(df, ref.iloc[1:]))              # 多样本
    bad = df.copy(); bad.loc[0, "y_true"] = (bad.loc[0, "y_true"] + 1) % 5
    assert raises(lambda: ev.check_against_reference(bad, ref))                      # 标签与真值不一致
    bad = df.copy(); bad.loc[0, "record_id"] = "OTHER"
    assert raises(lambda: ev.check_against_reference(bad, ref))                      # 元数据不一致
    dup = pd.concat([ref, ref.iloc[[0]]])
    assert raises(lambda: ev.check_against_reference(df, dup))                       # 清单重复 ID


def test_set_sha_changes_with_set_not_with_order():
    ids = [f"R{i:04d}" for i in range(10)]
    assert ev.sample_set_sha256(ids) == ev.sample_set_sha256(ids[::-1])
    assert ev.sample_set_sha256(ids) != ev.sample_set_sha256(ids[:-1])


def test_probability_anomalies_are_errors():
    good = hand_case()
    for value in (np.inf, -np.inf, np.nan, -0.1, 1.2):
        bad = good.copy(); bad.loc[0, "p_N1"] = value
        assert raises(lambda: ev.validate_predictions(bad)), value
    bad = good.copy(); bad[ev.PROB_COLS] = bad[ev.PROB_COLS] * 0.5                   # 行和 0.5
    assert raises(lambda: ev.validate_predictions(bad))
    bad = good.copy(); bad["y_true"] = bad["y_true"].astype(float) + 0.5             # 非整数标签
    assert raises(lambda: ev.validate_predictions(bad))
    bad = good.copy(); bad.loc[0, "sample_id"] = bad.loc[1, "sample_id"]             # 重复 ID
    assert raises(lambda: ev.validate_predictions(bad))


def test_cli_with_reference_writes_set_hash_and_fails_on_missing():
    d = Path(tempfile.mkdtemp())
    df = hand_case()
    df.to_csv(d / "p.csv", index=False)
    ref_of(df).to_csv(d / "ref.csv", index=False)
    ev.main([str(d / "p.csv"), "--reference", str(d / "ref.csv"), "--out", str(d / "m.json")])
    m = json.loads((d / "m.json").read_text(encoding="utf-8"))
    assert m["sample_set_sha256"] == ev.sample_set_sha256(df["sample_id"])
    df.iloc[1:].to_csv(d / "short.csv", index=False)
    try:
        ev.main([str(d / "short.csv"), "--reference", str(d / "ref.csv")])
    except SystemExit as e:
        assert "缺样本" in str(e)
    else:
        raise AssertionError("缺样本应当报错退出")


def test_metrics_unchanged_by_reference_check():
    d = Path(tempfile.mkdtemp())
    df = make_df([0, 1, 2, 3, 4, 1, 1, 2], [0, 1, 2, 3, 4, 0, 1, 1])
    df.to_csv(d / "p.csv", index=False)
    ref_of(df).to_csv(d / "ref.csv", index=False)
    a = ev.evaluate(d / "p.csv")
    b = ev.evaluate(d / "p.csv", reference=d / "ref.csv")
    for k in ("macro_f1", "accuracy", "confusion_matrix", "n_epochs"):
        assert a[k] == b[k]


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\n{len(tests)} tests passed")
