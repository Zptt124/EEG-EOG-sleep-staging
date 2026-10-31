"""evaluate.py 的假数据检查(不依赖真实数据)。

运行:python tests/test_evaluate.py   (也可用 pytest)
手算用例见 hand_case():10 段,期望值在注释中推导。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))
import evaluate as ev  # noqa: E402


def make_df(y_true, y_pred, subjects=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    p = np.full((n, 5), 0.05)
    p[np.arange(n), y_pred] = 0.8
    subjects = subjects if subjects is not None else ["S1"] * n
    return pd.DataFrame({
        "sample_id": [f"R{i:04d}" for i in range(n)], "subject_id": subjects,
        "record_id": [f"{s}1" for s in subjects], "epoch_index": range(n),
        "y_true": y_true, "y_pred": y_pred, **{c: p[:, i] for i, c in enumerate(ev.PROB_COLS)}})


def hand_case():
    # true: W W W N1 N1 N2 N2 N3 REM REM ; pred: W W N1 N1 N2 N2 N2 N3 REM W
    return make_df([0, 0, 0, 1, 1, 2, 2, 3, 4, 4], [0, 0, 1, 1, 2, 2, 2, 3, 4, 0])


def test_hand_checked_metrics():
    df = hand_case()
    m = ev.compute_metrics(df["y_true"], df["y_pred"])
    # 混淆矩阵(行真实、列预测)
    assert m["confusion_matrix"] == [[2, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 2, 0, 0],
                                     [0, 0, 0, 1, 0], [1, 0, 0, 0, 1]]
    # W: TP2 FP1 FN1 -> P=R=F1=2/3 ; N1: TP1 FP1 FN1 -> 1/2 ; N2: TP2 FP1 FN0 -> P=2/3 R=1 F1=0.8
    # N3: 1 ; REM: TP1 FP0 FN1 -> P=1 R=1/2 F1=2/3
    f1 = m["per_class_f1"]
    assert np.isclose(f1["W"], 2 / 3) and np.isclose(f1["N1"], 0.5) and np.isclose(f1["N2"], 0.8)
    assert np.isclose(f1["N3"], 1.0) and np.isclose(f1["REM"], 2 / 3)
    assert np.isclose(m["macro_f1"], (2 / 3 + 0.5 + 0.8 + 1 + 2 / 3) / 5)   # 0.72667
    assert np.isclose(m["accuracy"], 0.7)
    assert np.isclose(m["n1_precision"], 0.5) and np.isclose(m["n1_recall"], 0.5)
    assert m["per_class_support"] == {"W": 3, "N1": 2, "N2": 2, "N3": 1, "REM": 2}


def test_perfect_and_worst():
    y = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert ev.compute_metrics(y, y)["macro_f1"] == 1.0
    shifted = [(v + 1) % 5 for v in y]
    m = ev.compute_metrics(y, shifted)
    assert m["macro_f1"] == 0.0 and m["accuracy"] == 0.0


def test_missing_class_counts_as_zero_f1():
    # 真实标签里没有 REM、也没预测 REM:该类 F1 记 0,Macro-F1 仍除以 5(固定类别数)
    m = ev.compute_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert np.isclose(m["macro_f1"], 4 / 5)
    assert m["per_class_support"]["REM"] == 0


def test_matches_sklearn_on_random():
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 5, 5000)
    yp = np.where(rng.random(5000) < 0.6, yt, rng.integers(0, 5, 5000))
    m = ev.compute_metrics(yt, yp)
    labels = list(range(5))
    assert np.isclose(m["macro_f1"], f1_score(yt, yp, average="macro", labels=labels, zero_division=0))
    assert np.allclose(list(m["per_class_f1"].values()), f1_score(yt, yp, average=None, labels=labels))
    assert np.allclose(list(m["per_class_precision"].values()), precision_score(yt, yp, average=None, labels=labels))
    assert np.allclose(list(m["per_class_recall"].values()), recall_score(yt, yp, average=None, labels=labels))
    assert m["confusion_matrix"] == confusion_matrix(yt, yp, labels=labels).tolist()


def _raises(df):
    try:
        ev.validate_predictions(df)
    except ValueError:
        return True
    return False


def test_validation_rejects_bad_tables():
    good = hand_case()
    ev.validate_predictions(good)
    assert _raises(good.drop(columns=["p_REM"]))
    assert _raises(pd.concat([good, good.iloc[[0]]]))                        # sample_id 重复
    bad = good.copy(); bad.loc[0, "y_true"] = 5
    assert _raises(bad)                                                       # 标签越界
    bad = good.copy(); bad.loc[0, "p_W"] = 0.5
    assert _raises(bad)                                                       # 概率和 != 1
    bad = good.copy(); bad.loc[0, "y_pred"] = 3
    assert _raises(bad)                                                       # y_pred 不是 argmax
    bad = good.copy(); bad.loc[0, "p_N1"] = np.nan
    assert _raises(bad)                                                       # 缺失值
    assert _raises(good.iloc[0:0])                                            # 空表


def test_bootstrap_is_subject_level_and_deterministic():
    # 两个受试者:S1 全对,S2 全错 -> 重采样会出现 全S1/全S2/混合,区间应覆盖 0..1 的宽范围
    y = [0, 1, 2, 3, 4] * 4
    pred = y[:10] + [(v + 1) % 5 for v in y[10:]]
    df = make_df(y, pred, ["S1"] * 10 + ["S2"] * 10)
    a = ev.bootstrap_subjects(df, 500, seed=1)
    b = ev.bootstrap_subjects(df, 500, seed=1)
    assert a == b
    assert a["n_subjects"] == 2
    assert a["accuracy"]["lo"] == 0.0 and a["accuracy"]["hi"] == 1.0
    assert np.isclose(a["accuracy"]["point"], 0.5)


def test_paired_delta():
    y = [0, 1, 2, 3, 4] * 4
    subs = ["S1"] * 10 + ["S2"] * 10
    good, bad = make_df(y, y, subs), make_df(y, [(v + 1) % 5 for v in y], subs)
    d = ev.paired_bootstrap_delta(good, bad, 200, seed=0)
    assert np.isclose(d["macro_f1"]["point"], 1.0) and d["macro_f1"]["lo"] == 1.0
    assert np.isclose(ev.paired_bootstrap_delta(good, good, 50)["macro_f1"]["point"], 0.0)
    mismatched = good.copy(); mismatched["sample_id"] = "X" + mismatched["sample_id"]
    try:
        ev.paired_bootstrap_delta(good, mismatched)
        assert False, "应拒绝 sample_id 不一致"
    except ValueError:
        pass


def test_cli_roundtrip(tmp_path=None):
    import json, tempfile
    d = Path(tmp_path or tempfile.mkdtemp())
    hand_case().to_csv(d / "p.csv", index=False)
    ev.main([str(d / "p.csv"), "--out", str(d / "m.json"), "--bootstrap", "20"])
    m = json.loads((d / "m.json").read_text(encoding="utf-8"))
    assert m["class_order"] == ["W", "N1", "N2", "N3", "REM"] and np.isclose(m["accuracy"], 0.7)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\n{len(tests)} tests passed")
