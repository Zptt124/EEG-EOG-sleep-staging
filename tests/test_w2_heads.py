"""12_eegdm_budget_head.py / 13_context_head.py 的合成数据端到端检查(真实特征在团队存储,不入库)。

合成数据:14 名"选中训练者"+ 3 名"其余训练者"+ 2 名验证者;标签成段连续,特征 = 类别均值 + 噪声,
所以长度 3 上下文应当不比长度 1 差。load_pooled 被替换为读取内存中的合成矩阵;
同时记录被读取的 sample_id,用来证明"其余训练者"的特征行从未被读取。
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import w2_common as w  # noqa: E402


def load_script(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


win = load_script("win", "11_make_window_index.py")
budget_head = load_script("budget_head", "12_eegdm_budget_head.py")
ctx_head = load_script("ctx_head", "13_context_head.py")


def make_world(tmp, rng):
    chosen = [f"T{i:02d}" for i in range(14)]
    others = ["X00", "X01", "X02"]
    val = ["V00", "V01"]
    n_ep = 120
    means = rng.normal(size=(5, 5120)).astype(np.float32) * 0.6
    index, windows, subsets, store = [], [], [], {}
    for s, part in [(x, "train") for x in chosen + others] + [(x, "val") for x in val]:
        rec = f"{s}1"
        # 成段标签:每段 10-20 个 epoch;每条记录在中间去掉两个 epoch 制造缺口
        labels, k = [], 0
        while len(labels) < n_ep:
            labels += [int(rng.integers(0, 5))] * int(rng.integers(10, 20))
        labels = np.array(labels[:n_ep])
        keep = np.array([i for i in range(n_ep) if i not in (60, 61)])
        feats = means[labels[keep]] + rng.normal(size=(len(keep), 5120)).astype(np.float32) * 2.0
        for row, (i, y) in enumerate(zip(keep, labels[keep])):
            sid = win.sid(rec, i)
            index.append(dict(sample_id=sid, subject_id=s, record_id=rec, epoch_index=int(i), split=part,
                              y_true=int(y), feature_path=f"{rec}.npy", row_index=row))
            store[sid] = feats[row]
        if s in chosen or part == "val":
            windows += win.windows_for_record(s, rec, part, "label25" if part == "train" else "NA", keep)
        subsets.append(dict(sampling_seed=2024, subject_id=s, record_ids=rec, n_records=1, n_epochs=len(keep),
                            sampling_rank=len(subsets) + 1, label10=0, label25=int(s in chosen),
                            label50=0, label100=0))
    d = Path(tmp)
    for sub in ("eeg", "eog"):
        (d / sub).mkdir()
        pd.DataFrame(index).to_csv(d / sub / "feature_index.csv", index=False)
    pd.DataFrame(subsets).to_csv(d / "subsets.csv", index=False)
    pd.DataFrame(windows).to_csv(d / "window_index.csv", index=False)
    val_rows = [r for r in index if r["split"] == "val"]
    pd.DataFrame(val_rows)[["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]].to_csv(
        d / "val_reference.csv", index=False)
    return chosen, others, store


def patch_features(monkeypatch_target, store, read_log):
    """EEG 目录读前 2560 维,EOG 目录读后 2560 维;记录被读取的行。"""
    def fake_load_pooled(folder, rows):
        read_log.extend(r["sample_id"] for r in rows)
        half = slice(0, 2560) if Path(folder).name == "eeg" else slice(2560, 5120)
        return np.stack([store[r["sample_id"]][half] for r in rows]).astype(np.float32)
    monkeypatch_target.load_pooled = fake_load_pooled


def run_all(tmp_path):
    rng = np.random.default_rng(0)
    d = Path(tmp_path)
    chosen, others, store = make_world(d, rng)
    log = []
    patch_features(w, store, log)
    w.VAL_REFERENCE = d / "val_reference.csv"
    ctx_head.WINDOW_INDEX = d / "window_index.csv"
    common = ["--eeg-dir", str(d / "eeg"), "--eog-dir", str(d / "eog"), "--subsets", str(d / "subsets.csv"),
              "--runs-root", str(d / "runs")]
    ctx_args = common + ["--repr", "eegdm", "--window-index", str(d / "window_index.csv"), "--device", "cpu"]
    return chosen, others, log, common, ctx_args, store


def test_budget_head_trains_reads_only_selected_people_and_writes_six_files(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    chosen, others, log, common, _, store = run_all(tmp)
    # 合成名单只有 label25(14 人,与 BUDGET_SIZES 一致),用它检验流程与数据权限
    budget_head.main(common + ["--budget", "label25", "--seed", "0"])
    run = tmp / "runs" / "W2_B_EEGDMhead_label25_sseed2024_seed0_v01"
    for f in ("README.md", "config.yaml", "history.csv", "best.ckpt", "metrics.json", "predictions_val.csv",
              "labeled_subjects.csv", "normalization.npz"):
        assert (run / f).exists(), f
    read = set(log)
    assert not any(s.startswith(o) for s in read for o in others), "其余训练者的特征行被读取了"
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert m["macro_f1"] > 0.5 and m["n_subjects"] == 2 and "sample_set_sha256" in m
    pred = pd.read_csv(run / "predictions_val.csv")
    assert list(pred.columns) == ["sample_id", "subject_id", "record_id", "epoch_index", "y_true", "y_pred",
                                  "p_W", "p_N1", "p_N2", "p_N3", "p_REM"]
    norm = np.load(run / "normalization.npz")
    # 标准化只来自 14 名选中者:用同一批合成特征手算均值应当一致
    expect = np.stack([store[sid] for sid in sorted(read) if sid.startswith("T")]).mean(axis=0)
    assert norm["mean"].shape == (5120,) and np.allclose(norm["mean"], expect, atol=1e-4)
    try:
        budget_head.main(common + ["--budget", "label25", "--seed", "0"])
    except FileExistsError:
        pass
    else:
        raise AssertionError("已存在的 run 不得被覆盖")


def test_context_heads_share_center_set_and_length3_runs(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    _, _, _, _, ctx_args, _ = run_all(tmp)
    ctx_head.main(ctx_args + ["--context", "1", "--seed", "0"])
    ctx_head.main(ctx_args + ["--context", "3", "--seed", "0"])
    p1 = pd.read_csv(tmp / "runs" / "W2_B_CTX1_EEGDM_label25_sseed2024_seed0_v01" / "predictions_val.csv")
    p3 = pd.read_csv(tmp / "runs" / "W2_B_CTX3_EEGDM_label25_sseed2024_seed0_v01" / "predictions_val.csv")
    assert list(p1.sample_id) == list(p3.sample_id)                         # 同一中心样本集合与顺序
    m1 = json.loads((tmp / "runs" / "W2_B_CTX1_EEGDM_label25_sseed2024_seed0_v01" / "metrics.json").read_text(encoding="utf-8"))
    m3 = json.loads((tmp / "runs" / "W2_B_CTX3_EEGDM_label25_sseed2024_seed0_v01" / "metrics.json").read_text(encoding="utf-8"))
    assert m1["sample_set_sha256"] == m3["sample_set_sha256"]
    assert m3["macro_f1"] >= m1["macro_f1"] - 0.02                           # 成段标签 → 上下文不应更差
    cfg = (tmp / "runs" / "W2_B_CTX3_EEGDM_label25_sseed2024_seed0_v01" / "config.yaml").read_text(encoding="utf-8")
    assert "neighbor_labels_used: false" in cfg


def test_context_head_cnn_representation_reads_encoder_npz(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    _, _, _, _, ctx_args, store = run_all(tmp)
    index = pd.read_csv(tmp / "eeg" / "feature_index.csv")
    wi = pd.read_csv(tmp / "window_index.csv")
    need = index[index.sample_id.isin(set(wi.center_sample_id))]
    proc = tmp / "processed"; proc.mkdir()
    for rec, g in need.groupby("record_id"):                                  # 合成 NPZ:只放 labels/epoch_index
        np.savez(proc / f"{rec}.npz", labels=g.y_true.to_numpy(np.int8), epoch_index=g.epoch_index.to_numpy(np.int32))
    rng = np.random.default_rng(1)
    x = np.stack([store[s][:32] for s in need.sample_id]).astype(np.float32)   # 32 维"编码器输出"
    extra = ["--window-index", str(tmp / "window_index.csv")]
    np.savez(tmp / "enc.npz", sample_id=need.sample_id.to_numpy(str), x=x)
    args = ["--repr", "cnn", "--cnn-features", str(tmp / "enc.npz"), "--processed", str(proc), "--subsets",
            str(tmp / "subsets.csv"), "--runs-root", str(tmp / "runs"), "--device", "cpu"] + extra
    ctx_head.main(args + ["--context", "3", "--seed", "0"])
    assert (tmp / "runs" / "W2_B_CTX3_CNN_label25_sseed2024_seed0_v01" / "predictions_val.csv").exists()
    np.savez(tmp / "enc_missing.npz", sample_id=need.sample_id.to_numpy(str)[1:], x=x[1:])   # 缺一个 sample_id
    bad = [a if a != str(tmp / "enc.npz") else str(tmp / "enc_missing.npz") for a in args]
    try:
        ctx_head.main(bad + ["--context", "1", "--seed", "0"])
    except ValueError as e:
        assert "缺" in str(e)
    else:
        raise AssertionError("编码器特征缺样本应报错")


def test_aggregate_rebuilds_tables_pairs_context_and_updates_only_b_rows(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    chosen, others, log, common, ctx_args, store = run_all(tmp)
    agg = load_script("agg", "14_w2_aggregate.py")
    for seed in (0, 1):
        budget_head.main(common + ["--budget", "label25", "--seed", str(seed)])
        for L in (1, 3):
            ctx_head.main(ctx_args + ["--context", str(L), "--seed", str(seed)])
    out = tmp / "out"; out.mkdir()
    reg = out / "run_registry.csv"
    header = (ROOT / "reports" / "week2" / "run_registry.csv").read_text(encoding="utf-8").splitlines()[0]
    fields = header.split(",")
    other = {k: "NA" for k in fields}; other.update(run_id="W2_A_dummy", owner="A", status="planned")
    reg.write_text(header + "\n" + ",".join(other[k] for k in fields) + "\n", encoding="utf-8")
    agg.main(["--runs-root", str(tmp / "runs"), "--out-dir", str(out), "--pair-context", "--update-registry", "--n-boot", "50"])
    ms = pd.read_csv(out / "metrics_summary.csv")
    assert len(ms) == 6 and ms.sample_set_sha256.nunique() == 1 and ms.subject_count.eq(2).all()
    pc = pd.read_csv(out / "paired_comparison.csv")
    assert len(pc) == 2 and (pc.uncertainty_unit == "subject").all() and pc.comparison_run_id.str.contains("CTX3").all()
    ss = pd.read_csv(out / "seed_summary.csv")
    assert (ss.n_seeds == 2).all() and ss.macro_f1_std.notna().all()
    assert len(ss) == 3 and set(ss["head"]) == {"linear", "bigru_context"}             # 线性头不与长度 1 BiGRU 混组
    r = pd.read_csv(reg)
    assert list(r.owner).count("A") == 1 and (r.owner == "B").sum() == 6               # 他人的行保留
    assert (tmp / "runs" / "W2_B_CTX3_EEGDM_label25_sseed2024_seed0_v01" / "n1_errors.csv").exists()
    # 不同样本集合不得配对
    run_ref = tmp / "runs" / "W2_B_CTX1_EEGDM_label25_sseed2024_seed0_v01"
    cfg, m, df, _ = agg.load_run(run_ref)
    m2 = dict(m, sample_set_sha256="0" * 64)
    try:
        agg.paired_row((cfg, m, df, None), (cfg, m2, df, None), out)
    except ValueError as e:
        assert "样本集合不同" in str(e)
    else:
        raise AssertionError("样本集合不同必须拒绝配对")


def test_pooled_npz_input_matches_raw_dir_input_exactly(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    _, _, _, common, ctx_args, store = run_all(tmp)
    ids = np.array(sorted(store))
    np.savez(tmp / "eeg_pooled.npz", sample_id=ids, x=np.stack([store[i][:2560] for i in ids]))
    np.savez(tmp / "eog_pooled.npz", sample_id=ids, x=np.stack([store[i][2560:] for i in ids]))
    idx = str(tmp / "eeg" / "feature_index.csv")
    pooled = ["--eeg-pooled", str(tmp / "eeg_pooled.npz"), "--eog-pooled", str(tmp / "eog_pooled.npz"),
              "--eeg-index", idx, "--subsets", str(tmp / "subsets.csv")]
    budget_head.main(common + ["--budget", "label25", "--seed", "0"])                       # 原始目录
    budget_head.main(pooled + ["--budget", "label25", "--seed", "1", "--runs-root", str(tmp / "runs")])
    budget_head.main(common + ["--budget", "label25", "--seed", "1", "--runs-root", str(tmp / "runs_raw")])
    a = pd.read_csv(tmp / "runs" / "W2_B_EEGDMhead_label25_sseed2024_seed1_v01" / "predictions_val.csv")
    b = pd.read_csv(tmp / "runs_raw" / "W2_B_EEGDMhead_label25_sseed2024_seed1_v01" / "predictions_val.csv")
    assert np.allclose(a[ev_cols()].to_numpy(), b[ev_cols()].to_numpy())
    cfg = (tmp / "runs" / "W2_B_EEGDMhead_label25_sseed2024_seed1_v01" / "config.yaml").read_text(encoding="utf-8")
    assert "feature_input: pooled_npz" in cfg and "eeg_source_sha256" in cfg
    ctx_head.main(["--repr", "eegdm", "--eeg-pooled", str(tmp / "eeg_pooled.npz"), "--eog-pooled", str(tmp / "eog_pooled.npz"),
                   "--eeg-index", idx, "--subsets", str(tmp / "subsets.csv"), "--window-index", str(tmp / "window_index.csv"),
                   "--runs-root", str(tmp / "runs"), "--device", "cpu", "--context", "3", "--seed", "0"])
    bad = tmp / "eog_missing.npz"                                                           # 缺 sample_id 应报错
    np.savez(bad, sample_id=ids[1:], x=np.stack([store[i][2560:] for i in ids[1:]]))
    try:
        budget_head.main(pooled[:2] + ["--eog-pooled", str(bad)] + pooled[4:] + ["--budget", "label25", "--seed", "2",
                         "--runs-root", str(tmp / "runs")])
    except ValueError as e:
        assert "缺" in str(e)
    else:
        raise AssertionError("池化文件缺样本应报错")
    try:
        budget_head.main(["--eeg-pooled", str(tmp / "eeg_pooled.npz"), "--budget", "label25", "--seed", "3",
                          "--subsets", str(tmp / "subsets.csv"), "--runs-root", str(tmp / "runs")])
    except ValueError as e:
        assert "池化模式" in str(e)
    else:
        raise AssertionError("参数不全应报错")


def ev_cols():
    return ["p_W", "p_N1", "p_N2", "p_N3", "p_REM"]


def test_window_neighbors_never_cross_gap_and_mismatched_centers_fail(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    _, _, _, _, ctx_args, _ = run_all(tmp)
    wi = pd.read_csv(tmp / "window_index.csv")
    row59 = wi[wi.center_sample_id == "T001_0059"].iloc[0]                   # 60/61 被删除 → 59 的后邻居必须是填充
    assert row59.next_is_real == 0 and row59.next_sample_id == "T001_0059"
    wi.iloc[1:].to_csv(tmp / "window_bad.csv", index=False)                  # 少一个中心样本
    args = [a if a != str(tmp / "window_index.csv") else str(tmp / "window_bad.csv") for a in ctx_args]
    try:
        ctx_head.main(args + ["--context", "3", "--seed", "0"])
    except ValueError as e:
        assert "window_index" in str(e)
    else:
        raise AssertionError("中心样本集合不一致应报错")


def _prediction_check_case(p_w, y_pred, tmp_path):
    """分类层恒等:特征就是 logits,预测表写入给定概率,检查还原 A 预测时的判定。"""
    import torch
    run = Path(tmp_path)
    torch.save({"state_dict": {"classifier.weight": torch.eye(5), "classifier.bias": torch.zeros(5)}}, run / "best.ckpt")
    pd.DataFrame([dict(sample_id="R_0000", y_pred=y_pred, p_W=p_w, p_N1=1 - p_w, p_N2=0, p_N3=0, p_REM=0)]).to_csv(
        run / "predictions_val.csv", index=False)
    x = np.log(np.array([[0.5, 0.5, 1e-12, 1e-12, 1e-12]], dtype=np.float32))     # 自己算出的前两类打平
    x[0, 1] += 2e-4                                                                # 让本地 argmax 落在 N1
    return load_script("extract", "19_extract_cnn_features.py").check_against_predictions(run, np.array(["R_0000"]), x)


def test_extraction_check_allows_only_near_ties():
    import tempfile
    n, differ, _ = _prediction_check_case(0.5003, 0, tempfile.mkdtemp())     # A 判 W,差距 6e-4 < TIE_MARGIN
    assert (n, differ) == (1, 1)
    try:
        _prediction_check_case(0.51, 0, tempfile.mkdtemp())                  # 差距 0.02,不是打平
    except ValueError as e:
        assert "无法还原" in str(e)
    else:
        raise AssertionError("非打平段类别不同应报错")


def test_gap_registry_maps_reused_e1_runs_and_skips_failed(tmp_path=None):
    import tempfile
    path = Path(tmp_path or tempfile.mkdtemp()) / "r.csv"
    pd.DataFrame([dict(run_id="E1_EEG_label100_seed0_v01", status="reused"),
                  dict(run_id="W2_A_EEG_label10_sseed2024_seed0_v01", status="completed"),
                  dict(run_id="W2_A_EOG_label10_sseed2024_seed0_v01", status="failed")]).to_csv(path, index=False)
    names = load_script("gap", "18_gap_table.py").names_in_registry(path)
    assert names == {"W2_A_EEG_label100_sseed2024_seed0_v01", "W2_A_EEG_label10_sseed2024_seed0_v01"}


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
