"""第二周 B 的公共函数:预算名单、特征读取与对齐、训练循环、六件套 run 目录。

数据权限(协议 v2):监督头只读"该预算选中训练者"的特征与标签 + 8 名验证者(评估与选模);
训练池其余人员与测试者的特征行一律不读。标准化只在选中训练者的中心样本上拟合。
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))
import evaluate as ev  # noqa: E402  官方评估 v1

from common import BASE, RUNS, SPLIT  # noqa: E402

STAGES = ev.STAGE_NAMES
PROB_COLS = ev.PROB_COLS
POOLED_DIM = 2560                 # 每模态:20 层 × 128 维(30 个 1 秒窗取均值)
VAL_REFERENCE = SPLIT / "val_reference_v1.csv"
BUDGET_SIZES = {"label10": 5, "label25": 14, "label50": 27, "label100": 54}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_version():
    """训练时的代码 commit;工作区有未提交改动时加 +dirty,方便复现时核对。"""
    import subprocess
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=BASE, capture_output=True, text=True, check=True).stdout.strip()
        return run("rev-parse", "HEAD") + ("+dirty" if run("status", "--porcelain", "--untracked-files=no") else "")
    except Exception:
        return "unknown"


def read_csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ 名单
def selected_subjects(subsets_csv, budget):
    """按 v2 名单取预算内受试者(只看 label 标记列,不重排)。"""
    rows = read_csv_rows(subsets_csv)
    chosen = [r["subject_id"] for r in rows if r[budget] == "1"]
    if len(chosen) != BUDGET_SIZES[budget]:
        raise ValueError(f"{budget} 应有 {BUDGET_SIZES[budget]} 人,实际 {len(chosen)}")
    return chosen, {r["subject_id"]: r for r in rows}


def select_rows(index_rows, chosen):
    """训练行 = 选中者的 train 行;验证行 = 全部 val 行;其余(含 test)丢弃并保证不存在 test 泄漏。"""
    chosen = set(chosen)
    train = [r for r in index_rows if r["split"] == "train" and r["subject_id"] in chosen]
    val = [r for r in index_rows if r["split"] == "val"]
    if not train or not val:
        raise ValueError("训练行或验证行为空")
    if {r["subject_id"] for r in train} != chosen:
        raise ValueError("特征索引缺少部分预算内受试者")
    if {r["subject_id"] for r in train} & {r["subject_id"] for r in val}:
        raise ValueError("训练与验证受试者重叠")
    return train, val


# ------------------------------------------------------------------ 特征
def load_pooled(folder, rows):
    """与 D/C 相同:[N,20,30,128] 在 30 个窗上取均值并拉平为 2560;按 sample_id 的 row_index 取行。"""
    folder = Path(folder)
    x = np.empty((len(rows), POOLED_DIM), dtype=np.float32)
    grouped = {}
    for i, r in enumerate(rows):
        grouped.setdefault(r["feature_path"], []).append((i, int(r["row_index"])))
    for name, entries in grouped.items():
        f = np.load(folder / name, mmap_mode="r")
        if f.ndim != 4 or f.shape[1:] != (20, 30, 128):
            raise ValueError(f"特征形状异常 {folder / name}: {f.shape}")
        for s in range(0, len(entries), 64):
            block = entries[s:s + 64]
            data = np.asarray(f[[j for _, j in block]], dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError(f"特征含 NaN/Inf: {folder / name}")
            x[[i for i, _ in block]] = data.mean(axis=2).reshape(len(block), POOLED_DIM)
        del f
    return x


def align_modalities(rows_eeg, rows_eog):
    """EEG/EOG 索引按 sample_id 严格对齐;以 EEG 行序为准;y_true/split/subject 必须逐行一致。"""
    ids = [r["sample_id"] for r in rows_eeg]
    if len(set(ids)) != len(ids) or set(ids) != {r["sample_id"] for r in rows_eog}:
        raise ValueError("EEG/EOG 的 sample_id 集合不一致或有重复")
    by_id = {r["sample_id"]: r for r in rows_eog}
    for r in rows_eeg:
        o = by_id[r["sample_id"]]
        for k in ("y_true", "split", "subject_id"):
            if r[k] != o[k]:
                raise ValueError(f"{k} 不一致: {r['sample_id']}")
    return rows_eeg, [by_id[i] for i in ids]


def load_eegdm_pair(eeg_dir, eog_dir, chosen):
    """返回 (train_rows, val_rows, x_train[N,5120], x_val[N,5120]);只读被选中的行。"""
    eeg_rows = read_csv_rows(Path(eeg_dir) / "feature_index.csv")
    eog_rows = read_csv_rows(Path(eog_dir) / "feature_index.csv")
    eeg_rows, eog_rows = align_modalities(eeg_rows, eog_rows)
    train, val = select_rows(eeg_rows, chosen)
    keep = {r["sample_id"] for r in train + val}
    eog_by_id = {r["sample_id"]: r for r in eog_rows if r["sample_id"] in keep}
    out = []
    for rows in (train, val):
        x = np.concatenate([load_pooled(eeg_dir, rows),
                            load_pooled(eog_dir, [eog_by_id[r["sample_id"]] for r in rows])], axis=1)
        out.append(x)
    return train, val, out[0], out[1]


def load_encoder_features(path, rows):
    """A 的编码器逐段特征接口:npz,含 sample_id[str] 与 x[N,F];缺任何一个所需 sample_id 即报错。"""
    with np.load(path, allow_pickle=False) as d:
        ids, x = d["sample_id"].astype(str), d["x"].astype(np.float32)
    if x.ndim != 2 or len(ids) != len(x) or len(set(ids)) != len(ids) or not np.isfinite(x).all():
        raise ValueError(f"编码器特征文件格式异常: {path}")
    pos = {s: i for i, s in enumerate(ids)}
    missing = [r["sample_id"] for r in rows if r["sample_id"] not in pos]
    if missing:
        raise ValueError(f"编码器特征缺 {len(missing)} 个 sample_id,例如 {missing[:3]}")
    return x[[pos[r["sample_id"]] for r in rows]]


def fit_standardizer(x_train):
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x_train.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    return mean, std


# ------------------------------------------------------------------ 训练
def macro_f1(y_true, y_pred):
    return ev.compute_metrics(np.asarray(y_true), np.asarray(y_pred))["macro_f1"]


def pick_device(name="auto"):
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit(model, fwd_train, y_train, fwd_val, y_val, seed, device="cpu", lr=1e-3, weight_decay=1e-4,
        batch=128, max_epochs=50, patience=10):
    """AdamW + 交叉熵;按验证 Macro-F1 保留最佳权重;patience 轮无提升即停。
    fwd_*: (索引 LongTensor -> 模型输入) 的函数,便于线性头与窗口头共用。"""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_tr = torch.as_tensor(y_train, dtype=torch.long)
    n = len(y_tr)
    history, best, best_state, best_epoch, stale = [], -1.0, None, 0, 0
    t0 = time.perf_counter()
    updates = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for ids in torch.randperm(n).split(batch):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(fwd_train(ids).to(device)), y_tr[ids].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(ids)
            updates += 1
        probs = predict_probs(model, fwd_val, len(y_val), device)
        score = macro_f1(y_val, probs.argmax(1))
        history.append(dict(epoch=epoch, train_loss=total / n, val_macro_f1=score))
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return dict(history=history, best_epoch=best_epoch, best_val_macro_f1=best, updates=updates,
                elapsed_seconds=time.perf_counter() - t0, state=best_state)


def predict_probs(model, fwd, n, device="cpu"):
    import torch
    model.eval()
    out = []
    with torch.inference_mode():
        for ids in torch.arange(n).split(1024):
            out.append(model(fwd(ids).to(device)).softmax(1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


# ------------------------------------------------------------------ run 目录
def write_prediction_table(path, val_rows, probs):
    pred = probs.argmax(1)
    df = pd.DataFrame({k: [r[k] for r in val_rows] for k in
                       ("sample_id", "subject_id", "record_id", "epoch_index", "y_true")})
    df["epoch_index"] = df["epoch_index"].astype(int)
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = pred.astype(int)
    for j, c in enumerate(PROB_COLS):
        df[c] = probs[:, j]
    cols = ["sample_id", "subject_id", "record_id", "epoch_index", "y_true", "y_pred"] + PROB_COLS
    df[cols].to_csv(path, index=False, float_format="%.8g")


def make_run_dir(run_id, runs_root=RUNS):
    d = Path(runs_root) / run_id
    if d.exists():
        raise FileExistsError(f"run 目录已存在,拒绝覆盖: {d}")
    d.mkdir(parents=True)
    return d


def finish_run(run_dir, run_id, config, result, val_rows, probs, labeled_subjects, norm, notes, status="completed"):
    """写六件套 + 名单/标准化。metrics 必须由官方 evaluate.py + 验证参考清单算出。"""
    import torch
    run_dir = Path(run_dir)
    write_prediction_table(run_dir / "predictions_val.csv", val_rows, probs)
    m = ev.evaluate(run_dir / "predictions_val.csv", reference=VAL_REFERENCE)
    m.update(run_id=run_id, best_epoch=result["best_epoch"],
             best_val_macro_f1_during_training=result["best_val_macro_f1"])
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(result["history"]).to_csv(run_dir / "history.csv", index=False)
    torch.save(dict(state=result["state"], epoch=result["best_epoch"], mean=norm[0], std=norm[1]), run_dir / "best.ckpt")
    np.savez(run_dir / "normalization.npz", mean=norm[0], std=norm[1])
    pd.DataFrame({"subject_id": labeled_subjects}).to_csv(run_dir / "labeled_subjects.csv", index=False)
    config = dict(config, run_id=run_id, status=status, evaluate_version=ev.EVALUATE_VERSION,
                  evaluate_sha256=sha256_file(Path(ev.__file__)), code_commit=code_version(),
                  best_ckpt_sha256=sha256_file(run_dir / "best.ckpt"),
                  predictions_sha256=sha256_file(run_dir / "predictions_val.csv"),
                  val_reference_sha256=sha256_file(VAL_REFERENCE),
                  actual_updates=result["updates"], actual_epochs=len(result["history"]),
                  elapsed_seconds=round(result["elapsed_seconds"], 2))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "README.md").write_text(
        f"# {run_id}\n\n状态:**{status}**(最后成功步骤:写出 predictions_val.csv 并由官方 evaluate.py v1 复核)。\n\n"
        f"- 验证 Macro-F1 {m['macro_f1']:.4f};N1 精确率/召回率/F1 = "
        f"{m['n1_precision']:.4f} / {m['n1_recall']:.4f} / {m['n1_f1']:.4f}\n"
        f"- 最佳轮次 {result['best_epoch']} / 实际 {len(result['history'])} 轮;更新数 {result['updates']}\n"
        f"- 有标签受试者:{len(labeled_subjects)} 人(见 labeled_subjects.csv);标准化只拟合这些人\n"
        f"- 样本集合 SHA256(验证参考清单):{m.get('sample_set_sha256')}\n\n{notes}\n", encoding="utf-8")
    return m
