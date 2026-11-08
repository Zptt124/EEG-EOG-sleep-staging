#ssl_common.py —— 第二周 实验④ 公共函数（D0/D1/D2 共用）
#特征读取、标准化、线性学生头训练、预测表保存、官方评估
#线性头的训练细节与第一周 multiseed_heads.py 完全一致（同一随机数顺序），保证 D0 能复现第一周结果

import csv
import hashlib
import importlib.util
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

STAGES = ["W", "N1", "N2", "N3", "REM"]
PRED_COLS = ["sample_id", "subject_id", "record_id", "epoch_index", "y_true",
             "y_pred", "p_W", "p_N1", "p_N2", "p_N3", "p_REM"]
#本文件位置：sleepdm-eeg-eog/week2/exp4_ssl/src/ssl_common.py → parents[3] 是 sleepdm-eeg-eog
ROOT = Path(__file__).resolve().parents[3]
EVALUATE_PATH = ROOT / "week1" / "inputs" / "B_feat-b-evaluation" / "src" / "evaluation" / "evaluate.py"


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pooled(folder, rows):
    #与第一周 multiseed_heads.load_pooled 相同：按 rows 顺序读特征，30 个窗取平均，拉平成 2560
    #通过 feature_path + row_index 定位，不按行序
    x = np.empty((len(rows), 2560), dtype=np.float32)
    grouped = {}
    for i, r in enumerate(rows):
        grouped.setdefault(r["feature_path"], []).append((i, int(r["row_index"])))
    for n_done, (name, entries) in enumerate(grouped.items(), start=1):
        f = np.load(folder / name, mmap_mode="r")
        if f.ndim != 4 or f.shape[1:] != (20, 30, 128):
            print(f"ERROR: bad feature shape in {folder / name}: {f.shape}")
            sys.exit(1)
        for start in range(0, len(entries), 64):
            block = entries[start:start + 64]
            data = np.asarray(f[[j for _, j in block]], dtype=np.float32)
            if not np.isfinite(data).all():
                print(f"ERROR: NaN/Inf found in {folder / name}")
                sys.exit(1)
            x[[i for i, _ in block]] = data.mean(axis=2).reshape(len(block), 2560)
        del f
        if n_done % 20 == 0 or n_done == len(grouped):
            print(f"  loaded {n_done}/{len(grouped)} files from {folder.name}", flush=True)
    return x


def fit_standardizer(x, mask):
    #只用 mask 选中的样本（有标签训练者）算每列均值和标准差，和第一周一样
    mean = x[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x[mask].std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    return mean, std


def train_head(x_train, y_train, x_val, y_val, seed, max_epochs=50, patience=10):
    #训练线性学生头，按验证 Macro-F1 选最佳轮次，连续 patience 轮没有严格提高就停
    #随机数调用顺序与第一周 train_head 完全相同（先设种子 → 建 Linear → 每轮 randperm）
    from sklearn.metrics import f1_score
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = torch.from_numpy(x_train)
    val_x = torch.from_numpy(x_val)
    train_y = torch.from_numpy(y_train)

    model = torch.nn.Linear(x_train.shape[1], 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, best_epoch, best_probs, best_state, stale = -1.0, 0, None, None, 0
    history = []
    n_updates = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for ids in torch.randperm(len(train_x)).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(train_x[ids]), train_y[ids])
            loss.backward()
            optimizer.step()
            n_updates += 1
            loss_sum += float(loss.detach()) * len(ids)  #只记录数值，不影响随机数和梯度
            n_seen += len(ids)

        model.eval()
        with torch.inference_mode():
            probs = torch.cat([model(b).softmax(1) for b in val_x.split(1024)]).numpy()
        score = float(f1_score(y_val, probs.argmax(1), labels=list(range(5)), average="macro", zero_division=0))
        improved = score > best
        if improved:
            best, best_epoch, best_probs, stale = score, epoch, probs, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": round(loss_sum / n_seen, 6),
                        "val_macro_f1": round(score, 6), "is_best": int(improved), "updates_so_far": n_updates})
        if stale >= patience:
            break
    curve = [h["val_macro_f1"] for h in history]
    return {"best_epoch": best_epoch, "best_val_macro_f1": best, "probs": best_probs, "state": best_state,
            "history": history, "n_updates": n_updates, "curve_median": statistics.median(curve)}


def save_predictions(path, val_rows, probs):
    #全组统一 11 列格式
    pred = probs.argmax(1)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRED_COLS)
        w.writeheader()
        for i, r in enumerate(val_rows):
            out = {k: r[k] for k in PRED_COLS[:5]}
            out["y_pred"] = int(pred[i])
            out.update({f"p_{s}": float(probs[i][j]) for j, s in enumerate(STAGES)})
            w.writerow(out)


def load_official_evaluate():
    #按文件路径加载 B 的官方 evaluate.py v1
    if not EVALUATE_PATH.exists():
        print(f"ERROR: official evaluate.py not found at {EVALUATE_PATH}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("official_evaluate", EVALUATE_PATH)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def predict_probs(state, X_all, idx, cols, batch=8192):
    #用训练好的线性头权重，对 X_all 中 idx 这些行、cols 这些列做推理，分块避免大内存复制
    model = torch.nn.Linear(cols.stop - cols.start, 5)
    model.load_state_dict(state)
    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(idx), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X_all[idx[s:s + batch], cols]))
            out.append(model(xb).softmax(1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 5), dtype=np.float32)


def train_student(x_l, y_l, X_all, u_idx, u_label, x_val, y_val, seed, mu=3, lam=1.0,
                  max_epochs=50, patience=10):
    #D1/D2 学生头：混合批次，每步 = 128 个真标签 + mu*128 个伪标签
    #真标签部分的随机顺序与 D0 完全相同（同一个全局随机数流）；伪标签用独立的随机数流
    #每轮步数 = 真标签批次数，所以每轮更新次数与 D0 相同；早停规则与 D0 相同
    from sklearn.metrics import f1_score
    if len(u_idx) == 0:
        print("ERROR: no pseudo-labeled samples for the student")
        sys.exit(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen_u = torch.Generator().manual_seed(seed + 100003)  #伪标签流独立，不扰动真标签流
    train_x = torch.from_numpy(x_l)
    train_y = torch.from_numpy(y_l)
    val_x = torch.from_numpy(x_val)
    u_idx_t = torch.from_numpy(np.asarray(u_idx, dtype=np.int64))
    u_y_t = torch.from_numpy(np.asarray(u_label, dtype=np.int64))
    ub = mu * 128

    model = torch.nn.Linear(x_l.shape[1], 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, best_epoch, best_probs, best_state, stale = -1.0, 0, None, None, 0
    history, n_updates, pseudo_seen = [], 0, 0
    u_perm, u_pos = torch.randperm(len(u_idx_t), generator=gen_u), 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        sums = {"loss": 0.0, "loss_l": 0.0, "loss_u": 0.0}
        n_steps = 0
        for ids in torch.randperm(len(train_x)).split(128):
            #取下一批伪标签；用完一遍就重新打乱
            if u_pos + ub > len(u_perm):
                u_perm, u_pos = torch.randperm(len(u_idx_t), generator=gen_u), 0
            take = u_perm[u_pos:u_pos + ub] if len(u_perm) >= ub else u_perm
            u_pos += ub
            xu = torch.from_numpy(np.ascontiguousarray(X_all[u_idx_t[take].numpy()]))
            yu = u_y_t[take]
            optimizer.zero_grad(set_to_none=True)
            loss_l = torch.nn.functional.cross_entropy(model(train_x[ids]), train_y[ids])
            loss_u = torch.nn.functional.cross_entropy(model(xu), yu)
            loss = loss_l + lam * loss_u
            loss.backward()
            optimizer.step()
            n_updates += 1
            n_steps += 1
            pseudo_seen += len(take)
            sums["loss"] += float(loss.detach())
            sums["loss_l"] += float(loss_l.detach())
            sums["loss_u"] += float(loss_u.detach())

        model.eval()
        with torch.inference_mode():
            probs = torch.cat([model(b).softmax(1) for b in val_x.split(1024)]).numpy()
        score = float(f1_score(y_val, probs.argmax(1), labels=list(range(5)), average="macro", zero_division=0))
        improved = score > best
        if improved:
            best, best_epoch, best_probs, stale = score, epoch, probs, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": round(sums["loss"] / n_steps, 6),
                        "train_loss_labeled": round(sums["loss_l"] / n_steps, 6),
                        "train_loss_pseudo": round(sums["loss_u"] / n_steps, 6),
                        "val_macro_f1": round(score, 6), "is_best": int(improved), "updates_so_far": n_updates,
                        "pseudo_samples_seen": pseudo_seen})
        if stale >= patience:
            break
    curve = [h["val_macro_f1"] for h in history]
    return {"best_epoch": best_epoch, "best_val_macro_f1": best, "probs": best_probs, "state": best_state,
            "history": history, "n_updates": n_updates, "curve_median": statistics.median(curve),
            "pseudo_samples_seen": pseudo_seen}
