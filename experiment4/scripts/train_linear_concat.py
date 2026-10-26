#train_linear_concat.py —— 把EEG和EOG的EEGDM特征按sample_id拼接，训练线性分类器
#训练方法照搬C的train_linear100.py：时间均值池化、只用训练集算标准化参数、Linear分类头、AdamW、按验证集Macro-F1早停
#唯一区别：输入从2560维(只有EEG)变成5120维(EEG+EOG)
#用法：python train_linear_concat.py --eeg EEG特征文件夹 --eog EOG特征文件夹 --output 输出文件夹

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

STAGES = ["W", "N1", "N2", "N3", "REM"]


def read_index(folder):
    #读取特征文件夹里的feature_index.csv，返回一个列表，每个元素是一行（字典形式）
    #encoding='utf-8-sig'：能兼容文件开头可能带的BOM标记（Windows常见），C的脚本也是这样读的
    with (folder / "feature_index.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_pooled(folder, rows):
    #按rows的顺序，把每一段的特征读出来，并在时间维上取平均：[20,30,128] -> [20,128] -> 拉平成2560个数
    #这段逻辑和C的train_linear100.py一样
    x = np.empty((len(rows), 2560), dtype=np.float32)

    #先按"在哪个文件"分组，同一个.npy文件只打开一次，效率更高
    #grouped的样子：{"SC4001.npy": [(在rows里的位置, 在文件里的行号), ...], ...}
    grouped = {}
    for i, r in enumerate(rows):
        grouped.setdefault(r["feature_path"], []).append((i, int(r["row_index"])))

    for n_done, (name, entries) in enumerate(grouped.items(), start=1):
        f = np.load(folder / name, mmap_mode="r")
        if f.ndim != 4 or f.shape[1:] != (20, 30, 128):
            print(f"WARNING: bad feature shape in {folder / name}: {f.shape}")
            sys.exit(1)
        #每次取64段，避免一次读太多占内存
        for start in range(0, len(entries), 64):
            block = entries[start:start + 64]
            data = np.asarray(f[[j for _, j in block]], dtype=np.float32)
            if not np.isfinite(data).all():
                print(f"WARNING: NaN/Inf found in {folder / name}")
                sys.exit(1)
            #data.mean(axis=2)：在第2维(30个时间窗)上取平均，[64,20,30,128] -> [64,20,128]
            #reshape(...,2560)：把20层x128维拉平
            x[[i for i, _ in block]] = data.mean(axis=2).reshape(len(block), 2560)
        del f
        if n_done % 20 == 0:
            print(f"  loaded {n_done}/{len(grouped)} files from {folder.name}", flush=True)
    return x


def main():
    #argparse：让脚本能在命令行上接收参数（--eeg、--eog、--output）
    p = argparse.ArgumentParser()
    p.add_argument("--eeg", type=Path, required=True)
    p.add_argument("--eog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    #输出文件夹已经有完成标志就拒绝运行，防止覆盖已有结果（和C的保护一样）
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        print("WARNING: output folder already has a finished run, choose a new folder")
        sys.exit(1)

    #---------- 第一部分：读两边的索引，按sample_id对齐 ----------
    rows_eeg = read_index(args.eeg)
    rows_eog = read_index(args.eog)
    print(f"EEG index rows: {len(rows_eeg)}")
    print(f"EOG index rows: {len(rows_eog)}")

    ids_eeg = {r["sample_id"] for r in rows_eeg}
    ids_eog = {r["sample_id"] for r in rows_eog}
    if ids_eeg != ids_eog or len(ids_eeg) != len(rows_eeg):
        print(f"WARNING: sample_id sets differ or contain duplicates. Only in EEG: {len(ids_eeg - ids_eog)}, only in EOG: {len(ids_eog - ids_eeg)}")
        sys.exit(1)
    print("OK: sample_id sets match exactly")

    #以EEG的行顺序为准，把EOG的行重新排成同样的顺序
    eog_by_id = {r["sample_id"]: r for r in rows_eog}
    rows_eog = [eog_by_id[r["sample_id"]] for r in rows_eeg]

    #逐行核对：真实标签、训练/验证划分、受试者必须完全一致
    for re_, ro in zip(rows_eeg, rows_eog):
        for key in ("y_true", "split", "subject_id"):
            if re_[key] != ro[key]:
                print(f"WARNING: {key} mismatch for {re_['sample_id']}: EEG={re_[key]} EOG={ro[key]}")
                sys.exit(1)
    print("OK: y_true, split and subject_id match for every sample")

    #不允许出现测试集，训练集和验证集的受试者不能重叠
    splits = {r["split"] for r in rows_eeg}
    if splits != {"train", "val"}:
        print(f"WARNING: unexpected splits {splits}")
        sys.exit(1)
    subj = {s: {r["subject_id"] for r in rows_eeg if r["split"] == s} for s in ("train", "val")}
    if subj["train"] & subj["val"]:
        print("WARNING: train and val subjects overlap")
        sys.exit(1)
    print(f"OK: train subjects {len(subj['train'])}, val subjects {len(subj['val'])}, no overlap")

    #---------- 第二部分：读特征并拼接 ----------
    print("Loading EEG features...", flush=True)
    x_eeg = load_pooled(args.eeg, rows_eeg)
    print("Loading EOG features...", flush=True)
    x_eog = load_pooled(args.eog, rows_eog)

    #np.concatenate(..., axis=1)：沿列方向并排拼接，[N,2560] + [N,2560] -> [N,5120]
    #前2560列是EEG，后2560列是EOG
    x = np.concatenate([x_eeg, x_eog], axis=1)
    del x_eeg, x_eog
    print(f"Concatenated feature shape: {x.shape}")

    #---------- 第三部分：标准化（只用训练集算均值和标准差） ----------
    mask = np.array([r["split"] == "train" for r in rows_eeg])
    y = np.array([int(r["y_true"]) for r in rows_eeg], dtype=np.int64)
    mean = x[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x[mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)  #防止除以0
    x -= mean
    x /= std

    train_x = torch.from_numpy(x[mask].copy())
    val_x = torch.from_numpy(x[~mask].copy())
    train_y = torch.from_numpy(y[mask].copy())
    val_y = y[~mask].copy()
    del x

    #---------- 第四部分：训练（超参数和C完全一样） ----------
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    model = torch.nn.Linear(5120, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    history, best, stale = [], -1.0, 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        total = 0.0
        for ids in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(train_x[ids]), train_y[ids])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)

        model.eval()
        with torch.inference_mode():
            pred = torch.cat([model(b).argmax(1) for b in val_x.split(1024)]).numpy()
        score = float(f1_score(val_y, pred, labels=list(range(5)), average="macro", zero_division=0))
        item = dict(epoch=epoch, train_loss=total / len(train_x), val_macro_f1=score)
        history.append(item)
        print(json.dumps(item), flush=True)

        if score > best:
            best, stale = score, 0
            torch.save(dict(model=model.state_dict(), mean=torch.from_numpy(mean),
                            std=torch.from_numpy(std), epoch=epoch), args.output / "best_head.pt")
        else:
            stale += 1
        if stale >= 10:
            break

    #---------- 第五部分：用最佳那一轮的分类器出验证集预测和指标 ----------
    ckpt = torch.load(args.output / "best_head.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    with torch.inference_mode():
        probs = torch.cat([model(b).softmax(1) for b in val_x.split(1024)]).numpy()
    pred = probs.argmax(1)

    p, r, f, support = precision_recall_fscore_support(val_y, pred, labels=list(range(5)), zero_division=0)
    metrics = dict(
        accuracy=float((pred == val_y).mean()),
        macro_f1=float(f.mean()),
        per_class_f1={s: float(v) for s, v in zip(STAGES, f)},
        per_class_precision={s: float(v) for s, v in zip(STAGES, p)},
        per_class_recall={s: float(v) for s, v in zip(STAGES, r)},
        per_class_support={s: int(v) for s, v in zip(STAGES, support)},
        confusion_matrix=confusion_matrix(val_y, pred, labels=list(range(5))).tolist(),
    )

    #预测表格式和A的统一格式一样，方便放进align/fuse/flips/confusion脚本
    val_rows = [r for r in rows_eeg if r["split"] == "val"]
    with (args.output / "predictions_val.csv").open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["sample_id", "subject_id", "record_id", "epoch_index",
                                                   "y_true", "y_pred", "p_W", "p_N1", "p_N2", "p_N3", "p_REM"])
        writer.writeheader()
        for i, row in enumerate(val_rows):
            out = {k: row[k] for k in ("sample_id", "subject_id", "record_id", "epoch_index", "y_true")}
            out["y_pred"] = int(pred[i])
            out.update({f"p_{s}": float(probs[i][j]) for j, s in enumerate(STAGES)})
            writer.writerow(out)

    summary = dict(status="complete", provisional=True, test_used=False,
                   input="EEG cond14 + EOG cond14, mean over 30 windows, concat 5120",
                   eeg_features=str(args.eeg), eog_features=str(args.eog),
                   train_samples=len(train_x), val_samples=len(val_x),
                   best_epoch=int(ckpt["epoch"]), val_macro_f1=metrics["macro_f1"],
                   training_seconds=time.perf_counter() - start_time)
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "metrics_provisional.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()