#multiseed_heads.py —— 多随机种子重复训练线性分类头，检验结果是否稳定
#特征只读一次，然后对 EEG、EOG、EEG+EOG拼接 三种输入，各用多个种子训练
#训练方法和C的train_linear100.py完全一样（时间均值池化、训练集标准化、Linear、AdamW、早停）
#每个种子还会用EEG和EOG两个分类器的概率做一次平均融合
#用法：python multiseed_heads.py --eeg EEG特征文件夹 --eog EOG特征文件夹 --output 输出文件夹 --seeds 0 1 2 3 4

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

STAGES = ["W", "N1", "N2", "N3", "REM"]


def read_index(folder):
    with (folder / "feature_index.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_pooled(folder, rows):
    #和train_linear_concat.py一样：按rows顺序读特征，时间维取平均，拉平成2560
    x = np.empty((len(rows), 2560), dtype=np.float32)
    grouped = {}
    for i, r in enumerate(rows):
        grouped.setdefault(r["feature_path"], []).append((i, int(r["row_index"])))
    for n_done, (name, entries) in enumerate(grouped.items(), start=1):
        f = np.load(folder / name, mmap_mode="r")
        if f.ndim != 4 or f.shape[1:] != (20, 30, 128):
            print(f"WARNING: bad feature shape in {folder / name}: {f.shape}")
            sys.exit(1)
        for start in range(0, len(entries), 64):
            block = entries[start:start + 64]
            data = np.asarray(f[[j for _, j in block]], dtype=np.float32)
            if not np.isfinite(data).all():
                print(f"WARNING: NaN/Inf found in {folder / name}")
                sys.exit(1)
            x[[i for i, _ in block]] = data.mean(axis=2).reshape(len(block), 2560)
        del f
        if n_done % 40 == 0:
            print(f"  loaded {n_done}/{len(grouped)} files from {folder.name}", flush=True)
    return x


def standardize(x, mask):
    #只用训练集算均值和标准差，再作用到全部数据（和C一样）
    mean = x[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x[mask].std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    return (x - mean) / std


def train_head(x, y, mask, seed, max_epochs=50):
    #训练一个线性分类头，返回：最佳轮次、最佳验证集Macro-F1、最佳那一轮的验证集概率、每一轮的验证集分数
    #超参数和C完全一样
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = torch.from_numpy(x[mask])
    val_x = torch.from_numpy(x[~mask])
    train_y = torch.from_numpy(y[mask])
    val_y = y[~mask]

    model = torch.nn.Linear(x.shape[1], 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, best_epoch, best_probs, stale, curve = -1.0, 0, None, 0, []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for ids in torch.randperm(len(train_x)).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(train_x[ids]), train_y[ids])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            probs = torch.cat([model(b).softmax(1) for b in val_x.split(1024)]).numpy()
        score = float(f1_score(val_y, probs.argmax(1), labels=list(range(5)), average="macro", zero_division=0))
        curve.append(score)

        #和C的早停规则一样：分数严格更高才更新最佳；连续10轮没更新就停
        if score > best:
            best, best_epoch, best_probs, stale = score, epoch, probs, 0
        else:
            stale += 1
        if stale >= 10:
            break
    return best_epoch, best, best_probs, curve


def metrics_row(setting, seed, val_y, probs, best_epoch, curve):
    #根据概率算各项指标，整理成表格的一行
    pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(val_y, pred, labels=list(range(5)), zero_division=0)
    row = {"setting": setting, "seed": seed, "best_epoch": best_epoch,
           "macro_F1": round(float(f.mean()), 4)}
    for j, s in enumerate(STAGES):
        row[f"F1_{s}"] = round(float(f[j]), 4)
    row["N1_precision"] = round(float(p[1]), 4)
    row["N1_recall"] = round(float(r[1]), 4)
    #curve_median：这次训练所有轮次的验证集分数中位数，不受"挑最佳轮次"的运气影响
    row["curve_median"] = round(statistics.median(curve), 4) if curve else ""
    row["n_epochs_run"] = len(curve)
    return row


def save_predictions(path, val_rows, probs):
    #存成全组统一格式的预测表
    pred = probs.argmax(1)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "subject_id", "record_id", "epoch_index", "y_true",
                                          "y_pred", "p_W", "p_N1", "p_N2", "p_N3", "p_REM"])
        w.writeheader()
        for i, r in enumerate(val_rows):
            out = {k: r[k] for k in ("sample_id", "subject_id", "record_id", "epoch_index", "y_true")}
            out["y_pred"] = int(pred[i])
            out.update({f"p_{s}": float(probs[i][j]) for j, s in enumerate(STAGES)})
            w.writerow(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg", type=Path, required=True)
    ap.add_argument("--eog", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    #nargs="+"：这个参数可以接收多个值，比如 --seeds 0 1 2 3 4
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary_all_runs.csv").exists():
        print("WARNING: output folder already has results, choose a new folder")
        sys.exit(1)

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    #---------- 读索引并按sample_id对齐（和train_linear_concat.py一样的核对） ----------
    rows_eeg = read_index(args.eeg)
    rows_eog = read_index(args.eog)
    if {r["sample_id"] for r in rows_eeg} != {r["sample_id"] for r in rows_eog}:
        print("WARNING: sample_id sets differ between EEG and EOG")
        sys.exit(1)
    eog_by_id = {r["sample_id"]: r for r in rows_eog}
    rows_eog = [eog_by_id[r["sample_id"]] for r in rows_eeg]
    for re_, ro in zip(rows_eeg, rows_eog):
        for key in ("y_true", "split", "subject_id"):
            if re_[key] != ro[key]:
                print(f"WARNING: {key} mismatch for {re_['sample_id']}")
                sys.exit(1)
    print("OK: EEG and EOG indexes aligned by sample_id, labels/splits match")

    mask = np.array([r["split"] == "train" for r in rows_eeg])
    y = np.array([int(r["y_true"]) for r in rows_eeg], dtype=np.int64)
    val_y = y[~mask]
    val_rows = [r for r in rows_eeg if r["split"] == "val"]

    #---------- 特征只读一次，三种输入各自标准化 ----------
    print("Loading EEG features...", flush=True)
    raw_eeg = load_pooled(args.eeg, rows_eeg)
    print("Loading EOG features...", flush=True)
    raw_eog = load_pooled(args.eog, rows_eog)
        #先分别标准化EEG和EOG，再把标准化好的两份拼起来
    #这和"先拼接再整体标准化"结果完全一样：标准化是每一列单独算均值和标准差，
    #拼接只是把列并排放，不会改变任何一列的均值和标准差
    #这样写不用额外生成一份3.2GB的拼接中间结果，省内存
    x_eeg = standardize(raw_eeg, mask)
    x_eog = standardize(raw_eog, mask)
    del raw_eeg, raw_eog
    inputs = {
        "EEG": x_eeg,
        "EOG": x_eog,
        "EEGEOG": np.concatenate([x_eeg, x_eog], axis=1),
    }
    print("Features ready.", flush=True)

    results = []
    for seed in args.seeds:
        probs_by_setting = {}
        for setting, x in inputs.items():
            t0 = time.perf_counter()
            best_epoch, best, probs, curve = train_head(x, y, mask, seed)
            probs_by_setting[setting] = probs
            row = metrics_row(setting, seed, val_y, probs, best_epoch, curve)
            results.append(row)
            save_predictions(args.output / f"predictions_val_EEGDM_{setting}_cond14_seed{seed}.csv", val_rows, probs)
            print(json.dumps({**row, "seconds": round(time.perf_counter() - t0, 1)}), flush=True)

        #同一个种子下，EEG和EOG两个分类器的概率取平均（晚融合）
        avg = (probs_by_setting["EEG"] + probs_by_setting["EOG"]) / 2
        row = metrics_row("AVG_EEG_EOG", seed, val_y, avg, "", [])
        results.append(row)
        save_predictions(args.output / f"predictions_val_EEGDM_fusedavg_cond14_seed{seed}.csv", val_rows, avg)
        print(json.dumps(row), flush=True)

    #---------- 保存每次的结果，并按输入汇总：平均值和标准差 ----------
    with (args.output / "summary_all_runs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("\n=== Mean +- std over seeds ===")
    agg_rows = []
    for setting in ["EEG", "EOG", "AVG_EEG_EOG", "EEGEOG"]:
        rs = [r for r in results if r["setting"] == setting]
        agg = {"setting": setting, "n_seeds": len(rs)}
        for key in ["macro_F1", "F1_W", "F1_N1", "F1_N2", "F1_N3", "F1_REM", "N1_precision", "N1_recall"]:
            vals = [r[key] for r in rs]
            agg[f"{key}_mean"] = round(statistics.mean(vals), 4)
            agg[f"{key}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        agg_rows.append(agg)
        print(f"{setting:12s} macro_F1 {agg['macro_F1_mean']:.4f} +- {agg['macro_F1_std']:.4f}   "
              f"N1 {agg['F1_N1_mean']:.4f} +- {agg['F1_N1_std']:.4f}   "
              f"N3 {agg['F1_N3_mean']:.4f} +- {agg['F1_N3_std']:.4f}   "
              f"REM {agg['F1_REM_mean']:.4f} +- {agg['F1_REM_std']:.4f}")

    with (args.output / "summary_mean_std.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)
    print(f"\nSaved: {args.output / 'summary_all_runs.csv'}")
    print(f"Saved: {args.output / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()