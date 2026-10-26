#multiseed_budget.py —— 少标注（如25%标签）下的多种子线性头实验
#对每一份标签名单：只用名单里有标签的训练受试者来训练和标准化，其余训练受试者完全不用
#每份名单 × 每个训练种子，训练 EEG、EOG、EEG+EOG拼接 三个线性头，再做 EEG/EOG 概率平均
#训练细节完全复用 multiseed_heads.py 里的函数，保证和100%实验一致
#用法：python multiseed_budget.py --eeg EEG特征 --eog EOG特征 --output 输出文件夹 --subsets 名单1.csv 名单2.csv ... --seeds 0 1 2 3 4

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

#从同一个文件夹里的multiseed_heads.py借用写好的函数
#Python运行脚本时，会自动把脚本所在的文件夹加进"找模块的地方"，所以能直接import
from multiseed_heads import read_index, load_pooled, standardize, train_head, metrics_row, save_predictions


def read_subset(path, budget_col):
    #读B的标签名单。每行一个训练受试者，label25列为1表示在25%名单里
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    seeds = {r["sampling_seed"] for r in rows}
    if len(seeds) != 1:
        print(f"WARNING: {path} contains more than one sampling_seed: {seeds}")
        sys.exit(1)
    selected = {r["subject_id"] for r in rows if r[budget_col] == "1"}
    all_train = {r["subject_id"] for r in rows if r["label100"] == "1"}
    return seeds.pop(), selected, all_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg", type=Path, required=True)
    ap.add_argument("--eog", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--subsets", type=Path, nargs="+", required=True)
    ap.add_argument("--budget-col", default="label25")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary_all_runs.csv").exists():
        print("WARNING: output folder already has results, choose a new folder")
        sys.exit(1)

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    #---------- 读索引并按sample_id对齐（和multiseed_heads.py一样） ----------
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

    y = np.array([int(r["y_true"]) for r in rows_eeg], dtype=np.int64)
    subject = np.array([r["subject_id"] for r in rows_eeg])
    is_train = np.array([r["split"] == "train" for r in rows_eeg])
    is_val = np.array([r["split"] == "val" for r in rows_eeg])
    val_rows = [r for r in rows_eeg if r["split"] == "val"]
    val_y = y[is_val]
    train_subjects = set(subject[is_train])

    #---------- 特征只读一次（还没标准化，每份名单再各自标准化） ----------
    print("Loading EEG features...", flush=True)
    raw_eeg = load_pooled(args.eeg, rows_eeg)
    print("Loading EOG features...", flush=True)
    raw_eog = load_pooled(args.eog, rows_eog)
    print("Features ready.", flush=True)

    results = []
    for subset_path in args.subsets:
        sseed, selected, all_train = read_subset(subset_path, args.budget_col)

        #核对名单：名单里的"全部训练受试者"必须和特征索引里的训练集一致；挑中的人必须都是训练集的人
        if all_train != train_subjects:
            print(f"WARNING: training subjects in {subset_path.name} differ from feature index")
            sys.exit(1)
        if not selected <= train_subjects:
            print(f"WARNING: {subset_path.name} selects subjects outside the training split")
            sys.exit(1)

        #labeled：属于训练集、并且在名单里的段；keep：要用到的段 = 有标签的训练段 + 全部验证段
        #没被选中的训练受试者，特征和标签都不进入训练和标准化
        labeled = is_train & np.isin(subject, list(selected))
        keep = labeled | is_val
        mask_sub = labeled[keep]  #在keep这批段里，哪些是训练段（其余是验证段）
        y_sub = y[keep]
        print(f"\nSubset sseed={sseed}: {len(selected)} labeled subjects, {int(labeled.sum())} labeled segments", flush=True)

        #只用有标签的训练段算标准化参数（standardize内部用mask_sub）
        x_eeg = standardize(raw_eeg[keep], mask_sub)
        x_eog = standardize(raw_eog[keep], mask_sub)
        inputs = {"EEG": x_eeg, "EOG": x_eog, "EEGEOG": np.concatenate([x_eeg, x_eog], axis=1)}

        for seed in args.seeds:
            probs_by_setting = {}
            for setting, x in inputs.items():
                #train_head里：x[mask_sub]是训练段，x[~mask_sub]是验证段（顺序和val_rows一致）
                best_epoch, best, probs, curve = train_head(x, y_sub, mask_sub, seed)
                probs_by_setting[setting] = probs
                row = {"sampling_seed": sseed, **metrics_row(setting, seed, val_y, probs, best_epoch, curve)}
                results.append(row)
                save_predictions(args.output / f"predictions_val_EEGDM_{setting}_cond14_{args.budget_col}_sseed{sseed}_seed{seed}.csv", val_rows, probs)
                print(json.dumps(row), flush=True)

            avg = (probs_by_setting["EEG"] + probs_by_setting["EOG"]) / 2
            row = {"sampling_seed": sseed, **metrics_row("AVG_EEG_EOG", seed, val_y, avg, "", [])}
            results.append(row)
            save_predictions(args.output / f"predictions_val_EEGDM_fusedavg_cond14_{args.budget_col}_sseed{sseed}_seed{seed}.csv", val_rows, avg)
            print(json.dumps(row), flush=True)

        del x_eeg, x_eog, inputs  #这份名单用完，释放内存

    #---------- 保存每次的结果 ----------
    with (args.output / "summary_all_runs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    #---------- 汇总：每份名单各自的均值±标准差，以及全部名单合起来的 ----------
    keys = ["macro_F1", "F1_W", "F1_N1", "F1_N2", "F1_N3", "F1_REM", "N1_precision", "N1_recall"]
    settings = ["EEG", "EOG", "AVG_EEG_EOG", "EEGEOG"]
    groups = [(s, sseed) for sseed in sorted({r["sampling_seed"] for r in results}) for s in settings]
    groups += [(s, "all") for s in settings]

    agg_rows = []
    print("\n=== Mean +- std over training seeds (per subset) and over all runs ===")
    for setting, sseed in groups:
        rs = [r for r in results if r["setting"] == setting and (sseed == "all" or r["sampling_seed"] == sseed)]
        agg = {"setting": setting, "sampling_seed": sseed, "n_runs": len(rs)}
        for key in keys:
            vals = [r[key] for r in rs]
            agg[f"{key}_mean"] = round(statistics.mean(vals), 4)
            agg[f"{key}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        agg_rows.append(agg)
        print(f"sseed={str(sseed):5s} {setting:12s} macro_F1 {agg['macro_F1_mean']:.4f} +- {agg['macro_F1_std']:.4f}   "
              f"N1 {agg['F1_N1_mean']:.4f}   N3 {agg['F1_N3_mean']:.4f}   REM {agg['F1_REM_mean']:.4f}")

    with (args.output / "summary_mean_std.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)
    print(f"\nSaved: {args.output / 'summary_all_runs.csv'}")
    print(f"Saved: {args.output / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()