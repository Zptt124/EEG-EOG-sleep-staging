"""train_cnn_w3.py — 实验A 第三周 SmallCNN 统一训练入口:训练侧选轮(W3 v01)。

第三周协议(与 B 27_b_w3_linear_probe.py 同款选模):
开发集只用于最终评估一次;轮次在训练侧按受试者 5 折留出选择:
  1. 预算内训练者按人分 K=min(5,人数) 折(fold_seed=0,与下游 seed/模型无关);
  2. 每折标准化只拟合折内训练者,固定训练 MAX_EPOCHS(默认 60)轮,
     逐轮对留出折预测;
  3. 拼接各折留出预测(OOF),取 Macro-F1 最高轮次 E*(并列取最早);
  4. 在全部预算内训练者上重新标准化、同 seed 训练 E* 轮,对 8 人开发集预测一次。
其余配方与 W2 一致:SmallCNN、Adam lr=1e-3、batch=128、cudnn 确定性、
种子在模型构造前设定(每折与最终模型同一 seed,同初始化)。

产物:六件套 + inner_cv.csv(逐轮 OOF 宏 F1)+ oof_train.csv(E* 处训练侧
留出预测,列同 B 的 27 号脚本)+ normalization.json(最终统计)。

用法示例(Windows 本地):
    D:/python/python.exe scripts/train_cnn_w3.py ^
        --modality dual --budget label25 --seed 0 ^
        --data-dir D:/sleepedf/processed --runs-dir C:/Users/34364/Desktop/睡眠/exp1/runs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from cnn_model import N_CLASSES, SmallCNN, build_norm_json
from train_cnn_w2 import (PROB_COLS, SAMPLING_SEED, STAGE_NAMES, file_sha256,
                          gpu_driver_version, labeled_records, load_records,
                          load_subjects, macro_f1_np)
from w3_epoch_selection import FOLD_SEED, oof_curve, pick_epoch, subject_folds

CODE_VERSION = "v01"
MODALITY_CH = {"eeg": [0], "dual": [0, 1]}
MOD_KEY = {"eeg": "EEG", "dual": "EEGEOG"}


def seed_everything(seed: int, device: str):
    """每折/最终训练前调用,与 W2 同一顺序:模型构造之前设定。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fit_channel_stats(X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每通道标量 z-score 统计,只拟合 mask 选中的训练者片段(µV)。"""
    sub = X[mask]
    means = np.asarray([float(sub[:, i].mean()) for i in range(sub.shape[1])], dtype=np.float32)
    stds = np.asarray([float(sub[:, i].std()) for i in range(sub.shape[1])], dtype=np.float32)
    assert (stds > 1e-6).all(), f"通道标准差异常(退化为常量?): {stds}"
    return means, stds


def normalize(X: np.ndarray, means, stds) -> np.ndarray:
    return ((X - means[None, :, None]) / stds[None, :, None]).astype(np.float32)


def predict_probs(model, X, device, batch_size) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for b in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[b:b + batch_size]).to(device)
            probs.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(probs)


def train_fixed(model, Xn, ytr, Xho_n, device, cfg, epochs):
    """固定轮数训练(Xn/Xho_n 已按折内统计标准化);返回每轮留出预测与更新数。

    Xho_n 为 None 时不记录每轮留出预测(最终重训)。洗牌 rng 由 downstream_seed
    决定,与 B 的 torch.randperm(重新播种)同构:同 seed -> 同洗牌顺序。
    """
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss()
    n = len(Xn)
    idx = np.arange(n)
    rng = np.random.default_rng(cfg["downstream_seed"])
    per_epoch, updates = [], 0
    for _ in range(epochs):
        model.train()
        rng.shuffle(idx)
        for b in range(0, n, cfg["batch_size"]):
            bi = idx[b:b + cfg["batch_size"]]
            xb = torch.from_numpy(Xn[bi]).to(device)
            yb = torch.from_numpy(ytr[bi]).long().to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
        if Xho_n is not None:
            per_epoch.append(predict_probs(model, Xho_n, device, cfg["batch_size"]))
    return per_epoch, updates


def write_readme(out_dir: Path, cfg: dict, status: str, extra: dict | None = None):
    extra = extra or {}
    lines = [f"# {cfg['run_id']}", ""]
    run = [
        ("run_id", cfg["run_id"]),
        ("负责人/日期", f"{cfg.get('owner', 'A')} / {cfg.get('started_at', '')}"),
        ("状态", status),
        ("实验问题", cfg["task"]),
        ("Git提交号与未提交修改说明", f"{cfg.get('code_commit', '')};未提交修改见 PR/分支记录"),
        ("数据版本、划分版本、预处理版本",
         "Sleep-EDF SC,A 第一周已核验预处理;划分 manifests/split_subjects.csv"),
        ("输入模态、模型、权重来源与hash", f"{cfg['modality']} SmallCNN;见 config.yaml 与 best.ckpt"),
        ("标签预算与实际有标签人数", f"{cfg['nominal_budget']}({len(cfg['labeled_subjects'])} 人)"),
        ("未标注信号人数、名单位置、标签隐藏方式", "0(A 监督仅用有标签者信号)"),
        ("名单版本与SHA256", f"{cfg.get('subset_version', '')} {cfg.get('subset_sha256', '')}"),
        ("标准化拟合人员、参数文件与SHA256",
         "折内训练者(选轮)与全部预算内训练者(最终);normalization.json"),
        ("训练种子/标签抽样种子", f"downstream={cfg['downstream_seed']} / sampling={cfg['sampling_seed']}"),
        ("适配种子、教师种子", "NA / NA(A 监督,无适配/教师)"),
        ("S0/S1及适配权重选择规则", "NA"),
        ("教师/监督编码器来源", "NA(本 run 自身为监督基线)"),
        ("上下文长度、边界/mask、窗口索引", "context_length=1,无窗口处理"),
        ("完整命令、Python及依赖版本", extra.get("command", "")),
        ("GPU、峰值显存、耗时、缓存大小", extra.get("resource_line", "见 resource.json")),
        ("GPU显存、驱动、PyTorch/CUDA、精度、确定性设置、吞吐、可训练参数及实际更新数",
         extra.get("resource_line2", "见 resource.json")),
        ("选轮规则", cfg["selection"]),
        ("第三周N1 precision/recall/F1及REM在同一最佳权重(E* 最终模型)上报告", "见 metrics.json"),
        ("C适配停止规则", "NA"),
        ("D阈值内部按人校准位置", "NA"),
        ("结果文件位置", "六件套:README.md, config.yaml, history.csv, best.ckpt, "
         "metrics.json, predictions_val.csv(+ normalization.json, resource.json, "
         "inner_cv.csv, oof_train.csv)"),
        ("依赖的其他run_id", "无"),
        ("复用第二周结果时逐项配置匹配的证据", "不适用(新训练;W2 为开发集选轮,口径不同)"),
        ("初步观察与限制", extra.get("observations", "")),
        ("失败时最后成功步骤、报错与下一步", extra.get("failure", "无")),
    ]
    for k, v in run:
        lines.append(f"- {k}:{v}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--modality", required=True, choices=["eeg", "dual"],
                    help="dual=EEG+EOG 双通道;eeg=EEG-only(EOG 对照用)")
    ap.add_argument("--budget", required=True,
                    choices=["label10", "label25", "label100"],
                    help="按人标签预算(5/14/54 人,名单 seed2024)")
    ap.add_argument("--seed", type=int, default=0, help="下游训练种子 0..2(第三周配对种子)")
    ap.add_argument("--data-dir", required=True, help="A 已核验 NPZ 目录(processed)")
    ap.add_argument("--runs-dir", required=True, help="run 文件夹根目录(本地,不入库)")
    ap.add_argument("--run-id", default=None, help="默认 W3_A_<MOD>_<budget>_sseed2024_seed<N>_v01")
    ap.add_argument("--epochs", type=int, default=60, help="折内固定训练轮数(选轮扫描上限)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()

    if not 0 <= args.seed <= 4:
        sys.exit(f"--seed 必须是 0..4(协议正式下游种子;第三周主矩阵用 0..2),got {args.seed}")

    repo_root = Path(args.repo_root)
    data_dir = Path(args.data_dir)
    runs_dir = Path(args.runs_dir)
    subset_csv = repo_root / "manifests" / "label_subsets_v2.csv"
    split_csv = repo_root / "manifests" / "split_subjects.csv"
    val_ref = repo_root / "manifests" / "val_reference_v1.csv"
    evaluator = repo_root / "src" / "evaluation" / "evaluate.py"
    for p in (subset_csv, split_csv, val_ref, evaluator, data_dir):
        if not p.exists():
            sys.exit(f"缺少输入文件/目录: {p}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = args.run_id or f"W3_A_{MOD_KEY[args.modality]}_{args.budget}_sseed2024_seed{args.seed}_{CODE_VERSION}"
    out_dir = runs_dir / run_id
    if out_dir.exists():
        sys.exit(f"run 文件夹已存在: {out_dir}(团队规则:一个 id 一个 run)")

    # ---- 划分与预算(与 W2 入口一致) ----
    split_df = pd.read_csv(split_csv, dtype=str)
    rec_split = {}
    for _, r in split_df.iterrows():
        for rec in r["record_ids"].split(";"):
            rec_split[rec] = r["split"]
    train_recs = sorted(r for r, s in rec_split.items() if s == "train")
    val_recs = sorted(r for r, s in rec_split.items() if s == "val")
    train_subjects = {r[:5] for r in train_recs}

    subset_df = load_subjects(subset_csv)
    lab_recs = labeled_records(subset_df, args.budget)
    lab_subjects = sorted(subset_df[subset_df[args.budget] == 1]["subject_id"])
    assert set(lab_recs) <= set(train_recs), \
        f"{args.budget} 预算记录越界到验证/测试: {set(lab_recs) - set(train_recs)}"
    expected_n = {"label10": 5, "label25": 14, "label100": 54}[args.budget]
    assert len(lab_subjects) == expected_n, f"{args.budget} 人数 {len(lab_subjects)} != {expected_n}"

    ch_idx = MODALITY_CH[args.modality]
    Xtr, ytr, mtr = load_records(data_dir, lab_recs, ch_idx)      # (N,C,3000) µV
    Xva, yva, mva = load_records(data_dir, val_recs, ch_idx)
    tr_subj = mtr["record_id"].str[:5].to_numpy()
    print(f"[{run_id}] labeled records={len(lab_recs)} train epochs={len(Xtr)}  "
          f"val epochs={len(Xva)}  modality={args.modality}")

    counts_csv = repo_root / "manifests" / "label_counts_v2.csv"
    if counts_csv.exists():
        counts = pd.read_csv(counts_csv, dtype=str)
        row = counts[counts["budget"] == args.budget].iloc[0]
        assert int(row["labeled_epochs"]) == len(Xtr), \
            f"训练片段数 {len(Xtr)} 与 label_counts_v2.csv({row['labeled_epochs']}) 不一致"
        assert int(row["labeled_records"]) == len(lab_recs)

    out_dir.mkdir(parents=True)
    k = min(5, len(lab_subjects))
    folds = subject_folds(lab_subjects, k)
    cfg = {
        "run_id": run_id, "experiment": "W3_E1", "owner": "A",
        "task": "第三周实验A:SmallCNN 训练侧选轮(W3 v01)",
        "model": "SmallCNN (team-doc reference structure)",
        "modality": args.modality, "in_channels": len(ch_idx),
        "channel_names": (["EEG Fpz-Cz", "EOG horizontal"] if args.modality == "dual"
                          else ["EEG Fpz-Cz"]),
        "input_shape": [len(ch_idx), 3000], "n_classes": N_CLASSES,
        "nominal_budget": args.budget, "labeled_subjects": lab_subjects,
        "labeled_records": lab_recs,
        "unlabeled_signal_subjects": 0,
        "subset_version": "v2",
        "subset_sha256": file_sha256(subset_csv),
        "sampling_seed": SAMPLING_SEED, "downstream_seed": args.seed,
        "lr": args.lr, "batch_size": args.batch_size,
        "fold_scan_epochs": args.epochs, "n_folds": k, "fold_seed": FOLD_SEED,
        "selection": (f"inner subject-wise CV on labeled subjects, K={k}, fold_seed={FOLD_SEED}, "
                      f"fixed {args.epochs} epochs per fold, per-fold standardization fit on "
                      f"fold-train subjects only, epoch = argmax pooled OOF macro-F1 "
                      f"(ties -> earliest); final retrain on ALL labeled subjects for E* "
                      f"epochs; dev set used for final evaluation only"),
        "folds": folds,
        "normalization": {"method": "per-channel z-score", "fit_on": "fold-train subjects "
                          "in inner CV; all labeled subjects for final"},
        "code_version": CODE_VERSION,
        "command": " ".join(sys.argv),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device, "torch": str(torch.__version__),
        "cuda": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "driver": gpu_driver_version() if device == "cuda" else "NA",
        "n_train_epochs": int(len(Xtr)), "n_val_epochs": int(len(Xva)),
    }
    try:
        cfg["code_commit"] = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        cfg["code_commit"] = "unknown"
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    write_readme(out_dir, cfg, "运行中")

    step = "inner CV"
    t0 = time.time()
    try:
        # ---- 1-3. 训练侧按受试者留出,选轮次 ----
        fold_of = np.empty(len(ytr), dtype=int)
        oof = np.zeros((args.epochs, len(ytr), N_CLASSES), dtype=np.float32)
        fold_updates = 0
        for fk, held in enumerate(folds):
            ho = np.isin(tr_subj, held)
            fold_of[ho] = fk
            ft = ~ho
            means, stds = fit_channel_stats(Xtr, ft)
            seed_everything(args.seed, device)
            model = SmallCNN(in_channels=len(ch_idx)).to(device)
            per_epoch, updates = train_fixed(model, normalize(Xtr[ft], means, stds), ytr[ft],
                                             normalize(Xtr[ho], means, stds), device, cfg,
                                             args.epochs)
            fold_updates += updates
            oof[:, ho] = np.stack(per_epoch)
            print(f"  fold {fk + 1}/{k} held={held} updates={updates}", flush=True)
        curve = oof_curve(ytr, oof, macro_f1_np)
        best = pick_epoch(curve)
        e_star = best["epoch"]
        print(f"  E* = {e_star} (OOF macro-F1 {best['oof_macro_f1']:.4f})", flush=True)
        pd.DataFrame(curve).to_csv(out_dir / "inner_cv.csv", index=False)

        step = "final training"
        # ---- 4. 全部预算内训练者重训 E* 轮 ----
        means, stds = fit_channel_stats(Xtr, np.ones(len(ytr), dtype=bool))
        seed_everything(args.seed, device)
        model = SmallCNN(in_channels=len(ch_idx)).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        _, final_updates = train_fixed(model, normalize(Xtr, means, stds), ytr, None,
                                       device, cfg, e_star)
        torch.save({"state_dict": {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()},
                    "epoch": e_star, "config": cfg}, out_dir / "best.ckpt")
        cfg["selected_epoch"] = e_star
        cfg["oof_macro_f1_at_selected"] = best["oof_macro_f1"]
        cfg["epochs_run"] = e_star
        cfg["fold_updates"] = fold_updates
        cfg["final_updates"] = final_updates

        with open(out_dir / "normalization.json", "w", encoding="utf-8") as f:
            json.dump(build_norm_json(
                means.tolist(), stds.tolist(),
                fit_on={"nominal_budget": args.budget, "labeled_subjects": lab_subjects,
                        "labeled_records": lab_recs, "n_epochs": int(len(Xtr)),
                        "note": "final model normalization fit on ALL labeled subjects; "
                                "inner-CV folds used fold-train-only stats; "
                                "no unlabeled signal used"},
                in_channels=len(ch_idx), version=CODE_VERSION), f, indent=2, ensure_ascii=False)

        step = "predictions"
        P = np.round(predict_probs(model, normalize(Xva, means, stds), device,
                                   args.batch_size), 6)
        y_pred = P.argmax(1)
        pred_df = mva.copy()
        pred_df["subject_id"] = pred_df["record_id"].str[:5]
        pred_df["y_true"] = yva.astype(int)
        pred_df["y_pred"] = y_pred
        for c, col in enumerate(PROB_COLS):
            pred_df[col] = P[:, c]
        pred_df[["sample_id", "subject_id", "record_id", "epoch_index", "y_true", "y_pred",
                 *PROB_COLS]].to_csv(out_dir / "predictions_val.csv", index=False)

        # E* 处训练侧留出预测(列同 B 的 27 号脚本 oof_train.csv)
        oo = mtr.copy()
        oo["subject_id"] = oo["record_id"].str[:5]
        oo["y_true"] = ytr.astype(int)
        oo["fold"] = fold_of
        oo["y_pred"] = oof[e_star - 1].argmax(1)
        for c, col in enumerate(PROB_COLS):
            oo[col] = oof[e_star - 1][:, c]
        oo[["sample_id", "subject_id", "record_id", "epoch_index", "y_true", "fold",
            "y_pred", *PROB_COLS]].to_csv(out_dir / "oof_train.csv", index=False,
                                           float_format="%.8g")

        step = "metrics"
        eval_cmd = [sys.executable, str(evaluator), str(out_dir / "predictions_val.csv"),
                    "--out", str(out_dir / "metrics.json"),
                    "--reference", str(val_ref)]
        res = subprocess.run(eval_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"B 官方评估失败: {res.stderr.strip()}")
        m = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

        step = "resource"
        resource = {
            "run_id": run_id, "phase": "train",
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "driver": cfg["driver"],
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda if device == "cuda" else "NA",
            "precision": "float32",
            "determinism": "seeded BEFORE each model init (folds and final); "
                           "cudnn.deterministic=True; benchmark=False",
            "trainable_parameters": n_params,
            "selected_epoch": e_star,
            "fold_updates": fold_updates,
            "final_updates": final_updates,
            "elapsed_seconds": round(time.time() - t0, 1),
            "peak_memory_mib": int(torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else 0,
            "labeled_examples": int(len(Xtr)),
            "val_examples": int(len(Xva)),
        }
        with open(out_dir / "resource.json", "w", encoding="utf-8") as f:
            json.dump(resource, f, indent=2, ensure_ascii=False)

        with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        write_readme(out_dir, cfg, "成功", {
            "command": cfg["command"],
            "resource_line": f"耗时 {resource['elapsed_seconds']} s;峰值显存 "
                             f"{resource['peak_memory_mib']} MiB;驱动 {cfg['driver']}",
            "resource_line2": f"参数 {n_params:,};折内更新 {fold_updates:,};最终更新 "
                              f"{final_updates:,};E* {e_star}",
            "observations": f"E*={e_star}(OOF Macro-F1 {best['oof_macro_f1']:.4f});"
                            f"val Macro-F1 {m['macro_f1']:.4f}(B 官方评估);N1 F1 {m['n1_f1']:.4f}",
        })
        print(f"\nval Macro-F1: {m['macro_f1']:.4f}   N1 F1: {m['n1_f1']:.4f}  "
              f"(N1 P {m['n1_precision']:.4f} / R {m['n1_recall']:.4f})  E*={e_star}")
        print(f"wrote run folder: {out_dir}")
    except Exception as e:
        write_readme(out_dir, cfg, "失败", {
            "command": cfg.get("command", ""),
            "failure": f"最后成功步骤: {step};报错: {e};traceback 见运行日志\n"
                       f"```\n{traceback.format_exc()}\n```",
        })
        sys.exit(f"run 失败于步骤 {step}: {e}")


if __name__ == "__main__":
    main()
