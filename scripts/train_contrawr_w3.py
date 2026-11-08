"""train_contrawr_w3.py — 实验A 第三周额外基线统一训练入口(W3 v01)。

ContraWR 监督架构(scripts/contrawr_model.py,上游 CNNEncoder2D_SLEEP 重实现)
在团队协议下训练:54/8/16 划分、label10/25/100 三档按人预算、下游种子 0-2、
每 (谱图分量,频率 bin) 标量 z-score 只拟合本预算有标签者、验证 Macro-F1
早停 patience=10、B 官方 evaluate.py(--reference)出 metrics.json、六件套。

与 W2 CNN 训练入口保持同一随机性与训练循环(复用 scripts/train_cnn_w2.py 的
load_records/macro_f1_np/train/predict),唯一区别:
  - 输入:双通道原始信号 V 单位(NPZ µV × 1e-6,与上游 mne 读数一致);
  - 前向内部先 STFT(n_fft=256, hop=64, center=False)-> log|real/imag|+1e-8;
  - 优化器按上游文献 Adam lr=2e-4(batch=128, 上限 60 epoch 与组内一致)。

用法示例(Windows 本地):
    D:/python/python.exe scripts/train_contrawr_w3.py ^
        --budget label25 --seed 0 ^
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
import yaml

from contrawr_model import (HOP, LOG_EPS, N_CLASSES, N_FFT, SPECTROGRAM_SHAPE,
                            ContraWRSleepNet)
from train_cnn_w2 import (PROB_COLS, SAMPLING_SEED, STAGE_NAMES, file_sha256,
                          gpu_driver_version, labeled_records, load_records,
                          load_subjects, macro_f1_np, predict, train)

CODE_VERSION = "v01"
UPSTREAM = {
    "paper": "Yang, Xiao, Westover, Sun, 'Self-Supervised Electroencephalogram "
             "Representation Learning for Automatic Sleep Staging', JMIR AI "
             "2023;2(1):e46769 (arXiv:2110.15278)",
    "repo": "https://github.com/ycq091044/ContraWR",
    "commit": "217753a5711e1d753d7295b4eb9f35701f2739c8",
    "component": "CNNEncoder2D_SLEEP (supervised branch)",
    "input_note": "upstream Sleep-EDF input = Fpz-Cz + Pz-Oz dual EEG; group dual "
                  "(Fpz-Cz + horizontal EOG) used per week-3 same-input rule, "
                  "see reports/week3/a_baseline_selection.md",
    "license_note": "upstream repo has NO license file; architecture reimplemented "
                    "(not copied verbatim), see reports/week3/a_baseline_selection.md",
}


def fit_spectral_stats(model: ContraWRSleepNet, X: np.ndarray, device: str,
                       batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """按 (分量, 频率 bin) 流式 Welford 拟合 log 谱图统计(时间帧并入池化),
    只用传入的有标签训练信号。返回 (4,129) mean/std。"""
    count = torch.zeros(1, 4, SPECTROGRAM_SHAPE[1], 1, device=device, dtype=torch.float32)
    mean = torch.zeros(1, 4, SPECTROGRAM_SHAPE[1], 1, device=device, dtype=torch.float32)
    m2 = torch.zeros(1, 4, SPECTROGRAM_SHAPE[1], 1, device=device, dtype=torch.float32)
    t_frames = SPECTROGRAM_SHAPE[2]
    model.eval()
    with torch.no_grad():
        for b in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[b:b + batch_size]).to(device)
            spec = model.spectrogram(xb)              # (n,4,129,43),未 z-score
            n = spec.shape[0] * t_frames              # 时间帧并入池化
            delta = spec - mean                       # (1,4,129,1) 广播到 (n,4,129,43)
            count += n
            mean += delta.sum(dim=(0, 3), keepdim=True) / count
            m2 += (delta * (spec - mean)).sum(dim=(0, 3), keepdim=True)
    var = m2 / (count - 1)
    assert not torch.isnan(var).any(), "谱图统计出现 NaN(输入异常?)"
    # 实信号的虚部 DC/Nyquist bin 恒为 0,log 后为常数 -> 真方差为 0;float32 Welford
    # 下 m2 可能漂成微负值,sqrt 会得 NaN,须先钳到 0 再开方;最后钳 std 下限。
    var = var.clamp(min=0)
    std = torch.sqrt(var).clamp(min=1e-6)
    return mean.view(4, SPECTROGRAM_SHAPE[1]), std.view(4, SPECTROGRAM_SHAPE[1])


def write_norm_json(means: np.ndarray, stds: np.ndarray, fit_on: dict,
                    version: str = CODE_VERSION) -> dict:
    """谱图表征的标准化参数文件(协议"每通道标量 z-score"的谱图对应扩展)。"""
    return {
        "method": "per-(spectral-part, frequency-bin) scalar z-score on log "
                  "magnitude STFT parts (spectrogram analog of the per-channel "
                  "scalar z-score protocol rule); fit on labeled subjects only",
        "input_shape": [2, 3000],
        "input_units": "V (converted from uV NPZ by x1e-6; matches upstream mne loading)",
        "sampling_rate_hz": 100,
        "stft": {"n_fft": N_FFT, "hop_length": HOP, "center": False,
                 "onesided": True, "log": f"log(|x|+{LOG_EPS})",
                 "spectrogram_shape": list(SPECTROGRAM_SHAPE),
                 "spectral_order": ["EEG real", "EOG real", "EEG imag", "EOG imag"]},
        "bins": {"means": means.tolist(), "stds": stds.tolist()},
        "fit_on": fit_on,
        "version": version,
    }


def write_readme(out_dir: Path, cfg: dict, status: str, extra: dict | None = None):
    """RUN_TEMPLATE 字段;ContraWR 相关行与 CNN 入口分开写。"""
    extra = extra or {}
    lines = [f"# {cfg['run_id']}", ""]
    run = [
        ("run_id", cfg["run_id"]),
        ("负责人/日期", f"{cfg.get('owner', 'A')} / {cfg.get('started_at', '')}"),
        ("状态", status),
        ("实验问题", cfg["task"]),
        ("Git提交号与未提交修改说明", f"{cfg.get('code_commit', '')};未提交修改见 PR/分支记录"),
        ("数据版本、划分版本、预处理版本",
         "Sleep-EDF SC,A 第一周已核验预处理;划分 manifests/split_subjects.csv;"
         "输入 V 单位,STFT 见 config.yaml preprocessing"),
        ("输入模态、模型、权重来源与hash",
         "dual ContraWR 监督架构(上游 CNNEncoder2D_SLEEP 重实现,"
         f"上游 commit {UPSTREAM['commit']});见 config.yaml 与 best.ckpt"),
        ("标签预算与实际有标签人数", f"{cfg['nominal_budget']}({len(cfg['labeled_subjects'])} 人)"),
        ("未标注信号人数、名单位置、标签隐藏方式", "0(A 监督仅用有标签者信号)"),
        ("名单版本与SHA256", f"{cfg.get('subset_version', '')} {cfg.get('subset_sha256', '')}"),
        ("标准化拟合人员、参数文件与SHA256",
         "labeled subjects only;谱图每 (分量,频率 bin) 标量 z-score;normalization.json"),
        ("训练种子/标签抽样种子", f"downstream={cfg['downstream_seed']} / sampling={cfg['sampling_seed']}"),
        ("适配种子、教师种子", "NA / NA(A 监督,无适配/教师)"),
        ("S0/S1及适配权重选择规则", "NA"),
        ("教师/监督编码器来源", "NA(本 run 自身为第三周额外基线)"),
        ("上下文长度、边界/mask、窗口索引", "context_length=1,无窗口处理"),
        ("完整命令、Python及依赖版本", extra.get("command", "")),
        ("GPU、峰值显存、耗时、缓存大小", extra.get("resource_line", "见 resource.json")),
        ("GPU显存、驱动、PyTorch/CUDA、精度、确定性设置、吞吐、可训练参数及实际更新数",
         extra.get("resource_line2", "见 resource.json")),
        ("最佳权重选择规则", "验证集 Macro-F1(patience=10)"),
        ("第三周下游patience=10;N1 precision/recall/F1及REM在同一最佳权重上报告", "见 metrics.json"),
        ("C适配停止规则", "NA"),
        ("D阈值内部按人校准位置", "NA"),
        ("结果文件位置", "六件套:README.md, config.yaml, history.csv, best.ckpt, "
         "metrics.json, predictions_val.csv(+ normalization.json, resource.json)"),
        ("依赖的其他run_id", "无"),
        ("复用第一周结果时逐项配置匹配的证据", "不适用(新训练)"),
        ("初步观察与限制", extra.get("observations", "")),
        ("失败时最后成功步骤、报错与下一步", extra.get("failure", "无")),
    ]
    for k, v in run:
        lines.append(f"- {k}:{v}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--budget", required=True,
                    choices=["label10", "label25", "label100"],
                    help="按人标签预算(5/14/54 人,名单 seed2024;第三周三档)")
    ap.add_argument("--seed", type=int, default=0, help="下游训练种子 0..2(第三周配对种子)")
    ap.add_argument("--data-dir", required=True, help="A 已核验 NPZ 目录(processed)")
    ap.add_argument("--runs-dir", required=True, help="run 文件夹根目录(本地,不入库)")
    ap.add_argument("--run-id", default=None, help="默认 W3_A_CONTRAWR_<budget>_sseed2024_seed<N>_v01")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4, help="上游文献 Adam lr")
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

    # ---- 随机性:模型构造之前设定(与 W2 CNN 入口一致) ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    t0 = time.time()
    run_id = args.run_id or f"W3_A_CONTRAWR_{args.budget}_sseed2024_seed{args.seed}_{CODE_VERSION}"
    out_dir = runs_dir / run_id
    if out_dir.exists():
        sys.exit(f"run 文件夹已存在: {out_dir}(团队规则:一个 id 一个 run)")

    # ---- 划分与预算(与 W2 CNN 入口逐行一致) ----
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

    Xtr, ytr, mtr = load_records(data_dir, lab_recs, [0, 1])   # dual: [EEG, EOG] µV
    Xva, yva, mva = load_records(data_dir, val_recs, [0, 1])
    Xtr = (Xtr * 1e-6).astype(np.float32)                      # µV -> V,与上游 mne 一致
    Xva = (Xva * 1e-6).astype(np.float32)
    print(f"[{run_id}] labeled records={len(lab_recs)} train epochs={len(Xtr)}  "
          f"val records={len(val_recs)} val epochs={len(Xva)}  dual(V)")

    counts_csv = repo_root / "manifests" / "label_counts_v2.csv"
    if counts_csv.exists():
        counts = pd.read_csv(counts_csv, dtype=str)
        row = counts[counts["budget"] == args.budget].iloc[0]
        assert int(row["labeled_epochs"]) == len(Xtr), \
            f"训练片段数 {len(Xtr)} 与 label_counts_v2.csv({row['labeled_epochs']}) 不一致"
        assert int(row["labeled_records"]) == len(lab_recs)

    out_dir.mkdir(parents=True)
    cfg = {
        "run_id": run_id, "experiment": "W3_E1", "owner": "A",
        "task": "第三周实验A:额外基线 ContraWR 监督比较(W3 v01)",
        "model": "ContraWR supervised (reimplementation of upstream CNNEncoder2D_SLEEP)",
        "representation": "ContraWR STFT CNN (W3 external baseline)",
        "modality": "dual", "in_channels": 2,
        "channel_names": ["EEG Fpz-Cz", "EOG horizontal"],
        "input_shape": [2, 3000], "input_units": "V (uV x1e-6)",
        "spectrogram_shape": list(SPECTROGRAM_SHAPE),
        "n_classes": N_CLASSES,
        "preprocessing": {"stft": {"n_fft": N_FFT, "hop_length": HOP, "center": False,
                                   "onesided": True, "log": f"log(|x|+{LOG_EPS})"},
                          "normalization": "per-(spectral-part, frequency-bin) scalar "
                                           "z-score, fit on labeled subjects only"},
        "upstream": UPSTREAM,
        "nominal_budget": args.budget, "labeled_subjects": lab_subjects,
        "labeled_records": lab_recs,
        "unlabeled_signal_subjects": 0,   # A 监督仅用本预算有标签者
        "subset_version": "v2",
        "subset_sha256": file_sha256(subset_csv),
        "sampling_seed": SAMPLING_SEED, "downstream_seed": args.seed,
        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "early_stopping": {"monitor": "val_macro_f1", "patience": 10},
        "selection_rule": "best checkpoint by validation Macro-F1",
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

    step = "spectral stats"
    try:
        # ---- 谱图标准化统计:只用本预算有标签训练者信号(协议第4条) ----
        model = ContraWRSleepNet().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"params: {n_params:,}")
        means, stds = fit_spectral_stats(model, Xtr, device, args.batch_size)
        model.set_norm(means, stds)
        with open(out_dir / "normalization.json", "w", encoding="utf-8") as f:
            json.dump(write_norm_json(
                means.cpu().numpy(), stds.cpu().numpy(),
                fit_on={"nominal_budget": args.budget, "labeled_subjects": lab_subjects,
                        "labeled_records": lab_recs, "n_epochs": int(len(Xtr)),
                        "note": "validation uses the same stats; no unlabeled signal used"}),
                f, indent=2, ensure_ascii=False)

        step = "training"
        best_epoch = train(model, Xtr, ytr, Xva, yva, device, cfg, out_dir)
        step = "predictions"

        with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        model.load_state_dict(torch.load(out_dir / "best.ckpt", weights_only=True)["state_dict"])
        model.eval()
        with torch.no_grad():
            P = torch.softmax(torch.cat([
                model(torch.from_numpy(Xva[b:b + args.batch_size]).to(device))
                for b in range(0, len(Xva), args.batch_size)]), 1).cpu().numpy()
        P = np.round(P, 6)
        y_pred = P.argmax(1)                 # 四舍五入后 argmax,与官方校验一致
        pred_df = mva.copy()
        pred_df["subject_id"] = pred_df["record_id"].str[:5]
        pred_df["y_true"] = yva.astype(int)
        pred_df["y_pred"] = y_pred
        for c, col in enumerate(PROB_COLS):
            pred_df[col] = P[:, c]
        pred_df[["sample_id", "subject_id", "record_id", "epoch_index", "y_true", "y_pred",
                 *PROB_COLS]].to_csv(out_dir / "predictions_val.csv", index=False)
        step = "metrics"

        eval_cmd = [sys.executable, str(evaluator), str(out_dir / "predictions_val.csv"),
                    "--out", str(out_dir / "metrics.json"),
                    "--reference", str(val_ref)]
        res = subprocess.run(eval_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"B 官方评估失败: {res.stderr.strip()}")
        m = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        step = "resource"

        # ---- 推理吞吐(val,batch 128,eval+no_grad;与 CNN 同一设备同一 batch) ----
        model.eval()
        t_bench = time.time()
        with torch.no_grad():
            for b in range(0, len(Xva), args.batch_size):
                model(torch.from_numpy(Xva[b:b + args.batch_size]).to(device))
        val_secs = time.time() - t_bench
        eps = round(len(Xva) / max(val_secs, 1e-6), 1)
        resource = {
            "run_id": run_id, "phase": "train",
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "driver": cfg["driver"],
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda if device == "cuda" else "NA",
            "precision": "float32",
            "determinism": "torch/np/cuda seeded BEFORE model init; cudnn.deterministic=True; "
                           "benchmark=False",
            "trainable_parameters": n_params,
            "actual_epochs": cfg["epochs_run"],
            "actual_updates": cfg["actual_updates"],
            "elapsed_seconds": round(time.time() - t0, 1),
            "peak_memory_mib": int(torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else 0,
            "labeled_examples": int(len(Xtr)),
            "val_examples": int(len(Xva)),
            "inference_val_epochs_per_second": eps,
        }
        with open(out_dir / "resource.json", "w", encoding="utf-8") as f:
            json.dump(resource, f, indent=2, ensure_ascii=False)

        write_readme(out_dir, cfg, "成功", {
            "command": cfg["command"],
            "resource_line": f"耗时 {resource['elapsed_seconds']} s;峰值显存 "
                             f"{resource['peak_memory_mib']} MiB;驱动 {cfg['driver']}",
            "resource_line2": f"参数 {n_params:,};更新数 {cfg['actual_updates']:,};"
                              f"epoch {cfg['epochs_run']};val 吞吐 {eps} epoch/s",
            "observations": f"val Macro-F1 {m['macro_f1']:.4f}(best epoch {best_epoch},"
                            f"B 官方评估);N1 F1 {m['n1_f1']:.4f}",
        })
        print(f"\nval Macro-F1: {m['macro_f1']:.4f}   N1 F1: {m['n1_f1']:.4f}  "
              f"(N1 P {m['n1_precision']:.4f} / R {m['n1_recall']:.4f})")
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
