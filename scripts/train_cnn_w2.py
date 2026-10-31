"""train_cnn_w2.py — 实验① A 统一训练入口(W2 v1.1)。

支持单/双通道与按人标签预算,沿用第一周 SmallCNN 与训练规则,生成六件套,
并用 B 官方 evaluate.py(带 --reference)产出 metrics.json。

用法示例(Windows 本地):
    D:/python/python.exe scripts/train_cnn_w2.py ^
        --modality dual --budget label25 --seed 0 ^
        --data-dir D:/sleepedf/processed --runs-dir C:/Users/34364/Desktop/睡眠/exp1/runs

固定设置(与第一周一致,保证 100% 预算可登记复用):
    SmallCNN(团队文档结构;dual 仅首层输入通道 2)、Adam lr=1e-3、batch=128、
    最多 60 epoch、验证 Macro-F1 早停 patience=10、cudnn 确定性模式。
随机性:torch/np/cuda 种子在模型构造**之前**设定(与第一周 05_train_cnn.py 一致),
同一 --seed 得到同一初始权重与洗牌顺序。
标准化:每通道标量 z-score,只用本预算有标签者(及其夜晚全部片段)拟合;
验证集用同一统计。监督训练不读其他训练者标签,不使用无标签信号。

D 内折重训:--subjects 覆盖预算名单(仍须全部在训练集),默认 run_id 带名单哈希;
fold run 的早停仍用外部 8 人验证集(接口限制,见 docs/week2/cnn_interface.md)。
"""
from __future__ import annotations

import argparse
import hashlib
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

from cnn_model import CHANNEL_NAMES, N_CLASSES, SmallCNN, build_norm_json

PROB_COLS = ["p_W", "p_N1", "p_N2", "p_N3", "p_REM"]
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
CODE_VERSION = "v01"
SAMPLING_SEED = 2024                      # 名单种子固定 2024(协议v2)

MODALITY_CH = {"eeg": [0], "eog": [1], "dual": [0, 1]}


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_subjects(csv_path: Path) -> pd.DataFrame:
    """label_subsets_v2.csv:subject_id/record_ids 与各预算标记。"""
    df = pd.read_csv(csv_path, dtype=str)
    for col in ("label10", "label25", "label50", "label100"):
        df[col] = df[col].astype(int)
    return df


def labeled_records(subset_df: pd.DataFrame, budget: str) -> list[str]:
    """预算选中的训练记录列表(按 record_id 排序,与第一周加载顺序一致)。"""
    sel = subset_df[subset_df[budget] == 1]
    recs = []
    for _, r in sel.iterrows():
        recs.extend(r["record_ids"].split(";"))
    return sorted(recs)


def load_records(data_dir: Path, recs: list[str], ch_idx: list[int]):
    """加载若干记录的 (signal, labels, meta)。signal 形状 (N, C, 3000),单位 µV。

    双通道的两列取自同一 npz 行,record_id/epoch_index 天然同步,不存在独立
    打乱(第二周工作文档 三.实验① 第3步)。
    """
    sigs, labs, metas = [], [], []
    for rec in sorted(recs):
        z = np.load(data_dir / f"{rec}.npz")
        sigs.append(z["signal"][:, ch_idx])
        labs.append(z["labels"])
        ei = z["epoch_index"]
        metas.append(pd.DataFrame({
            "record_id": rec,
            "epoch_index": ei,
            "sample_id": [f"{rec}_{int(i):04d}" for i in ei],
        }))
    return (np.concatenate(sigs).astype(np.float32), np.concatenate(labs),
            pd.concat(metas, ignore_index=True))


def macro_f1_np(y_true, y_pred) -> float:
    """与 B 官方 evaluate.py 同定义:numpy 混淆矩阵,五类 F1 平均,零分母记 0。"""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    pred_n = cm.sum(0)
    prec = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


def train(model, Xtr, ytr, Xva, yva, device, cfg, out_dir, early_stop=10):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    idx = np.arange(n)
    rng = np.random.default_rng(cfg["downstream_seed"])   # 洗牌顺序由种子决定
    history = []
    best = {"macro_f1": -1.0, "state": None, "epoch": -1}
    bad = 0
    for ep in range(cfg["epochs"]):
        ep_t0 = time.time()
        model.train()
        rng.shuffle(idx)
        tot_loss, nb = 0.0, 0
        for b in range(0, n, cfg["batch_size"]):
            bi = idx[b:b + cfg["batch_size"]]
            xb = torch.from_numpy(Xtr[bi]).to(device)
            yb = torch.from_numpy(ytr[bi]).long().to(device)
            out = model(xb)
            loss = lossf(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(bi)
            nb += len(bi)
        train_loss = tot_loss / nb
        model.eval()
        val_loss, yp, yt = predict(model, Xva, yva, device, cfg["batch_size"], with_loss=True)
        mf1 = macro_f1_np(yt, yp)
        ep_sec = round(time.time() - ep_t0, 2)
        history.append({"epoch": ep + 1, "train_loss": round(train_loss, 6),
                        "val_loss": round(val_loss, 6),
                        "val_macro_f1": round(mf1, 6), "epoch_seconds": ep_sec})
        print(f"epoch {ep + 1:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"val_macro_f1 {mf1:.4f}  ({ep_sec}s)", flush=True)
        if ep == 0:
            eta_min = round(ep_sec * cfg["epochs"] / 60, 1)
            print(f"ETA (max {cfg['epochs']} epochs): ~{eta_min} min (early stop may cut short)")
        if mf1 > best["macro_f1"] + 1e-8:
            best = {"macro_f1": mf1, "epoch": ep + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0                        # 与第一周一致:按验证 Macro-F1 选权重
        else:
            bad += 1
            if bad >= early_stop:
                print(f"early stop at epoch {ep + 1} (no val Macro-F1 gain for "
                      f"{early_stop} consecutive epochs)")
                break
    model.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "epoch": best["epoch"],
                "val_macro_f1": best["macro_f1"], "config": cfg},
               out_dir / "best.ckpt")
    cfg["epochs_run"] = len(history)
    cfg["actual_updates"] = len(history) * max(1, int(np.ceil(n / cfg["batch_size"])))
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    return best["epoch"]


def predict(model, X, y, device, batch_size, with_loss=False):
    lossf = nn.CrossEntropyLoss()
    probs, ys, losses, nb = [], [], 0.0, 0
    with torch.no_grad():
        for b in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[b:b + batch_size]).to(device)
            yb = torch.from_numpy(y[b:b + batch_size]).long().to(device)
            out = model(xb)
            probs.append(torch.softmax(out, 1).cpu().numpy())
            ys.append(yb.cpu().numpy())
            if with_loss:
                losses += lossf(out, yb).item() * len(xb)
                nb += len(xb)
    P = np.concatenate(probs)
    y_true = np.concatenate(ys)
    y_pred = P.argmax(1)
    if with_loss:
        return losses / nb, y_pred, y_true
    return y_pred, y_true


def gpu_driver_version() -> str:
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=15)
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def write_readme(out_dir: Path, cfg: dict, status: str, extra: dict | None = None):
    """按 RUN_TEMPLATE 字段生成 README;数值细节指向 config.yaml 六件套文件。"""
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
        ("标准化拟合人员、参数文件与SHA256", f"labeled subjects only;normalization.json"),
        ("训练种子/标签抽样种子", f"downstream={cfg['downstream_seed']} / sampling={cfg['sampling_seed']}"),
        ("适配种子、教师种子", "NA / NA(A 监督,无适配/教师)"),
        ("S0/S1及适配权重选择规则", "NA"),
        ("教师/监督编码器来源", "NA(本 run 自身为监督基线)"),
        ("上下文长度、边界/mask、窗口索引", "context_length=1,无窗口处理"),
        ("完整命令、Python及依赖版本", extra.get("command", "")),
        ("GPU、峰值显存、耗时、缓存大小", extra.get("resource_line", "见 resource.json")),
        ("GPU显存、驱动、PyTorch/CUDA、精度、确定性设置、吞吐、可训练参数及实际更新数",
         extra.get("resource_line2", "见 resource.json")),
        ("最佳权重选择规则", "验证集 Macro-F1(patience=10)"),
        ("第二周下游patience=10;N1 precision/recall/F1及REM在同一最佳权重上报告", "见 metrics.json"),
        ("C适配停止规则", "NA"),
        ("D阈值内部按人校准位置", "NA"),
        ("结果文件位置", "六件套:README.md, config.yaml, history.csv, best.ckpt, "
         "metrics.json, predictions_val.csv(+ normalization.json, resource.json)"),
        ("依赖的其他run_id", "无"),
        ("复用第一周结果时逐项配置匹配的证据", "不适用(新训练);第一周复用见 reports/week2/a_reuse_check_w1.md"),
        ("初步观察与限制", extra.get("observations", "")),
        ("失败时最后成功步骤、报错与下一步", extra.get("failure", "无")),
    ]
    for k, v in run:
        lines.append(f"- {k}:{v}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--modality", required=True, choices=["eeg", "eog", "dual"],
                    help="输入模态;dual=EEG+EOG 双通道(首层输入通道 2)")
    ap.add_argument("--budget", required=True,
                    choices=["label10", "label25", "label50", "label100"],
                    help="按人标签预算(5/14/27/54 人,名单 seed2024);--subjects 覆盖名单时仅作标注")
    ap.add_argument("--seed", type=int, default=0, help="下游训练种子 0..4")
    ap.add_argument("--subjects", default=None,
                    help="逗号分隔 subject_id,覆盖 --budget 名单(D 内折重训等自定义名单;"
                         "仍须全部在训练集,不读其余训练者标签)")
    ap.add_argument("--owner", default="A", help="run 负责人(内折重训可写 D)")
    ap.add_argument("--data-dir", required=True, help="A 已核验 NPZ 目录(processed)")
    ap.add_argument("--runs-dir", required=True, help="run 文件夹根目录(本地,不入库)")
    ap.add_argument("--run-id", default=None, help="默认 W2_A_<MOD>_<budget>_sseed2024_seed<N>_v01")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()

    if not 0 <= args.seed <= 4:
        sys.exit(f"--seed 必须是 0..4(协议v2 正式下游种子),got {args.seed}")

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

    # ---- 随机性:模型构造之前设定(与第一周一致;同一 seed -> 同一初始权重) ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    t0 = time.time()

    mod_key = {"eeg": "EEG", "eog": "EOG", "dual": "EEGEOG"}[args.modality]
    if args.subjects:
        subj_hash = hashlib.sha256(",".join(sorted(s.strip() for s in
                                    args.subjects.split(",") if s.strip())).encode()).hexdigest()[:8]
        budget_tag = f"custom{subj_hash}"
    else:
        budget_tag = args.budget
    run_id = args.run_id or f"W2_A_{mod_key}_{budget_tag}_sseed2024_seed{args.seed}_{CODE_VERSION}"
    out_dir = runs_dir / run_id
    if out_dir.exists():
        sys.exit(f"run 文件夹已存在: {out_dir}(团队规则:一个 id 一个 run)")

    # ---- 划分与预算 ----
    split_df = pd.read_csv(split_csv, dtype=str)
    rec_split = {}
    for _, r in split_df.iterrows():
        for rec in r["record_ids"].split(";"):
            rec_split[rec] = r["split"]
    train_recs = sorted(r for r, s in rec_split.items() if s == "train")
    val_recs = sorted(r for r, s in rec_split.items() if s == "val")
    train_subjects = {r[:5] for r in train_recs}

    subset_df = load_subjects(subset_csv)
    if args.subjects:
        lab_subjects = sorted(s.strip() for s in args.subjects.split(",") if s.strip())
        missing = set(lab_subjects) - train_subjects
        assert not missing, f"--subjects 含非训练人员或拼写错误: {sorted(missing)}"
        lab_recs = sorted(r for r in train_recs if r[:5] in set(lab_subjects))
        nominal_budget = f"custom({len(lab_subjects)} subjects)"
    else:
        lab_recs = labeled_records(subset_df, args.budget)
        lab_subjects = sorted(subset_df[subset_df[args.budget] == 1]["subject_id"])
        nominal_budget = args.budget
    # 隔离检查:预算人员必须全部在训练集(协议v2;8人验证/16人测试不进训练)
    assert set(lab_recs) <= set(train_recs), \
        f"{nominal_budget} 预算记录越界到验证/测试: {set(lab_recs) - set(train_recs)}"
    if not args.subjects:
        # 预算只取前 5/14/27/54 人(嵌套),不重排不挑人
        expected_n = {"label10": 5, "label25": 14, "label50": 27, "label100": 54}[args.budget]
        assert len(lab_subjects) == expected_n, f"{args.budget} 人数 {len(lab_subjects)} != {expected_n}"

    ch_idx = MODALITY_CH[args.modality]
    in_channels = len(ch_idx)

    Xtr, ytr, mtr = load_records(data_dir, lab_recs, ch_idx)
    Xva, yva, mva = load_records(data_dir, val_recs, ch_idx)
    print(f"[{run_id}] labeled records={len(lab_recs)} train epochs={len(Xtr)}  "
          f"val records={len(val_recs)} val epochs={len(Xva)}  "
          f"channels={CHANNEL_NAMES[ch_idx[0]] if in_channels == 1 else CHANNEL_NAMES}")

    # 与 B 发布统计核对(仅标准预算;自定义名单跳过)
    counts_csv = repo_root / "manifests" / "label_counts_v2.csv"
    if counts_csv.exists() and not args.subjects:
        counts = pd.read_csv(counts_csv, dtype=str)
        row = counts[counts["budget"] == args.budget].iloc[0]
        assert int(row["labeled_epochs"]) == len(Xtr), \
            f"训练片段数 {len(Xtr)} 与 label_counts_v2.csv({row['labeled_epochs']}) 不一致"
        assert int(row["labeled_records"]) == len(lab_recs)

    # ---- 标准化:每通道标量 z-score,只用本预算有标签者拟合(协议v2 第4条) ----
    means = [float(Xtr[:, i].mean()) for i in range(in_channels)]
    stds = [float(Xtr[:, i].std()) for i in range(in_channels)]
    assert all(s > 1e-6 for s in stds), f"通道标准差异常(退化为常量?): {stds}"
    Xtr = ((Xtr - np.array(means, dtype=np.float32)[None, :, None])
           / np.array(stds, dtype=np.float32)[None, :, None]).astype(np.float32)
    Xva = ((Xva - np.array(means, dtype=np.float32)[None, :, None])
           / np.array(stds, dtype=np.float32)[None, :, None]).astype(np.float32)

    out_dir.mkdir(parents=True)
    norm_json = build_norm_json(
        means, stds,
        fit_on={"nominal_budget": nominal_budget, "labeled_subjects": lab_subjects,
                "labeled_records": lab_recs, "n_epochs": int(len(Xtr)),
                "note": "validation uses the same stats; no unlabeled signal used"},
        in_channels=in_channels, version=CODE_VERSION)
    with open(out_dir / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(norm_json, f, indent=2, ensure_ascii=False)

    cfg = {
        "run_id": run_id, "experiment": "W2_E1", "owner": args.owner,
        "task": "公平监督CNN(W2 v1.1 实验①)",
        "model": "SmallCNN (team-doc reference structure)",
        "modality": args.modality, "in_channels": in_channels,
        "channel_names": CHANNEL_NAMES[:in_channels] if in_channels == 2 else [CHANNEL_NAMES[ch_idx[0]]],
        "input_shape": [in_channels, 3000], "n_classes": N_CLASSES,
        "nominal_budget": nominal_budget, "labeled_subjects": lab_subjects,
        "labeled_records": lab_recs,
        "unlabeled_signal_subjects": 0,   # A 监督仅用本预算有标签者
        "subset_version": "v2" if not args.subjects else "custom",
        "subset_sha256": file_sha256(subset_csv),
        "sampling_seed": SAMPLING_SEED, "downstream_seed": args.seed,
        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "early_stopping": {"monitor": "val_macro_f1", "patience": 10},
        "selection_rule": "best checkpoint by validation Macro-F1",
        "normalization": {"method": "per-channel z-score", "fit_on": "labeled subjects only",
                          "path": "normalization.json"},
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
    # 六件套先写 config/README,失败 run 也留记录
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    write_readme(out_dir, cfg, "运行中")

    step = "training"
    try:
        model = SmallCNN(in_channels=in_channels).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"params: {n_params:,}")
        best_epoch = train(model, Xtr, ytr, Xva, yva, device, cfg, out_dir)
        step = "predictions"

        with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        # ---- 验证预测(11 列,与 B 官方 evaluate.py 一致) ----
        model.load_state_dict(torch.load(out_dir / "best.ckpt", weights_only=True)["state_dict"])
        model.eval()
        with torch.no_grad():
            P = torch.softmax(torch.cat([
                model(torch.from_numpy(Xva[b:b + args.batch_size]).to(device))
                for b in range(0, len(Xva), args.batch_size)]), 1).cpu().numpy()
        P = np.round(P, 6)
        y_pred = P.argmax(1)                 # 基于四舍五入后概率取 argmax,保证与官方校验一致
        pred_df = mva.copy()
        pred_df["subject_id"] = pred_df["record_id"].str[:5]
        pred_df["y_true"] = yva.astype(int)
        pred_df["y_pred"] = y_pred
        for c, col in enumerate(PROB_COLS):
            pred_df[col] = P[:, c]
        pred_df[["sample_id", "subject_id", "record_id", "epoch_index", "y_true", "y_pred",
                 *PROB_COLS]].to_csv(out_dir / "predictions_val.csv", index=False)
        step = "metrics"

        # ---- B 官方评估(带 val_reference 核对;统一口径,metrics.json 为官方输出) ----
        eval_cmd = [sys.executable, str(evaluator), str(out_dir / "predictions_val.csv"),
                    "--out", str(out_dir / "metrics.json"),
                    "--reference", str(val_ref)]
        res = subprocess.run(eval_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"B 官方评估失败: {res.stderr.strip()}")
        m = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        step = "resource"

        # ---- 资源记录(协议v2:耗时/显存/参数/更新数;第4步:环境/单轮耗时/ETA) ----
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
        }
        with open(out_dir / "resource.json", "w", encoding="utf-8") as f:
            json.dump(resource, f, indent=2, ensure_ascii=False)

        write_readme(out_dir, cfg, "成功", {
            "command": cfg["command"],
            "resource_line": f"耗时 {resource['elapsed_seconds']} s;峰值显存 "
                             f"{resource['peak_memory_mib']} MiB;驱动 {cfg['driver']}",
            "resource_line2": f"参数 {n_params:,};更新数 {cfg['actual_updates']:,};"
                              f"epoch {cfg['epochs_run']}",
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
