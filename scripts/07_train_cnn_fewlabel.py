"""Step B3: 少标注小 CNN(EEG 或 EOG),复用 A 的 SmallCNN / 训练循环 / 预测格式。

与 A 的 05_train_cnn.py 的差别只有三点,其余(结构、lr、batch、早停、选模规则)完全一致:
  1. 只用 manifests/label_subsets_<tag>.csv 中 label<pct>==1 的训练受试者(标签与信号)。
  2. 归一化均值/方差只由"有标签的训练受试者"计算(未标注对象的信号也不碰)。
  3. 设备优先级 cuda > mps > cpu,并写入 config(A 的 100% 基线是 CUDA;跨设备数值不会逐位一致)。

Run folder(团队规则)同 A:runs/<run_id>/ 含 README.md, config.yaml, history.csv,
best.ckpt, metrics.json, predictions_val.csv。metrics 用官方 src/evaluation/evaluate.py 计算。

用法:
  python scripts/07_train_cnn_fewlabel.py --channel eeg --label-pct 25 --sampling-seed 2024 --seed 0
  python scripts/07_train_cnn_fewlabel.py --channel eog --label-pct 25 --sampling-seed 2024 --seed 0
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))
import evaluate as ev  # noqa: E402  官方评估
from common import BASE, RUNS, SPLIT, STAGE_NAMES, N_CLASSES, EPOCH_PTS  # noqa: E402

a = importlib.import_module("05_train_cnn")   # A 的 SmallCNN / load_channel / train / predict
PROB_COLS = a.PROB_COLS
CODE_VERSION = "v01"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, choices=["eeg", "eog"])
    ap.add_argument("--label-pct", type=int, default=25, choices=[25, 50, 100])
    ap.add_argument("--sampling-seed", type=int, required=True, help="必须与名单文件的 sampling_seed 一致")
    ap.add_argument("--subset-tag", default="v1")
    ap.add_argument("--seed", type=int, default=0, help="训练种子")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--subset-file", default=None, help="覆盖默认名单文件(多抽样种子重复实验用)")
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    t0 = time.time()

    subset_path = Path(args.subset_file) if args.subset_file else SPLIT / f"label_subsets_{args.subset_tag}.csv"
    subset = pd.read_csv(subset_path, dtype=str)
    if int(subset["sampling_seed"].iloc[0]) != args.sampling_seed:
        sys.exit(f"--sampling-seed {args.sampling_seed} 与名单文件中的 {subset['sampling_seed'].iloc[0]} 不一致")
    col = f"label{args.label_pct}"
    labeled = subset[subset[col] == "1"]
    train_recs = [r for rs in labeled["record_ids"] for r in rs.split(";")]
    unlabeled_subjects = sorted(subset[subset[col] == "0"]["subject_id"])
    rec_split = a.load_split_records()
    val_recs = [r for r, s in rec_split.items() if s == "val"]
    # 防呆:名单中的记录必须全部属于 A 划分的 train
    assert all(rec_split[r] == "train" for r in train_recs), "名单含非 train 记录"

    ch_idx = 0 if args.channel == "eeg" else 1
    Xtr, ytr, mtr = a.load_channel(ch_idx, train_recs)
    Xva, yva, mva = a.load_channel(ch_idx, val_recs)
    mean, std = float(Xtr.mean()), float(Xtr.std())          # 仅有标签的训练受试者
    Xtr = ((Xtr - mean) / std).astype(np.float32)
    Xva = ((Xva - mean) / std).astype(np.float32)

    run_id = args.run_id or (f"E2_{args.channel.upper()}_label{args.label_pct}"
                             f"_sseed{args.sampling_seed}_seed{args.seed}_v01")
    out_dir = RUNS / run_id
    if out_dir.exists():
        sys.exit(f"run folder already exists: {out_dir} (team rule: one id per run; pick a new id)")
    out_dir.mkdir(parents=True)

    def class_counts(y):
        return {STAGE_NAMES[c]: int((y == c).sum()) for c in range(N_CLASSES)}

    full_ref = f"E1_{args.channel.upper()}_label100_seed0_v01"
    cfg = {
        "run_id": run_id, "experiment": "E2", "owner": "B",
        "model": "SmallCNN (team-doc reference structure, A's 05_train_cnn.SmallCNN)",
        "channel": args.channel, "channel_name": "EEG Fpz-Cz" if ch_idx == 0 else "EOG horizontal",
        "n_classes": N_CLASSES, "input_points": EPOCH_PTS,
        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "early_stopping": {"monitor": "val_macro_f1", "patience": 10},
        "seed": args.seed, "device": device,
        "label_frac": args.label_pct / 100,
        "label_budget": {"unit": "subject", "list_file": str(subset_path.relative_to(BASE)
                                                            if subset_path.is_relative_to(BASE) else subset_path),
                         "column": col, "sampling_seed": args.sampling_seed,
                         "n_labeled_subjects": int(len(labeled)), "n_labeled_records": len(train_recs),
                         "labeled_subjects": labeled["subject_id"].tolist(),
                         "n_unlabeled_train_subjects": len(unlabeled_subjects),
                         "unlabeled_train_subjects_used": False},
        "depends_on_run_ids": [full_ref],
        "code_version": CODE_VERSION, "evaluate_version": ev.EVALUATE_VERSION,
        "normalization": {"method": "z-score", "mean": mean, "std": std,
                          "computed_on": "labeled train subjects only"},
        "n_train_epochs": int(len(Xtr)), "n_val_epochs": int(len(Xva)),
        "train_class_counts": class_counts(ytr), "val_class_counts": class_counts(yva),
        "torch": str(torch.__version__),
        "accelerator": torch.cuda.get_device_name(0) if device == "cuda" else device,
    }
    print(json.dumps(cfg, indent=2, default=str, ensure_ascii=False))
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    (out_dir / "README.md").write_text(
        f"# {run_id}\n\n- 负责人: B (实验② 少标注)\n- 内容: 小 CNN,{args.channel.upper()},{args.label_pct}% 标签受试者\n"
        f"- 状态: 运行中 (started)\n", encoding="utf-8")

    model = a.SmallCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}   train epochs: {len(Xtr)}  val epochs: {len(Xva)}  device: {device}")
    best_epoch = a.train(model, Xtr, ytr, Xva, yva, device, cfg, out_dir)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    model.load_state_dict(torch.load(out_dir / "best.ckpt", weights_only=True)["state_dict"])
    model.eval()
    with torch.no_grad():
        P = torch.cat([torch.softmax(model(torch.from_numpy(Xva[b:b + args.batch_size])
                                           .unsqueeze(1).to(device)), 1).cpu()
                       for b in range(0, len(Xva), args.batch_size)]).numpy()
    pred_df = mva.copy()
    pred_df["subject_id"] = pred_df["record_id"].str[:5]
    pred_df["y_true"] = yva
    pred_df["y_pred"] = P.argmax(1)
    for c, colname in enumerate(PROB_COLS):
        pred_df[colname] = np.round(P[:, c], 6)
    pred_df = pred_df[["sample_id", "subject_id", "record_id", "epoch_index", "y_true", "y_pred", *PROB_COLS]]
    pred_path = out_dir / "predictions_val.csv"
    pred_df.to_csv(pred_path, index=False)

    # 官方评估(读回 CSV,保证与交接给 D 的文件一致;含按受试者 bootstrap)
    m = ev.evaluate(pred_path, n_boot=1000, seed=0)
    m["run_id"], m["n_params"] = run_id, n_params
    m["best_epoch"] = best_epoch
    m["train_minutes"] = round((time.time() - t0) / 60, 2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    (out_dir / "README.md").write_text(
        f"# {run_id}\n\n- 负责人: B (实验② 少标注)\n"
        f"- 内容: 小 CNN,{args.channel.upper()},{args.label_pct}% 标签受试者"
        f"({len(labeled)} 人 / {len(train_recs)} 晚),抽样种子 {args.sampling_seed},训练种子 {args.seed}\n"
        f"- 状态: 成功\n- 主指标 val Macro-F1: **{m['macro_f1']:.4f}** (best epoch {best_epoch})\n"
        f"- 全标签对照: {full_ref}(A)\n- 设备: {device};耗时 {m['train_minutes']} min\n"
        f"- 评估: 官方 src/evaluation/evaluate.py {ev.EVALUATE_VERSION}\n", encoding="utf-8")
    print("\n" + ev.metrics_table(m))
    print(f"\nwrote run folder: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
