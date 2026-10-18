"""Step 6/7: train the SmallCNN on EEG or EOG, identical settings per channel.

Reference structure from the team doc (input 30 s x 100 Hz = 3000 pts):
    block(1,32,50,6) -> block(32,64,8,1) -> block(64,128,8,1) -> GAP -> Linear(128,5)

Normalization: per-channel z-score with mean/std computed on TRAIN subjects only.
Best checkpoint: highest validation Macro-F1.
Run folder (team rules): runs/<run_id>/ with README.md, config.yaml,
history.csv, best.ckpt, metrics.json, predictions_val.csv

Usage:
  python 05_train_cnn.py --channel eeg   --run-id E1_EEG_label100_seed0_v01
  python 05_train_cnn.py --channel eog   --run-id E1_EOG_label100_seed0_v01
  python 05_train_cnn.py --channel eeg   --overfit-check      # pipeline sanity (step 6)
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from common import PROCESSED, RUNS, SPLIT, STAGE_NAMES, N_CLASSES, EPOCH_PTS, sample_id
from metrics import compute_metrics, metrics_table, save_metrics

CODE_VERSION = "v01"
PROB_COLS = ["p_W", "p_N1", "p_N2", "p_N3", "p_REM"]


class SmallCNN(nn.Module):
    """Small 1-D CNN from the team doc, verbatim structure."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        def block(cin, cout, k, s):
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, stride=s, padding=k // 2),
                nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(4))

        self.features = nn.Sequential(
            block(1, 32, 50, 6),     # 大卷积核，抓低频慢波
            block(32, 64, 8, 1),
            block(64, 128, 8, 1),
            nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):             # x: (batch, 1, 3000)
        return self.classifier(self.features(x).flatten(1))


def load_split_records():
    """record_id -> split, from split_subjects.csv."""
    split_df = pd.read_csv(SPLIT / "split_subjects.csv", dtype=str)
    out = {}
    for _, r in split_df.iterrows():
        for rec in r["record_ids"].split(";"):
            out[rec] = r["split"]
    return out


def load_channel(channel_idx: int, recs: list[str]):
    """Load (signal, labels, meta) for a set of records, one channel only.

    Returns arrays in uV (unnormalized) plus per-epoch sample metadata.
    """
    sigs, labs, metas = [], [], []
    for rec in sorted(recs):
        z = np.load(PROCESSED / f"{rec}.npz")
        sigs.append(z["signal"][:, channel_idx])
        labs.append(z["labels"])
        ei = z["epoch_index"]
        metas.append(pd.DataFrame({
            "record_id": rec,
            "epoch_index": ei,
            "sample_id": [sample_id(rec, int(i)) for i in ei],
        }))
    return np.concatenate(sigs), np.concatenate(labs), pd.concat(metas, ignore_index=True)


def train(model, Xtr, ytr, Xva, yva, device, cfg, out_dir, early_stop=10):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    idx = np.arange(n)
    history = []
    best = {"macro_f1": -1.0, "state": None, "epoch": -1}
    bad = 0
    rng = np.random.default_rng(cfg["seed"])
    for ep in range(cfg["epochs"]):
        model.train()
        rng.shuffle(idx)
        tot_loss, nb = 0.0, 0
        for b in range(0, n, cfg["batch_size"]):
            bi = idx[b:b + cfg["batch_size"]]
            xb = torch.from_numpy(Xtr[bi]).unsqueeze(1).to(device)
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
        m = compute_metrics(yt, yp)
        history.append({"epoch": ep + 1, "train_loss": round(train_loss, 6),
                        "val_loss": round(val_loss, 6),
                        "val_acc": round(m["accuracy"], 6),
                        "val_macro_f1": round(m["macro_f1"], 6)})
        print(f"epoch {ep + 1:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"val_acc {m['accuracy']:.4f}  val_macro_f1 {m['macro_f1']:.4f}", flush=True)
        if m["macro_f1"] > best["macro_f1"] + 1e-8:
            best = {"macro_f1": m["macro_f1"], "epoch": ep + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0                        # reset patience on improvement
        else:
            bad += 1
            if bad >= early_stop:
                print(f"early stop at epoch {ep + 1} (no val Macro-F1 gain for {early_stop} consecutive epochs)")
                break
    model.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "epoch": best["epoch"],
                "val_macro_f1": best["macro_f1"], "config": cfg}, out_dir / "best.ckpt")
    cfg["epochs_run"] = len(history)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    return best["epoch"]


def predict(model, X, y, device, batch_size, with_loss=False):
    lossf = nn.CrossEntropyLoss()
    probs, ys, losses, nb = [], [], 0.0, 0
    with torch.no_grad():
        for b in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[b:b + batch_size]).unsqueeze(1).to(device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, choices=["eeg", "eog"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit-check", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    t0 = time.time()

    rec_split = load_split_records()
    train_recs = [r for r, s in rec_split.items() if s == "train"]
    val_recs = [r for r, s in rec_split.items() if s == "val"]
    ch_idx = 0 if args.channel == "eeg" else 1

    Xtr, ytr, mtr = load_channel(ch_idx, train_recs)
    Xva, yva, mva = load_channel(ch_idx, val_recs)

    # normalization stats from TRAIN set only (team rule)
    mean, std = float(Xtr.mean()), float(Xtr.std())
    Xtr = ((Xtr - mean) / std).astype(np.float32)
    Xva = ((Xva - mean) / std).astype(np.float32)

    if args.overfit_check:
        # step 6: model must fit a tiny batch to ~100% — pipeline sanity only
        n = 128
        X = torch.from_numpy(Xtr[:n]).unsqueeze(1).to(device)
        y = torch.from_numpy(ytr[:n]).long().to(device)
        model = SmallCNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        lossf = nn.CrossEntropyLoss()
        print(f"overfit check: {n} samples, {args.channel}")
        for ep in range(300):
            out = model(X)
            loss = lossf(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            acc = (out.argmax(1) == y).float().mean().item()
            if ep % 25 == 0 or acc > 0.99:
                print(f"  ep {ep:3d} loss {loss.item():.4f} acc {acc:.4f}")
            if loss.item() < 0.01 and acc > 0.99:
                print(f"OVERFIT CHECK PASSED at epoch {ep} (acc {acc:.4f}, loss {loss.item():.4f})")
                return
        print("overfit check: reached max epochs without full overfit (see acc above)")
        return

    run_id = args.run_id or f"E1_{args.channel.upper()}_label100_seed{args.seed}_v01"
    out_dir = RUNS / run_id
    if out_dir.exists():
        sys.exit(f"run folder already exists: {out_dir} (team rule: one id per run; pick a new id)")
    out_dir.mkdir(parents=True)

    def class_counts(y):
        return {STAGE_NAMES[c]: int((y == c).sum()) for c in range(N_CLASSES)}

    cfg = {
        "run_id": run_id, "experiment": "E1", "owner": "A",
        "model": "SmallCNN (team-doc reference structure)",
        "channel": args.channel, "channel_name": "EEG Fpz-Cz" if ch_idx == 0 else "EOG horizontal",
        "n_classes": N_CLASSES, "input_points": EPOCH_PTS,
        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "early_stopping": {"monitor": "val_macro_f1", "patience": 10},
        "seed": args.seed, "device": device,
        "label_frac": 1.0, "code_version": CODE_VERSION,
        "normalization": {"method": "z-score", "mean": mean, "std": std,
                          "computed_on": "train subjects only"},
        "n_train_epochs": int(len(Xtr)), "n_val_epochs": int(len(Xva)),
        "train_class_counts": class_counts(ytr), "val_class_counts": class_counts(yva),
        "torch": str(torch.__version__), "cuda": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    print(json.dumps(cfg, indent=2, default=str))
    # write config + README up front so a failed run still leaves a record (team save rules)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    (out_dir / "README.md").write_text(
        f"# {run_id}\n\n"
        f"- 负责人: A (实验① 数据与单模态基线)\n"
        f"- 内容: 小 CNN 在 **{args.channel.upper()}** 单通道上的 100% 标签训练\n"
        f"- 状态: 运行中 (started)\n", encoding="utf-8")

    model = SmallCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}   train epochs: {len(Xtr)}  val epochs: {len(Xva)}")
    best_epoch = train(model, Xtr, ytr, Xva, yva, device, cfg, out_dir)
    # rewrite config.yaml now that epochs_run is known (upfront copy kept for crash forensics)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # final evaluation with best weights + validation predictions (for D)
    model.load_state_dict(torch.load(out_dir / "best.ckpt", weights_only=True)["state_dict"])
    model.eval()
    yp, yt = predict(model, Xva, yva, device, cfg["batch_size"])
    m = compute_metrics(yt, yp, STAGE_NAMES)
    m["run_id"] = run_id
    m["n_params"] = n_params
    m["train_minutes"] = round((time.time() - t0) / 60, 2)
    if device == "cuda":
        m["gpu_mem_MiB_peak"] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
    save_metrics(m, out_dir / "metrics.json")

    # predictions_val.csv in the group format (one row per epoch)
    with torch.no_grad():
        P = torch.softmax(torch.cat([
            model(torch.from_numpy(Xva[b:b + args.batch_size]).unsqueeze(1).to(device))
            for b in range(0, len(Xva), args.batch_size)]), 1).cpu().numpy()
    pred_df = mva.copy()
    pred_df["subject_id"] = pred_df["record_id"].str[:5]
    pred_df["y_true"] = yt
    pred_df["y_pred"] = yp
    for c, col in enumerate(PROB_COLS):
        pred_df[col] = np.round(P[:, c], 6)
    pred_df[["sample_id", "subject_id", "record_id", "epoch_index", "y_true", "y_pred",
             *PROB_COLS]].to_csv(out_dir / "predictions_val.csv", index=False)

    # update README with final status (team save rules)
    (out_dir / "README.md").write_text(
        f"# {run_id}\n\n"
        f"- 负责人: A (实验① 数据与单模态基线)\n"
        f"- 内容: 小 CNN 在 **{args.channel.upper()}** 单通道上的 100% 标签训练\n"
        f"- 状态: 成功\n"
        f"- 主指标 val Macro-F1: **{m['macro_f1']:.4f}** (best epoch {best_epoch})\n"
        f"- 训练耗时: {m['train_minutes']} min; "
        f"GPU 峰值显存: {m.get('gpu_mem_MiB_peak', 'n/a')} MiB\n"
        f"- 评估脚本: 暂用本仓库 provisional evaluate.py,收到 B 的官方脚本后需重算\n", encoding="utf-8")

    print("\n" + metrics_table(m, STAGE_NAMES))
    print(f"\nwrote run folder: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
