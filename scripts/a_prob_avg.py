"""a_prob_avg.py — 实验① A:两单模态概率平均(W2 v1.1,无新增训练)。

同一预算、同一下游 seed 的 EEG 与 EOG 预测表,五类概率逐段平均后取 argmax。
样本集合与 y_true 必须完全一致(逐 sample_id 对齐检查),输出仍为 11 列表,
并用 B 官方 evaluate.py(带 --reference)产出 metrics.json。

用法:
    D:/python/python.exe scripts/a_prob_avg.py ^
        --eeg <runs>/W2_A_EEG_label25_sseed2024_seed0_v01/predictions_val.csv ^
        --eog <runs>/W2_A_EOG_label25_sseed2024_seed0_v01/predictions_val.csv ^
        --out <runs>/W2_A_PROBAVG_label25_sseed2024_seed0_v01 ^
        --repo-root .
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROB_COLS = ["p_W", "p_N1", "p_N2", "p_N3", "p_REM"]


def pairing_of(pred_csv: Path) -> tuple[str, str]:
    """从预测表所在 run 的 config.yaml 取 (nominal_budget, downstream_seed)。

    第一周 run 无 nominal_budget,按 label_frac==1.0 映射为 label100。
    """
    run_dir = Path(pred_csv).parent
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"找不到源 run 配置: {cfg_path}(概率平均必须来自登记 run)")
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if "nominal_budget" in cfg and cfg["nominal_budget"]:
        return cfg["nominal_budget"], str(cfg["downstream_seed"])
    if cfg.get("label_frac") == 1.0:            # 第一周 100% 标签 run
        return "label100", str(cfg.get("seed"))
    sys.exit(f"无法从 {cfg_path} 解析预算/seed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eeg", required=True)
    ap.add_argument("--eog", required=True)
    ap.add_argument("--out", required=True, help="输出文件夹(写 11 列表 + metrics.json)")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    # 配对校验:两源表必须同一预算、同一下游 seed(协议v2:逐种子配对平均)
    pair_e = pairing_of(Path(args.eeg))
    pair_o = pairing_of(Path(args.eog))
    if pair_e != pair_o:
        sys.exit(f"预算/seed 不配对: EEG {pair_e} != EOG {pair_o}")
    budget, seed = pair_e
    print(f"pairing ok: {budget} seed {seed}")

    a = pd.read_csv(args.eeg)
    b = pd.read_csv(args.eog)
    a = a.sort_values("sample_id").reset_index(drop=True)
    b = b.sort_values("sample_id").reset_index(drop=True)
    if not a["sample_id"].equals(b["sample_id"]):
        sys.exit("两表 sample_id 集合不一致,不能平均")
    if not a["y_true"].equals(b["y_true"]):
        sys.exit("两表 y_true 不一致,不能平均")
    assert a["subject_id"].equals(b["subject_id"]) and a["record_id"].equals(b["record_id"])
    assert a["epoch_index"].equals(b["epoch_index"])

    P = np.round((a[PROB_COLS].to_numpy(float) + b[PROB_COLS].to_numpy(float)) / 2, 6)
    out_df = a[["sample_id", "subject_id", "record_id", "epoch_index", "y_true"]].copy()
    out_df["y_pred"] = P.argmax(1)
    for c, col in enumerate(PROB_COLS):
        out_df[col] = P[:, c]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "predictions_val.csv", index=False)

    val_ref = repo_root / "manifests" / "val_reference_v1.csv"
    cmd = [sys.executable, str(repo_root / "src" / "evaluation" / "evaluate.py"),
           str(out_dir / "predictions_val.csv"), "--out", str(out_dir / "metrics.json"),
           "--reference", str(val_ref)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"B 官方评估失败: {res.stderr.strip()}")
    m = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    (out_dir / "README.md").write_text(
        f"# {out_dir.name}\n\n"
        f"- 负责人: A(实验① 概率平均,无新增训练)\n"
        f"- 配方: (EEG + EOG) 五类概率逐段平均后 argmax\n"
        f"- 预算/seed: {budget} / seed {seed}\n"
        f"- 源表: {args.eeg}\n  {args.eog}\n"
        f"- val Macro-F1: **{m['macro_f1']:.4f}**  N1 F1: {m['n1_f1']:.4f}\n",
        encoding="utf-8")
    print(f"val Macro-F1: {m['macro_f1']:.4f}   N1 F1: {m['n1_f1']:.4f}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
