"""Step W3-B5: C 的 6 个 NPZ 到位后,一次补齐 B 第三周剩余实验并统一复评。

  1. 在 --npz-dir 下按文件名找 6 个 *_pooled_53496x2560_unstandardized.npz,
     按 experiment3/week3/deliveries/B_week3_features/delivery_manifest.json 校验 SHA256。
  2. 27_b_w3_linear_probe.py:mean_t1 × label10/label25 × seed 0-2;std_t10、std_t25 × label25 × seed 0。
     样本索引用交付目录里的 sample_index.csv;已存在的 run 跳过。
  3. 28_b_n1_score_adjust.py --arms std_t1 mean_t1。
  4. 29_b_w3_evaluate.py:登记 B、A v02/v01、C,输出全部配对。

用法:
  python scripts/30_b_w3_finish.py --npz-dir <放 6 个 NPZ 的目录>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import BASE, RUNS
import w2_common as w

DELIVERY = BASE / "experiment3/week3/deliveries/B_week3_features"
REPORTS = BASE / "reports/week3"
JOBS = ([("mean_t1", b, s) for b in ("label25", "label10") for s in range(3)]
        + [(arm, "label25", 0) for arm in ("std_t10", "std_t25")])
EXTRA = ["b_w2_devselected_registry.csv", "b_a_w3_registry.csv", "b_a_w3_v01devsel_registry.csv", "b_c_w3_registry.csv"]
PAIRS = ["mean_t1:std_t1", "std_t10:std_t1", "std_t25:std_t1", "std_t1:std_t1_w2devsel",
         "A_contrawr_dual:A_cnn_dual", "A_cnn_dual:A_cnn_eeg", "std_t1:A_cnn_dual", "std_t1:A_contrawr_dual",
         "mean_t1:A_contrawr_dual", "A_cnn_dual:A_cnn_dual_v01devsel", "A_contrawr_dual:A_contrawr_dual_v01devsel",
         "C_lft_std_t1:C_lin_std_t1", "C_lft_mean_t1:C_lft_std_t1", "C_lin_mean_t1:C_lin_std_t1",
         "C_lft6seg_std_t1:C_lft_std_t1", "C_lft_std_t1_last5:C_lft_std_t1", "C_lin_std_t1_last5:C_lin_std_t1",
         "C_tconv_std_t1:C_lft_std_t1", "C_lft_std_t2:C_lft_std_t1", "C_lft_std_t3:C_lft_std_t1",
         "C_lin_std_t1:std_t1", "C_lin_mean_t1:mean_t1", "C_lft_std_t1:std_t1", "C_lft_std_t1:A_contrawr_dual"]


def run(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], cwd=BASE / "scripts", check=True)


def locate(npz_dir):
    manifest = json.loads((DELIVERY / "delivery_manifest.json").read_text(encoding="utf-8"))
    found = {}
    for bundle, info in manifest["bundles"].items():
        name = next(n for n in info["sha256"] if n.endswith(".npz"))
        hits = list(Path(npz_dir).rglob(name))
        if len(hits) != 1:
            sys.exit(f"错误: {name} 在 {npz_dir} 下找到 {len(hits)} 个")
        if w.sha256_file(hits[0]) != info["sha256"][name]:
            sys.exit(f"错误: {name} 的 SHA256 与交付清单不一致")
        if w.sha256_file(DELIVERY / bundle / "sample_index.csv") != info["sha256"]["sample_index.csv"]:
            sys.exit(f"错误: {bundle}/sample_index.csv 的 SHA256 与交付清单不一致")
        found[bundle] = hits[0]
        print(f"SHA256 通过: {name}", flush=True)
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--npz-dir", required=True)
    args = ap.parse_args(argv)

    npz = locate(args.npz_dir)
    for arm, budget, seed in JOBS:
        if (RUNS / f"W3_B_{arm}_{budget}_sseed2024_seed{seed}_v01").exists():
            print(f"跳过已存在: {arm} {budget} seed{seed}", flush=True)
            continue
        reduce, t = arm.split("_")
        eeg, eog = f"eeg_{reduce}_{t}_cond14_label25", f"eog_{reduce}_{t}_cond14_label25"
        run("27_b_w3_linear_probe.py", "--arm", arm, "--budget", budget, "--seed", seed,
            "--eeg-pooled", npz[eeg], "--eog-pooled", npz[eog],
            "--eeg-index", DELIVERY / eeg / "sample_index.csv", "--eog-index", DELIVERY / eog / "sample_index.csv")
    run("28_b_n1_score_adjust.py", "--arms", "std_t1", "mean_t1")
    run("29_b_w3_evaluate.py", "--extra", *[REPORTS / f for f in EXTRA], "--pairs", *PAIRS)


if __name__ == "__main__":
    main()
