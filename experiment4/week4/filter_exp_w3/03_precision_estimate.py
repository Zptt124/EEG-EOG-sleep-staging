"""D：实验一主指标 ΔB−ΔA 的精度估计（正态近似，开跑前规划用，不看任何新得分）。

输入：关灯窗口现有"新人 − 随机"的 95% 配对区间（D 的 analysis_lights_off，固定尝试口径，26 份 × 5 次），
      以及历史跨窗口配对区间（旧 → 关灯，−1.32 [−2.34, −0.38]）。
假设：两个条件的 Δ 方差相同（SE_Δ 取关灯窗口区间半宽 / 1.96），条件间相关系数为 ρ。
      SE(ΔB−ΔA) = SE_Δ × sqrt(2(1−ρ))；最小可检测差 MDE（双侧 5%、功效 80%）= 2.80 × SE；
      检出真实差 δ 的功效 = Φ(δ/SE − 1.96)（忽略另一侧尾部）。
另报：历史跨窗口设计隐含的 ρ（由两窗口各自区间与跨窗口区间反推）。
"""
import argparse
import csv
import json
import math
from pathlib import Path


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lights-summary", type=Path, required=True)
    ap.add_argument("--cross-window", type=Path, required=True)
    ap.add_argument("--overlap", type=float, required=True, help="mean share of selected blocks shared between A and B")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.cross_window)))
    get = lambda q: next(r for r in rows if r["contrast"] == "new_minus_random" and r["analysis"] == "fixed_600_attempts" and r["quantity"] == q)
    se = lambda r: (float(r["ci95_hi"]) - float(r["ci95_lo"])) / 3.92
    se_old, se_l, se_d = se(get("old_window")), se(get("lights_off_window")), se(get("lights_minus_old"))
    rho_hist = (se_old ** 2 + se_l ** 2 - se_d ** 2) / (2 * se_old * se_l)
    out = []
    for rho in (rho_hist, 0.7, 0.8, 0.9, 0.95):
        s = se_l * math.sqrt(2 * (1 - rho))
        out.append(dict(rho=round(rho, 3), se_diff_pp=round(s, 3), ci95_halfwidth_pp=round(1.96 * s, 3),
                        mde80_pp=round(2.80 * s, 3), power_delta_0_5=round(phi(0.5 / s - 1.96), 3),
                        power_delta_1_0=round(phi(1.0 / s - 1.96), 3),
                        note="historical cross-window design" if rho == rho_hist else "assumed"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(json.dumps(dict(se_old=round(se_old, 3), se_lights=round(se_l, 3), se_cross=round(se_d, 3),
                          rho_hist=round(rho_hist, 3), block_overlap_A_B=a.overlap)))
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
