"""D：实验二 10 次重复设计的预计精度（开跑前，用第一轮归档的条件 A 分数；只重抽名单）。

模型：名单 r 的 (V − R) 均值 = 真效应 + 名单间异质 τ_r + 抽块噪声均值。
用 P（新人，同为随机化策略）代理 V：var(P̄−R̄) = τ² + σ_P²/5 + σ_R²/5，由此反推 τ²，
再代入 10 次：半宽 = 1.96·sqrt((τ² + σ_V²/10 + σ_R²/10)/26)，σ_V 取 σ_P。未计入评估者重抽，实际会略宽。
"""
import json, sys
import numpy as np, pandas as pd
out = {}
for name in ("primary_linear", "bigru_ab"):
    s = pd.read_csv(f"{sys.argv[1]}/{name}/scores.csv"); s = s[s.condition == "A"]
    p = s.pivot_table(index=["roster", "draw"], columns="strategy", values="macro_f1_percent")
    sdP = p["add_people"].groupby(level=0).std().pow(2).mean() ** .5
    sdR = p["random"].groupby(level=0).std().pow(2).mean() ** .5
    d = p["add_people"].groupby(level=0).mean() - p["random"].groupby(level=0).mean()
    tau2 = max(d.var(ddof=1) - (sdP**2 + sdR**2) / 5, 0.0)
    res = {}
    for n in (5, 10):
        hw = 1.96 * np.sqrt((tau2 + sdP**2 / n + sdR**2 / n) / 26)
        res[f"draws_{n}"] = dict(half_width_pp=round(float(hw), 3), mde_80pct_pp=round(float(hw / 1.96 * 2.80), 3))
    out[name] = dict(within_roster_sd_P=round(float(sdP), 3), within_roster_sd_R=round(float(sdR), 3),
                     between_roster_sd_tau=round(float(tau2**.5), 3), **res)
    print(name, json.dumps(out[name]))
open(sys.argv[2], "w").write(json.dumps(out, indent=2) + "\n")
