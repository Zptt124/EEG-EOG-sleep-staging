"""Step W2-B16: 复核 C 的补充实验预测(W2X_*)。

每个 W2X_C_* run 的 predictions_val.csv 过官方 evaluate.py v1 + 验证参考清单,指标与该 run 的 metrics.json 逐项比对。
这些是探索性补充,不进入主比较、曲线和 run_registry 的主矩阵。

用法:
  python scripts/24_check_c_supplement.py --runs experiment3/week2/runs --out reports/week2/external_C/supplement_check.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import w2_common as w

ev = w.ev
TOL = 5e-6


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = []
    for d in sorted(Path(args.runs).glob("W2X_C_*")):
        m = ev.evaluate(d / "predictions_val.csv", reference=w.VAL_REFERENCE)
        reported = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        bad = [k for k in ("macro_f1", "n1_f1") if abs(m[k] - reported[k]) > TOL]
        rows.append(dict(run_id=d.name, macro_f1=round(m["macro_f1"], 6), n1_f1=round(m["n1_f1"], 6),
                         sample_set_sha256=m["sample_set_sha256"], mismatch=";".join(bad)))
    t = pd.DataFrame(rows)
    t.to_csv(args.out, index=False)
    print(f"{len(t)} 个补充 run 通过官方评估,样本集合 {t.sample_set_sha256.nunique()} 种,与 metrics.json 不一致 {(t.mismatch != '').sum()} 个 -> {args.out}")


if __name__ == "__main__":
    main()
