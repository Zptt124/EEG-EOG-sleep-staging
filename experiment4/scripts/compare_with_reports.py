#compare_with_reports.py —— 核对"重新跑出来的结果"和"仓库里提交的结果"是否一致
#做法：reports/ 里每个 csv / json，如果 real_data/ 里有同名同位置的文件，就逐格比较
#   数字允许极小的误差（不同电脑、不同 pandas 版本打印小数的方式可能略有不同）
#   文字必须完全一样
#只要有一个文件对不上，就报错并终止
#用法（Ubuntu，在仓库根目录，先按 README 的顺序跑完分析脚本）：
#   python experiment4/scripts/compare_with_reports.py

import json
import math
import sys
from pathlib import Path

import pandas as pd

EXP4 = Path(__file__).resolve().parents[1]
REPORTS = EXP4 / "reports"
DATA = EXP4 / "real_data"
#这两个文件夹是原始记录，不是分析脚本的输出，不参与比较
SKIP_DIRS = {"predictions_E4", "runs_E4"}
RTOL, ATOL = 1e-6, 1e-9
#json里记录"输入文件路径"的字段，每台电脑都不一样，不参与比较
IGNORE_KEYS = {"source"}
#已知会不一样、并且原因已经写进README的文件：只提示，不算错误
KNOWN_DIFF = {
    "cases/typical_cases.csv": "category B was picked with the old non-stable sort; many candidates tie at "
                               "probability 1.0, so the stable sort in pick_cases.py picks an equally valid set",
}


def close(a, b):
    #两个值是否"相等"：数字按误差比较，其余按原样比较
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = a.keys() - IGNORE_KEYS, b.keys() - IGNORE_KEYS
        return ka == kb and all(close(a[k], b[k]) for k in ka)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def same_csv(p, q):
    a, b = pd.read_csv(p), pd.read_csv(q)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if not all(close(float(x), float(y)) for x, y in zip(a[c], b[c])):
                return False
        elif not a[c].astype(str).equals(b[c].astype(str)):
            return False
    return True


def same_json(p, q):
    return close(json.loads(p.read_text(encoding="utf-8")), json.loads(q.read_text(encoding="utf-8")))


compared, skipped, bad = 0, 0, []
for p in sorted(REPORTS.rglob("*")):
    rel = p.relative_to(REPORTS)
    if not p.is_file() or rel.parts[0] in SKIP_DIRS or p.suffix not in (".csv", ".json"):
        continue
    q = DATA / rel
    if not q.exists():
        print(f"SKIP: {rel} (not regenerated)")
        skipped += 1
        continue
    ok = same_csv(p, q) if p.suffix == ".csv" else same_json(p, q)
    compared += 1
    key = rel.as_posix()
    if not ok and key in KNOWN_DIFF:
        print(f"KNOWN: {key} ({KNOWN_DIFF[key]})")
        continue
    print(f"{'OK  ' if ok else 'DIFF'}: {key}")
    if not ok:
        bad.append(key)

print(f"\nCompared {compared} files, skipped {skipped}")
if compared == 0:
    print("ERROR: nothing was compared. Run the analysis scripts first.")
    sys.exit(1)
if bad:
    print(f"ERROR: {len(bad)} files differ: {bad}")
    sys.exit(1)
print("DONE: all regenerated results match reports/")
