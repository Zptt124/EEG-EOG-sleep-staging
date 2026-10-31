"""Step W2-B8: 全组预测表滚动核验——遍历 git 引用里的 predictions_val*.csv,逐个过官方评估 + 验证参考清单。

对每个不同内容(按 blob SHA 去重)的预测表检查:列/ID 唯一/标签/概率/argmax(evaluate.py v1)、
是否恰好覆盖 8 人 21186 段、y_true 与真值一致;通过者记录 Macro-F1 与 N1 P/R/F1。失败原因原样写入,
不修复、不剔除。只读 git 对象,不检出分支。

用法:
  python scripts/16_audit_predictions.py origin/main origin/feat/b-evaluation origin/feat/c-eegdm \
      origin/feat/d-fusion --out reports/week2/b_prediction_audit.csv
"""
import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

from common import BASE
import w2_common as w

ev = w.ev
FIELDS = ["blob_sha", "status", "error", "n_rows", "macro_f1", "n1_precision", "n1_recall", "n1_f1",
          "first_ref", "first_path", "n_copies"]


def git(*args, binary=False):
    out = subprocess.run(["git", *args], cwd=BASE, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode("utf-8")


def find_predictions(refs):
    found = {}
    for ref in refs:
        for line in git("ls-tree", "-r", "-z", ref).split("\0"):
            if not line:
                continue
            meta, path = line.split("\t", 1)
            name = Path(path).name
            if name.startswith("predictions_val") and name.endswith(".csv") and "template" not in name:
                blob = meta.split()[2]
                found.setdefault(blob, []).append((ref, path))
    return found


def audit(blob, copies):
    row = dict(blob_sha=blob, first_ref=copies[0][0], first_path=copies[0][1], n_copies=len(copies))
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "p.csv"
        f.write_bytes(git("cat-file", "blob", blob, binary=True))
        try:
            m = ev.evaluate(f, reference=w.VAL_REFERENCE)
        except ValueError as e:
            return dict(row, status="FAIL", error=str(e).replace("\n", " ")[:200])
        except Exception as e:  # 非 CSV 等
            return dict(row, status="FAIL", error=f"{type(e).__name__}: {e}"[:200])
    return dict(row, status="PASS", error="", n_rows=m["n_epochs"], macro_f1=f"{m['macro_f1']:.4f}",
                n1_precision=f"{m['n1_precision']:.4f}", n1_recall=f"{m['n1_recall']:.4f}", n1_f1=f"{m['n1_f1']:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--out", default="reports/week2/b_prediction_audit.csv")
    args = ap.parse_args(argv)
    found = find_predictions(args.refs)
    rows = sorted((audit(b, c) for b, c in found.items()), key=lambda r: (r["status"], r["first_path"]))
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        wr.writeheader()
        wr.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    n_pass = sum(r["status"] == "PASS" for r in rows)
    print(f"{len(rows)} 个不同内容的预测表:PASS {n_pass},FAIL {len(rows) - n_pass} -> {args.out}")
    for r in rows:
        if r["status"] == "FAIL":
            print(f"  FAIL {r['first_ref']}:{r['first_path']}  {r['error']}")


if __name__ == "__main__":
    main()
