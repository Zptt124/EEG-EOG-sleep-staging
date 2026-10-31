"""Step W2-B9: 把其他成员已发布的 run 并入全组统一口径(只读 git 对象,不合并分支)。

对 C 的 run_registry.csv 中每个 run:取 predictions_val.csv,过官方 evaluate.py v1 + 验证参考清单(重算,
不信任对方 metrics),并与对方登记的 Macro-F1 / N1 指标逐项比对,不一致写入 mismatch 列而不是静默覆盖。
同一 modality/预算/下游 seed 下,S1(各适配 seed)减 S0 做同样本配对,按受试者 bootstrap。
输出到独立目录,**不改**共享的 reports/week2/metrics_summary.csv 等(避免与对方分支同名文件冲突)。

用法:
  python scripts/17_ingest_external_runs.py --ref origin/feat/c-eegdm \
      --registry experiment3/week2/reports/week2_final_C_20261029/run_registry.csv \
      --out-dir reports/week2/external_C
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from common import BASE
import w2_common as w

ev = w.ev
spec = importlib.util.spec_from_file_location("agg", Path(__file__).with_name("14_w2_aggregate.py"))
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)

TOL = 5e-5


def git_bytes(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=BASE, capture_output=True, check=True).stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--registry", required=True, help="对方 run_registry.csv 在该 ref 里的路径")
    ap.add_argument("--registry-dir-prefix", default=None, help="registry 里 predictions_path 的前缀目录(默认取 registry 所在目录的上级的上级)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    reg = list(csv.DictReader(io.StringIO(git_bytes(args.ref, args.registry).decode("utf-8-sig"))))
    base = args.registry_dir_prefix or str(Path(args.registry).parents[2])   # .../experiment3/week2
    commit = agg.evaluator_commit()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs, rows, subj_rows, mismatches = {}, [], [], []
    for r in reg:
        raw = git_bytes(args.ref, f"{base}/{r['predictions_path']}")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "p.csv"
            f.write_bytes(raw)
            m = ev.evaluate(f, reference=w.VAL_REFERENCE)            # 失败即抛错:不静默跳过
            df = ev.load_predictions(f)
        budget, down = r["budget"], int(r["downstream_seed"])
        s1 = r["representation"] == "S1"
        cfg = dict(run_id=r["run_id"], representation=f"C_{r['representation']}_EEGDM_raw" if not s1 else "C_S1_EEGDM_adapted",
                   modalities=r["modality"].replace("_", "+"), context_length=1, nominal_budget=budget,
                   labeled_subjects=int(r["train_subjects"]), unlabeled_signal_subjects=int(r["local_unlabeled_adaptation_subjects"]),
                   downstream_seed=down, adaptation_seed=r["adaptation_seed"] or "NA")
        runs[r["run_id"]] = (cfg, m, df, None)
        g = m["per_class_f1"]
        rows.append({"run_id": r["run_id"], "representation": cfg["representation"], "modalities": cfg["modalities"],
                     "context_length": 1, "nominal_budget": budget, "labeled_subjects": cfg["labeled_subjects"],
                     "unlabeled_signal_subjects": cfg["unlabeled_signal_subjects"],
                     "adaptation_version": "S1_steps10000" if s1 else "S0_raw", "adaptation_seed": cfg["adaptation_seed"],
                     "teacher_id": "NA", "downstream_seed": down, "best_epoch": r["best_epoch"],
                     "macro_f1": f"{m['macro_f1']:.6f}", "W_f1": f"{g['W']:.6f}", "N1_precision": f"{m['n1_precision']:.6f}",
                     "N1_recall": f"{m['n1_recall']:.6f}", "N1_f1": f"{m['n1_f1']:.6f}", "N2_f1": f"{g['N2']:.6f}",
                     "N3_f1": f"{g['N3']:.6f}", "REM_f1": f"{g['REM']:.6f}", "sample_count": m["n_epochs"],
                     "subject_count": m["n_subjects"], "metrics_path": f"{args.ref}:{base}/{Path(r['predictions_path']).parent}/metrics.json",
                     "predictions_path": f"{args.ref}:{base}/{r['predictions_path']}",
                     "predictions_sha256": hashlib.sha256(raw).hexdigest(), "by_subject_path": "by_subject.csv",
                     "n1_errors_path": "NA", "evaluator_commit": commit, "sample_set_sha256": m["sample_set_sha256"]})
        for t in ev.per_subject_table(df).to_dict("records"):
            subj_rows.append({"run_id": r["run_id"], **t})
        for ours, theirs in (("macro_f1", "macro_f1"), ("n1_precision", "n1_precision"), ("n1_recall", "n1_recall"),
                             ("n1_f1", "n1_f1")):
            if abs(m[ours] - float(r[theirs])) > TOL:
                mismatches.append((r["run_id"], ours, m[ours], float(r[theirs])))
        if abs(m["per_class_f1"]["REM"] - float(r["rem_f1"])) > TOL:
            mismatches.append((r["run_id"], "rem_f1", m["per_class_f1"]["REM"], float(r["rem_f1"])))

    pd.DataFrame(rows, columns=agg.SUMMARY_FIELDS).to_csv(out / "metrics_summary.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(out / "by_subject.csv", index=False)
    seeds = agg.seed_summary(rows)
    seeds.to_csv(out / "seed_summary.csv", index=False)

    pairs = []
    for rid, (cfg, *_rest) in runs.items():
        if cfg["representation"] != "C_S1_EEGDM_adapted":
            continue
        ref_id = rid.replace("_S1_", "_S0_").replace(f"_aseed{cfg['adaptation_seed']}", "")
        if ref_id not in runs:
            raise ValueError(f"找不到与 {rid} 配对的 S0: {ref_id}")
        p = agg.paired_row(runs[ref_id], runs[rid], out, args.n_boot)
        p["adaptation_seed"] = cfg["adaptation_seed"]
        p["by_subject_path"] = "by_subject.csv"
        p["limitations"] = (agg.LIMITS + ";S1 额外使用 54 名训练池无标签信号,不是相同信号条件下的纯架构比较")
        pairs.append(p)
    pd.DataFrame(pairs, columns=agg.PAIR_FIELDS).to_csv(out / "paired_comparison.csv", index=False)
    pd.DataFrame(mismatches, columns=["run_id", "metric", "recomputed_by_B", "reported_by_owner"]).to_csv(
        out / "mismatch_vs_reported.csv", index=False)
    print(f"{len(rows)} 个 run、{len(pairs)} 组 S1-S0 配对;与对方登记值不一致 {len(mismatches)} 项 -> {out}")


if __name__ == "__main__":
    main()
