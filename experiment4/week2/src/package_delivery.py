"""package_delivery.py - Week-2 Experiment 4 (D): build the delivery for the team repo and the GitHub Release.

Never modifies the original results. It copies into <out>/repo/experiment4/week2/ (small files, for git) and
<out>/release/w2_d_artifacts_v01.zip (weights and large traces, for a GitHub Release), following CONTRIBUTING.md:
  - predictions_val.csv files go to git (same as B's and C's week-2 runs);
  - weights (best.ckpt, teacher ckpts) and the large pseudo_label_trace.csv files go to the Release zip only,
    each listed with sha256 and size in artifact_index_D.csv;
  - local absolute paths are replaced by repo-relative paths or placeholders in every copied text file,
    and the script fails if any local path survives;
  - because path replacement changes .py hashes, the byte-identical original code (src/ and src_snapshots/)
    is also put in the Release zip, and code_sha_map.csv maps original sha256 -> delivered sha256, so the
    code_sha256 recorded in every run config stays verifiable.

Usage (Ubuntu):
  python -u package_delivery.py --exp-dir <D_local_project>/week2/exp4_ssl --out <D_local_project>/week2/delivery
"""

import argparse
import csv
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

RELEASE_TAG = "d-ssl-w2-v01-20261101"
ZIP_NAME = "w2_d_artifacts_v01.zip"
REPO_PREFIX = "experiment4/week2"
TEXT_EXT = {".csv", ".json", ".yaml", ".yml", ".md", ".py", ".txt"}
# Local path pieces are joined at run time so this file itself contains no local absolute path.
_MNT = "/" + "mnt/d"
_HOME = "/" + "home/" + "langt"
_PROJ = _MNT + "/03_Study/Research/sleepdm-eeg-eog"
# longest prefixes first
PATH_MAP = [
    (_PROJ + "/week2/exp4_ssl", REPO_PREFIX),
    (_PROJ + "/week1/inputs/B_feat-b-evaluation", "<B_feat-b-evaluation>"),
    (_PROJ + "/week2/logs", "<D_local_logs>"),
    (_PROJ, "<D_local_project>"),
    (_HOME + "/week1/C_experiment3/features", "<D_local_eegdm_features>"),
    (_HOME + "/week2/cache", "<D_local_cache>"),
    (_HOME + "/week2/r2_fold_cnns", "<D_local_r2_fold_cnns>"),
    (_HOME + "/week2/a_release", "<A_release_a-cnn-w2-v01-20261031>"),
    (_HOME + "/repos/w2_main", "<team_repo>"),
    (_HOME + "/data/sleepedf_sc_processed", "<A_processed_npz>"),
    (_HOME, "<D_local_home>"),
    (_MNT + "/...", "<D_local_project>"),
]
FORBIDDEN = [_MNT + "/", _HOME, "D:" + "\\", "langt" + "@"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_text(src, dst):
    # copy with local paths replaced; binary files are copied byte for byte
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() not in TEXT_EXT:
        shutil.copyfile(src, dst)
        return
    s = src.read_text(encoding="utf-8")
    for old, new in PATH_MAP:
        s = s.replace(old, new)
    for bad in FORBIDDEN:
        if bad in s:
            print(f"ERROR: local path '{bad}' still present in {src}")
            sys.exit(1)
    dst.write_text(s, encoding="utf-8", newline="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        print(f"ERROR: {args.out} already exists, choose a new folder")
        sys.exit(1)
    exp = args.exp_dir
    repo = args.out / "repo" / REPO_PREFIX
    rel_dir = args.out / "release"
    rel_dir.mkdir(parents=True)

    # ---------- small files for git ----------
    n_repo = 0
    for sub in ("src", "manifests", "calibration", "analysis", "src_snapshots"):
        for f in sorted((exp / sub).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                copy_text(f, repo / f.relative_to(exp))
                n_repo += 1
    copy_text(exp / "semisupervised_config.yaml", repo / "semisupervised_config.yaml")
    copy_text(exp / "README.md", repo / "README.md")
    n_repo += 2
    runs = list(csv.DictReader(open(exp / "runs" / "run_index.csv", encoding="utf-8")))
    if len(runs) != 60:
        print(f"ERROR: expected 60 runs in run_index.csv, found {len(runs)}")
        sys.exit(1)
    for name in ("run_index.csv", "teacher_index.csv"):
        copy_text(exp / "runs" / name, repo / "runs" / name)
        n_repo += 1
    keep = ["README.md", "config.yaml", "history.csv", "metrics.json", "predictions_val.csv", "pseudo_label_stats.csv"]
    for r in runs:
        d = exp / "runs" / r["run_id"]
        for name in keep:
            if (d / name).exists():
                copy_text(d / name, repo / "runs" / r["run_id"] / name)
                n_repo += 1
            elif name != "pseudo_label_stats.csv" or r["method"] != "D0":
                print(f"ERROR: {d / name} missing")
                sys.exit(1)
    for tdir in sorted((exp / "runs" / "teachers").iterdir()):
        for f in sorted(tdir.glob("history_*.csv")):
            copy_text(f, repo / "runs" / "teachers" / tdir.name / f.name)
            n_repo += 1

    # ---------- code hash map (original vs delivered, path replacement changes .py hashes) ----------
    code_rows = []
    for sub in ("src", "src_snapshots"):
        for f in sorted((exp / sub).rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            rel = f.relative_to(exp).as_posix()
            code_rows.append({"path": rel, "original_sha256": sha256(f), "delivered_sha256": sha256(repo / rel),
                              "original_in_release": f"original_code/{rel}"})
    with (repo / "code_sha_map.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(code_rows[0].keys()))
        w.writeheader()
        w.writerows(code_rows)

    # ---------- large files for the Release zip ----------
    big = [(exp / r["path"], r["original_in_release"], "original_code") for r in code_rows]
    for r in runs:
        d = exp / "runs" / r["run_id"]
        big.append((d / "best.ckpt", f"runs/{r['run_id']}/best.ckpt", "student_weights"))
        if (d / "pseudo_label_trace.csv").exists():
            big.append((d / "pseudo_label_trace.csv", f"runs/{r['run_id']}/pseudo_label_trace.csv", "pseudo_label_trace"))
    for tdir in sorted((exp / "runs" / "teachers").iterdir()):
        for f in sorted(tdir.iterdir()):
            if f.suffix == ".ckpt":
                big.append((f, f"runs/teachers/{tdir.name}/{f.name}", "teacher_weights"))
            elif f.name.startswith("predictions_val_"):
                big.append((f, f"runs/teachers/{tdir.name}/{f.name}", "teacher_predictions_val"))
    index_rows = []
    zpath = rel_dir / ZIP_NAME
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for k, (src, arc, kind) in enumerate(big, start=1):
            z.write(src, arc)
            index_rows.append({"artifact_path": arc, "artifact_type": kind, "sha256": sha256(src),
                               "size_bytes": src.stat().st_size,
                               "storage_uri": f"https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/{RELEASE_TAG}/{ZIP_NAME}#/{arc}"})
            if k % 50 == 0:
                print(f"  zipped {k}/{len(big)}", flush=True)
    with (repo / "artifact_index_D.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)
    zsha = sha256(zpath)
    (rel_dir / f"{ZIP_NAME}.sha256").write_text(f"{zsha}  {ZIP_NAME}\n", encoding="utf-8")
    (repo / "release_info.txt").write_text(
        f"release_tag: {RELEASE_TAG}\nasset: {ZIP_NAME}\nasset_sha256: {zsha}\nasset_size_bytes: {zpath.stat().st_size}\n"
        f"files_in_asset: {len(index_rows)} (per-file sha256 in artifact_index_D.csv)\n", encoding="utf-8")

    # ---------- final check: no local path anywhere in the git part ----------
    for f in repo.rglob("*"):
        if f.is_file() and f.suffix.lower() in TEXT_EXT:
            s = f.read_text(encoding="utf-8")
            if any(bad in s for bad in FORBIDDEN):
                print(f"ERROR: local path found in {f}")
                sys.exit(1)
    size_repo = sum(f.stat().st_size for f in repo.rglob("*") if f.is_file())
    print(f"git part: {n_repo + 3} files, {size_repo / 1e6:.1f} MB -> {repo}")
    print(f"release asset: {len(index_rows)} files, {zpath.stat().st_size / 1e6:.1f} MB -> {zpath}")
    print(f"release asset sha256: {zsha}")
    print("OK: no local absolute paths in the git part")


if __name__ == "__main__":
    main()
