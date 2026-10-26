#prepare_inputs.py —— 把实验④分析要用的所有输入表，按固定的短文件名放进 experiment4/real_data/
#为什么需要它：分析脚本（align、confusion、official_eval_all等）都从 real_data/ 读表，
#而这些表分散在 A、B、C 的提交和 D 自己的 reports/predictions_E4/ 里，文件名也各不相同。
#这个脚本按下面的清单逐个取出来、核对格式，再改成分析脚本认识的短名字。
#
#取文件的顺序：
#   1. 仓库工作区里如果有这个文件（比如B/C的分支已经合并），直接复制
#   2. 没有的话，用 git show <分支>:<路径> 从对应分支里读出来（不需要切换分支，也不改动别人的文件）
#每个文件来自哪个分支的哪个commit、SHA256是多少，都记进 real_data/_upstream/provenance.csv
#
#用法（Ubuntu，在仓库根目录）：
#   python experiment4/scripts/prepare_inputs.py
#   如果分支名不一样：python experiment4/scripts/prepare_inputs.py --b-ref origin/feat/b-evaluation --c-ref origin/feat/c-eegdm

import argparse
import csv
import hashlib
import io
import subprocess
import sys
from pathlib import Path

EXP4 = Path(__file__).resolve().parents[1]   #experiment4/
REPO = EXP4.parent                           #仓库根目录
DATA = EXP4 / "real_data"
UP = DATA / "_upstream"

PRED_COLS = ["sample_id", "subject_id", "record_id", "epoch_index", "y_true", "y_pred",
             "p_W", "p_N1", "p_N2", "p_N3", "p_REM"]
N_VAL = 21186

#---------- 清单：(放进real_data后的名字, 来自谁, 在那边的路径) ----------
#"a"=A（在main上）、"b"=B的分支、"c"=C的分支
A_PRED = "交付包/实验①交付包_20261018/3_验证集预测_给D"
UPSTREAM = [
    ("predictions_val_EEG.csv", "a", f"{A_PRED}/predictions_val_EEG.csv"),                     #E1_EEG_label100_seed0_v01
    ("predictions_val_EOG.csv", "a", f"{A_PRED}/predictions_val_EOG.csv"),                     #E1_EOG_label100_seed0_v01
    ("split_subjects.csv", "a", "manifests/split_subjects.csv"),
    ("predictions_val_CNN_EEG_label25.csv", "b", "reports/predictions_E2/predictions_val_E2_EEG_label25_sseed2024_seed0_v01.csv"),
    ("predictions_val_CNN_EOG_label25.csv", "b", "reports/predictions_E2/predictions_val_E2_EOG_label25_sseed2024_seed0_v01.csv"),
    ("_upstream/evaluate.py", "b", "src/evaluation/evaluate.py"),
    ("predictions_val_EEGDM_EEG_cond14.csv", "c", "experiment3/reports/predictions_E3/predictions_val_E3_EEG_label100_seed0_provisional_v01.csv"),
    ("predictions_val_EEGDM_EEG_cond14_label25.csv", "c", "experiment3/reports/predictions_E3/predictions_val_E3_EEG_label25_sseed2024_seed0_v01.csv"),
]

#D自己的预测表：reports/predictions_E4/里的正式文件名 -> 分析脚本用的短名字
OWN = [
    ("predictions_val_CNN_fusedavg.csv", "predictions_val_E4_CNN_fusedavg_label100_seed0_v01.csv"),
    ("predictions_val_CNN_fusedavg_label25.csv", "predictions_val_E4_CNN_fusedavg_label25_sseed2024_seed0_v01.csv"),
    ("predictions_val_EEGDM_EOG_cond14.csv", "predictions_val_E4_EEGDM_EOG_cond14_label100_seed0_v01.csv"),
    ("predictions_val_EEGDM_EOG_cond10.csv", "predictions_val_E4_EEGDM_EOG_cond10_label100_seed0_v01.csv"),
    ("predictions_val_EEGDM_fusedavg_cond14.csv", "predictions_val_E4_EEGDM_fusedavg_cond14_label100_seed0_v01.csv"),
    ("predictions_val_EEGDM_EEGEOG_cond14.csv", "predictions_val_E4_EEGDM_EEGEOG_cond14_label100_seed0_v01.csv"),
    ("predictions_val_EEGDM_EOG_cond14_label25.csv", "predictions_val_E4_EEGDM_EOG_cond14_label25_sseed2024_seed0_v01.csv"),
    ("predictions_val_EEGDM_fusedavg_cond14_label25.csv", "predictions_val_E4_EEGDM_fusedavg_cond14_label25_sseed2024_seed0_v01.csv"),
    ("predictions_val_EEGDM_EEGEOG_cond14_label25.csv", "predictions_val_E4_EEGDM_EEGEOG_cond14_label25_sseed2024_seed0_v01.csv"),
]


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def git(*args):
    #在仓库根目录运行一条git命令，返回输出的原始字节；失败返回None
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def fetch(ref, path):
    #先找工作区，再找分支；返回(文件内容字节, 来源说明, commit)
    local = REPO / path
    if local.is_file():
        head = git("rev-parse", "HEAD")
        return local.read_bytes(), "working tree", head.decode().strip() if head else ""
    data = git("show", f"{ref}:{path}")
    if data is None:
        fail(f"cannot find {path} in working tree or in {ref}. Did you run 'git fetch origin'?")
    commit = git("rev-parse", ref)
    return data, ref, commit.decode().strip() if commit else ""


def check_predictions(name, data):
    #预测表必须：11列标准表头、正好21186行、sample_id不重复、每行概率加起来约等于1
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    if not rows or list(rows[0].keys()) != PRED_COLS:
        fail(f"{name}: columns are not the standard 11 columns")
    if len(rows) != N_VAL:
        fail(f"{name}: {len(rows)} rows, expected {N_VAL}")
    if len({r["sample_id"] for r in rows}) != N_VAL:
        fail(f"{name}: duplicated sample_id")
    for r in rows:
        s = sum(float(r[c]) for c in PRED_COLS[6:])
        if not 0.98 <= s <= 1.02:
            fail(f"{name}: probabilities of {r['sample_id']} sum to {s}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-ref", default="origin/main")
    ap.add_argument("--b-ref", default="origin/feat/b-evaluation")
    ap.add_argument("--c-ref", default="origin/feat/c-eegdm")
    args = ap.parse_args()
    refs = {"a": args.a_ref, "b": args.b_ref, "c": args.c_ref}

    if git("rev-parse", "--show-toplevel") is None:
        fail(f"{REPO} is not a git repository")
    UP.mkdir(parents=True, exist_ok=True)
    records = []

    #---------- 第一部分：A、B、C 的文件 ----------
    for target, who, path in UPSTREAM:
        data, source, commit = fetch(refs[who], path)
        n = check_predictions(target, data) if target.startswith("predictions_val_") else ""
        (DATA / target).write_bytes(data)
        records.append([target, who.upper(), path, source, commit, hashlib.sha256(data).hexdigest(), n])
        print(f"OK: {target:48s} <- {who.upper()} {source}")

    #---------- 第二部分：D 自己的预测表 ----------
    own_dir = EXP4 / "reports" / "predictions_E4"
    for target, filename in OWN:
        p = own_dir / filename
        if not p.is_file():
            fail(f"missing {p}")
        data = p.read_bytes()
        n = check_predictions(target, data)
        (DATA / target).write_bytes(data)
        records.append([target, "D", f"experiment4/reports/predictions_E4/{filename}", "working tree", "",
                        hashlib.sha256(data).hexdigest(), n])
        print(f"OK: {target:48s} <- D reports/predictions_E4")

    #---------- 第三部分：所有预测表的 sample_id 和 y_true 必须彼此一致 ----------
    ref_rows = None
    for target, *_ in records:
        if not target.startswith("predictions_val_"):
            continue
        with open(DATA / target, encoding="utf-8-sig", newline="") as f:
            truth = {r["sample_id"]: r["y_true"] for r in csv.DictReader(f)}
        if ref_rows is None:
            ref_rows = truth
        elif truth != ref_rows:
            fail(f"{target}: sample_id or y_true differs from predictions_val_EEG.csv")
    print("OK: every prediction table has the same sample_id set and y_true")

    with open(UP / "provenance.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "owner", "source_path", "source", "commit", "sha256", "rows"])
        w.writerows(records)
    print(f"Wrote {UP / 'provenance.csv'}")
    print(f"DONE: {len(records)} files ready in {DATA}")


if __name__ == "__main__":
    main()
