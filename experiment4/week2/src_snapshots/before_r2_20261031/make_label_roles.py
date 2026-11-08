#make_label_roles.py —— 第二周 实验④ 阶段一：生成有标签/无标签名单 + 数据核对
#做三件事：
#  1. 核对 EEG、EOG 两份 feature_index.csv：sample_id 唯一、标签合法、两边逐段一致
#  2. 核对受试者：训练 54 人、验证 8 人、没有测试集；名单人数、嵌套关系、片段数与 B 的名单一致
#  3. 生成 labeled_unlabeled_subjects.csv（每人在 10%/25% 预算下的角色）和 budget_counts.csv
#     无标签者只统计片段数，不统计类别分布（统计类别就等于偷看了他们的标签）
#--self-test：用假数据故意制造错误，确认每项检查都会报错
#用法（Ubuntu）：python -u make_label_roles.py --subsets 名单.csv --split split_subjects.csv
#              --eeg-index EEG的feature_index.csv --eog-index EOG的feature_index.csv --output 输出文件夹

import argparse
import copy
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from label_guard import LabelLeakError, assert_hidden, hide_labels

STAGES = ["W", "N1", "N2", "N3", "REM"]
N_TRAIN, N_VAL = 54, 8
BUDGET_SIZES = {"label10": 5, "label25": 14, "label50": 27, "label100": 54}
#B 预设的主名单（sampling_seed 2024）10% 统计，第二周文档第 6 页
EXPECTED_LABEL10_2024 = {"n_epochs": 11936, "n_N1": 1719}


class CheckError(Exception):
    #任何一项检查不通过就抛出；正式运行时在 main 里转成 sys.exit(1)
    pass


def fail(msg):
    raise CheckError(msg)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#---------------- 检查 1：单份索引自身是否完好 ----------------
def check_index(rows, name):
    ids = [r["sample_id"] for r in rows]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        fail(f"{name}: {len(dup)} duplicated sample_id, e.g. {dup[:3]}")
    for r in rows:
        if r["y_true"] not in {"0", "1", "2", "3", "4"}:
            fail(f"{name}: invalid y_true {r['y_true']!r} at {r['sample_id']}")
        if r["split"] not in {"train", "val"}:
            fail(f"{name}: unexpected split {r['split']!r} at {r['sample_id']} (test must not be extracted)")
        #sample_id 必须等于 <record_id>_<epoch_index 四位>
        if r["sample_id"] != f"{r['record_id']}_{int(r['epoch_index']):04d}":
            fail(f"{name}: sample_id {r['sample_id']} does not match record_id/epoch_index")
        if not r["record_id"].startswith(r["subject_id"]):
            fail(f"{name}: record {r['record_id']} does not belong to subject {r['subject_id']}")
    #同一个 npy 文件里 row_index 不能重复
    pos = Counter((r["feature_path"], r["row_index"]) for r in rows)
    clash = [k for k, v in pos.items() if v > 1]
    if clash:
        fail(f"{name}: {len(clash)} (feature_path,row_index) pairs used twice, e.g. {clash[:2]}")


#---------------- 检查 2：EEG 与 EOG 逐段一致 ----------------
def check_pair(rows_eeg, rows_eog):
    a = {r["sample_id"]: r for r in rows_eeg}
    b = {r["sample_id"]: r for r in rows_eog}
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    if only_a or only_b:
        fail(f"sample_id sets differ: {len(only_a)} only in EEG, {len(only_b)} only in EOG, "
             f"e.g. {sorted(only_a)[:2]} {sorted(only_b)[:2]}")
    for sid, ra in a.items():
        rb = b[sid]
        for key in ("subject_id", "record_id", "epoch_index", "split", "y_true"):
            if ra[key] != rb[key]:
                fail(f"{key} mismatch at {sid}: EEG={ra[key]} EOG={rb[key]}")
    same_order = [r["sample_id"] for r in rows_eeg] == [r["sample_id"] for r in rows_eog]
    return same_order


#---------------- 检查 3：受试者划分与 split_subjects.csv 一致 ----------------
def check_split(rows_index, split_rows):
    split_of = {r["subject_id"]: r["split"] for r in split_rows}
    records_of = {r["subject_id"]: set(r["record_ids"].split(";")) for r in split_rows}
    idx_subjects = defaultdict(set)
    idx_records = defaultdict(set)
    for r in rows_index:
        idx_subjects[r["split"]].add(r["subject_id"])
        idx_records[r["subject_id"]].add(r["record_id"])
    train, val = idx_subjects["train"], idx_subjects["val"]
    if len(train) != N_TRAIN or len(val) != N_VAL:
        fail(f"index has {len(train)} train / {len(val)} val subjects, expected {N_TRAIN}/{N_VAL}")
    for s in train | val:
        if s not in split_of:
            fail(f"subject {s} in index but not in split_subjects.csv")
        expected = "train" if s in train else "val"
        if split_of[s] != expected:
            fail(f"subject {s} is {expected} in index but {split_of[s]} in split_subjects.csv")
        if idx_records[s] != records_of[s]:
            fail(f"subject {s}: records in index {sorted(idx_records[s])} != split {sorted(records_of[s])}")
    test = {s for s, sp in split_of.items() if sp == "test"}
    if test & (train | val):
        fail(f"test subjects found in index: {sorted(test & (train | val))}")
    return train, val, test


#---------------- 检查 4：B 的标签名单 ----------------
def check_subsets(sub_rows, train, epochs_per_subject):
    seeds = {r["sampling_seed"] for r in sub_rows}
    if len(seeds) != 1:
        fail(f"subset file has several sampling_seed values: {seeds}")
    sseed = seeds.pop()
    subjects = [r["subject_id"] for r in sub_rows]
    if len(subjects) != len(set(subjects)):
        fail("subset file has duplicated subject_id")
    if set(subjects) != train:
        fail(f"subset subjects != index train subjects (diff {sorted(set(subjects) ^ train)[:5]})")
    ranks = sorted(int(r["sampling_rank"]) for r in sub_rows)
    if ranks != list(range(1, N_TRAIN + 1)):
        fail("sampling_rank is not exactly 1..54")

    #label10：v2 有该列就直接用；v1 没有就按 sampling_rank 前 5 名推出（临时，只用于搭流程）
    has_label10 = "label10" in sub_rows[0]
    source = "B_v2_column" if has_label10 else "derived_rank_le_5_PROVISIONAL"
    members = {}
    for budget in BUDGET_SIZES:
        if budget == "label10" and not has_label10:
            members[budget] = {r["subject_id"] for r in sub_rows if int(r["sampling_rank"]) <= 5}
        else:
            members[budget] = {r["subject_id"] for r in sub_rows if r[budget] == "1"}
        if len(members[budget]) != BUDGET_SIZES[budget]:
            fail(f"{budget}: {len(members[budget])} subjects, expected {BUDGET_SIZES[budget]}")
    #嵌套：10 ⊂ 25 ⊂ 50 ⊂ 100
    order = ["label10", "label25", "label50", "label100"]
    for small, big in zip(order, order[1:]):
        if not members[small] <= members[big]:
            fail(f"{small} is not contained in {big}: {sorted(members[small] - members[big])}")
    #每个人前 k 名必须正好对应名单（B 规定按 sampling_rank 取前 k）
    rank_of = {r["subject_id"]: int(r["sampling_rank"]) for r in sub_rows}
    for budget, k in BUDGET_SIZES.items():
        if {s for s, rk in rank_of.items() if rk <= k} != members[budget]:
            fail(f"{budget} is not the top-{k} of sampling_rank")
    #B 名单里的 n_epochs 必须等于特征索引里该人的片段数
    for r in sub_rows:
        if int(r["n_epochs"]) != epochs_per_subject[r["subject_id"]]:
            fail(f"{r['subject_id']}: n_epochs {r['n_epochs']} in subset != {epochs_per_subject[r['subject_id']]} in index")
    return sseed, members, rank_of, source


#---------------- 检查 5：标签隐藏能否生效 ----------------
def check_hiding(rows_index, labeled, unlabeled):
    y = np.array([int(r["y_true"]) for r in rows_index])
    subject = np.array([r["subject_id"] for r in rows_index])
    split = np.array([r["split"] for r in rows_index])
    try:
        y_train, is_lab, is_unl = hide_labels(y, subject, split, labeled, unlabeled)
        assert_hidden(y_train, is_unl, is_lab)
    except LabelLeakError as e:
        fail(f"label hiding failed: {e}")
    #验证集样本绝不能被当成有标签或无标签训练样本
    if (is_lab & (split == "val")).any() or (is_unl & (split == "val")).any():
        fail("validation samples entered the training pool")
    return int(is_lab.sum()), int(is_unl.sum())


def class_counts(rows):
    c = Counter(int(r["y_true"]) for r in rows)
    return {f"n_{STAGES[k]}": c.get(k, 0) for k in range(5)}


def run_checks(rows_eeg, rows_eog, split_rows, sub_rows, verbose=True):
    #把全部检查串起来；正式运行和自测共用同一套
    check_index(rows_eeg, "EEG index")
    check_index(rows_eog, "EOG index")
    same_order = check_pair(rows_eeg, rows_eog)
    train, val, test = check_split(rows_eeg, split_rows)
    epochs_per_subject = Counter(r["subject_id"] for r in rows_eeg)
    sseed, members, rank_of, source = check_subsets(sub_rows, train, epochs_per_subject)
    hiding = {}
    for budget in ("label10", "label25"):
        hiding[budget] = check_hiding(rows_eeg, members[budget], train - members[budget])
    if verbose:
        print(f"OK: EEG/EOG indexes valid, {len(rows_eeg)} samples each, same row order: {same_order}")
        print(f"OK: subjects train={len(train)} val={len(val)}, test subjects not in index ({len(test)} in split)")
        print(f"OK: subset sampling_seed={sseed}, budgets nested, n_epochs match index, label10 source={source}")
        for b, (n_lab, n_unl) in hiding.items():
            print(f"OK: {b} label hiding works: {n_lab} labeled train samples, {n_unl} unlabeled set to -1")
    return dict(train=train, val=val, test=test, sseed=sseed, members=members, rank_of=rank_of,
                source=source, epochs_per_subject=epochs_per_subject, same_order=same_order)


#---------------- 自测：故意破坏数据，确认每项检查都能抓到 ----------------
def make_fake():
    #造一份迷你数据：54 名训练者 + 8 名验证者 + 2 名测试者，每人 1 晚 6 段
    split_rows, rows, sub_rows = [], [], []
    rng = np.random.default_rng(0)
    for i in range(N_TRAIN + N_VAL + 2):
        s = f"SC{500 + i}"
        rec = f"{s}1"
        sp = "train" if i < N_TRAIN else ("val" if i < N_TRAIN + N_VAL else "test")
        split_rows.append({"subject_id": s, "record_ids": rec, "split": sp, "seed": "42"})
        if sp == "test":
            continue
        for k in range(6):
            rows.append({"sample_id": f"{rec}_{k:04d}", "subject_id": s, "record_id": rec,
                         "epoch_index": str(k), "split": sp, "y_true": str(int(rng.integers(0, 5))),
                         "feature_path": f"{rec}.npy", "row_index": str(k), "config_id": "fake"})
    for i in range(N_TRAIN):
        rank = i + 1
        sub_rows.append({"sampling_seed": "2024", "subject_id": f"SC{500 + i}", "record_ids": f"SC{500 + i}1",
                         "n_records": "1", "n_epochs": "6", "sampling_rank": str(rank),
                         "label25": "1" if rank <= 14 else "0", "label50": "1" if rank <= 27 else "0",
                         "label100": "1"})
    return rows, copy.deepcopy(rows), split_rows, sub_rows


def self_test():
    base = make_fake()
    run_checks(*copy.deepcopy(base), verbose=False)
    print("SELF-TEST: clean fake data passes")

    def corrupt_dup(e, o, sp, sb):
        e.append(dict(e[0]))
    def corrupt_missing(e, o, sp, sb):
        o.pop(3)
    def corrupt_label_mismatch(e, o, sp, sb):
        o[5]["y_true"] = str((int(o[5]["y_true"]) + 1) % 5)
    def corrupt_bad_label(e, o, sp, sb):
        e[2]["y_true"] = "7"
        o[2]["y_true"] = "7"
    def corrupt_test_in_index(e, o, sp, sb):
        for r in (e, o):
            r[0]["split"] = "test"
    def corrupt_subject_split(e, o, sp, sb):
        sp[0]["split"] = "val"
    def corrupt_not_nested(e, o, sp, sb):
        sb[20]["label25"] = "1"
        sb[0]["label25"] = "0"
    def corrupt_epoch_count(e, o, sp, sb):
        sb[0]["n_epochs"] = "7"
    def corrupt_two_seeds(e, o, sp, sb):
        sb[1]["sampling_seed"] = "2025"
    def corrupt_row_clash(e, o, sp, sb):
        e[1]["row_index"] = e[0]["row_index"]

    cases = [corrupt_dup, corrupt_missing, corrupt_label_mismatch, corrupt_bad_label, corrupt_test_in_index,
             corrupt_subject_split, corrupt_not_nested, corrupt_epoch_count, corrupt_two_seeds, corrupt_row_clash]
    for fn in cases:
        data = copy.deepcopy(base)
        fn(*data)
        try:
            run_checks(*data, verbose=False)
        except CheckError as e:
            print(f"SELF-TEST: {fn.__name__:28s} caught -> {e}")
            continue
        print(f"SELF-TEST FAILED: {fn.__name__} was NOT caught")
        sys.exit(1)

    #标签隐藏本身：漏置 -1 必须被 assert_hidden 抓到
    y = np.array([0, 1, 2, 3])
    try:
        assert_hidden(y, np.array([False, False, True, True]))
    except LabelLeakError as e:
        print(f"SELF-TEST: {'leaked_true_label':28s} caught -> {e}")
    else:
        print("SELF-TEST FAILED: leaked label was NOT caught")
        sys.exit(1)
    print("SELF-TEST: all corruption cases caught")


#---------------- 写输出 ----------------
def write_outputs(out, info, rows_index, args):
    out.mkdir(parents=True, exist_ok=True)
    roles_path = out / "labeled_unlabeled_subjects.csv"
    if roles_path.exists():
        print(f"ERROR: {roles_path} already exists, choose a new output folder")
        sys.exit(1)

    split_of = {s: "train" for s in info["train"]} | {s: "val" for s in info["val"]} | {s: "test" for s in info["test"]}
    role_rows = []
    for s in sorted(split_of):
        row = {"subject_id": s, "split": split_of[s], "sampling_seed": info["sseed"],
               "sampling_rank": info["rank_of"].get(s, ""), "n_epochs": info["epochs_per_subject"].get(s, 0)}
        for budget in ("label10", "label25"):
            if split_of[s] == "train":
                row[f"role_{budget}"] = "labeled" if s in info["members"][budget] else "unlabeled"
            else:
                row[f"role_{budget}"] = split_of[s]  #验证/测试者不进伪标签池
        row["label10_source"] = info["source"]
        role_rows.append(row)
    with roles_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(role_rows[0].keys()))
        w.writeheader()
        w.writerows(role_rows)

    #片段统计：有标签者和验证集给类别数；无标签者只给片段数
    count_rows = []
    by_subject = defaultdict(list)
    for r in rows_index:
        by_subject[r["subject_id"]].append(r)
    for budget in ("label10", "label25"):
        lab = info["members"][budget]
        unl = info["train"] - lab
        lab_rows = [r for s in lab for r in by_subject[s]]
        count_rows.append({"budget": budget, "role": "labeled", "n_subjects": len(lab), "n_epochs": len(lab_rows),
                           **class_counts(lab_rows)})
        count_rows.append({"budget": budget, "role": "unlabeled", "n_subjects": len(unl),
                           "n_epochs": sum(len(by_subject[s]) for s in unl),
                           **{f"n_{st}": "hidden" for st in STAGES}})
    val_rows = [r for s in info["val"] for r in by_subject[s]]
    count_rows.append({"budget": "-", "role": "val", "n_subjects": len(info["val"]), "n_epochs": len(val_rows),
                       **class_counts(val_rows)})
    with (out / "budget_counts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(count_rows[0].keys()))
        w.writeheader()
        w.writerows(count_rows)

    #记录输入文件的 SHA，方便以后核对用的是哪一版
    with (out / "input_files.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "path", "sha256"])
        for role, p in [("subsets", args.subsets), ("split", args.split),
                        ("eeg_index", args.eeg_index), ("eog_index", args.eog_index)]:
            w.writerow([role, str(p), sha256(p)])

    print("\n=== budget counts ===")
    for r in count_rows:
        print(", ".join(f"{k}={v}" for k, v in r.items()))
    print(f"\nSaved: {roles_path}")
    print(f"Saved: {out / 'budget_counts.csv'}")
    print(f"Saved: {out / 'input_files.csv'}")
    return count_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", type=Path)
    ap.add_argument("--split", type=Path)
    ap.add_argument("--eeg-index", type=Path)
    ap.add_argument("--eog-index", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    for name in ("subsets", "split", "eeg_index", "eog_index", "output"):
        if getattr(args, name) is None:
            print(f"ERROR: --{name.replace('_', '-')} is required")
            sys.exit(1)

    rows_eeg = read_csv(args.eeg_index)
    rows_eog = read_csv(args.eog_index)
    split_rows = read_csv(args.split)
    sub_rows = read_csv(args.subsets)
    print(f"Read {len(rows_eeg)} EEG rows, {len(rows_eog)} EOG rows, {len(split_rows)} split rows, {len(sub_rows)} subset rows")

    try:
        info = run_checks(rows_eeg, rows_eog, split_rows, sub_rows)
    except CheckError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    count_rows = write_outputs(args.output, info, rows_eeg, args)

    #B 预设的 10% 主名单统计（只对 sampling_seed 2024 有意义）
    if info["sseed"] == "2024":
        lab10 = next(r for r in count_rows if r["budget"] == "label10" and r["role"] == "labeled")
        got = {"n_epochs": lab10["n_epochs"], "n_N1": lab10["n_N1"]}
        if got != EXPECTED_LABEL10_2024:
            print(f"ERROR: label10 counts {got} != expected {EXPECTED_LABEL10_2024} from the week-2 document")
            sys.exit(1)
        print(f"OK: label10 counts match the week-2 document {EXPECTED_LABEL10_2024}")
    if info["source"].endswith("PROVISIONAL"):
        print("NOTE: label10 was derived from sampling_rank (v1 file). Re-run with B's label_subsets_v2.csv before formal training.")


if __name__ == "__main__":
    main()
