"""D：C/D 上下文实验（2026-11-29 修订版）实验二的选块锁定。

只读：条件 A 候选块表、起点块、混合配额、冻结 384 维特征、起点块的分期标签（U 的线性头需要）。
不读取任何候选块的分期标签或不可评分标记。

输出（--out）：
  D_blocks.csv               每块一行：roster, draw, strategy, record_id, first_epoch, last_epoch, block_id,
                             subject_id, night_third, sources
  composition.csv            逐名单、重复、策略的来源构成（原人原夜 / 原人另一夜 / 新人块数、人数、夜数）
  u_scores.csv / v_trace.csv U 块得分、V 每步增量（便于复核）
  lock.json                  种子、输入与代码 SHA-256、检查结果

策略：
  random / add_people  R / P：沿用 filter_exp/04_lock_selection.py 的共享排序（SEED 2026112903，
                       default_rng([SEED, 名单, 重复, 1])），第 1–5 次必须与 C 交接的表逐行一致，再生成第 6–10 次。
  coverage             V：候选池特征按本名单候选池均值/标准差标准化，余弦相似度；目标点 = 候选池去重 epoch
                       （>10,000 时按 default_rng([SEED_T, 名单]) 抽 10,000，10 次重复共用）；覆盖度 = 目标点到
                       “起点 1,200 段 + 已选块全部 epoch”的最大相似度的均值；每步在时段配额未满的块中按增量排序，
                       从前 3 名中按 default_rng([SEED_V, 名单, 重复]) 等概率取 1 块。
  coverage_det         V 确定性版：每步取第 1 名，每份名单只做 draw 1。
  disagreement         U：5 个线性头（head seed 0–4），各自按 default_rng([SEED_U, 名单, 头]) 对 20 个起点块
                       有放回抽 20 块（保留重数）训练；每段分歧 = 平均概率的熵 − 各头熵的均值；块得分 = 60 段均值；
                       每个时段按配额，从剩余块得分前 3 名中按 default_rng([SEED_US, 名单, 重复]) 等概率取 1 块。
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 2026112903      # R/P 共享排序（与第一轮相同）
SEED_T = 2026112921    # V 目标点抽样
SEED_V = 2026112922    # V 前 3 名抽取
SEED_U = 2026112923    # U 起点块 bootstrap
SEED_US = 2026112924   # U 前 3 名抽取
BLOCK = 60
DRAWS = range(1, 11)
TOP = 3
N_TARGET = 10000
HEADS = range(5)
HEAD_EPOCHS = 30


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def wr(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Features:
    """冻结 384 维特征：共享缓存 + 关灯补提 + 候选池补提（EEG 在前）。"""

    def __init__(self, cache, supplements):
        info = json.loads((cache / "features_info.json").read_text())
        if sha(cache / "index.csv") != info["index_sha256"]:
            fail("shared index hash mismatch")
        self.arrays = [np.load(cache / f"{m}_std.npy", mmap_mode="r") for m in ("eeg", "eog")]
        self.pos, self.y = {}, {}
        for i, r in enumerate(rd(cache / "index.csv")):
            k = (r["record_id"], int(r["epoch_index"]))
            self.pos[k] = i
            self.y[k] = int(r["y_true"])
        self.extra = {}
        for d in supplements:
            for f in sorted(glob.glob(str(d / "*.npz"))):
                z = np.load(f)
                rec = os.path.basename(f)[:6]
                for e, x in zip(z["missing_epoch"], z["x"]):
                    k = (rec, int(e))
                    if k in self.pos:
                        continue
                    if k in self.extra and not np.allclose(self.extra[k], x, atol=2e-4, rtol=5e-3):
                        fail(f"conflicting supplemental features at {k}")
                    self.extra[k] = np.asarray(x, np.float32)

    def get(self, keys):
        out = np.empty((len(keys), 384), np.float32)
        for j, k in enumerate(keys):
            if k in self.pos:
                i = self.pos[k]
                out[j] = np.concatenate([self.arrays[0][i], self.arrays[1][i]])
            elif k in self.extra:
                out[j] = self.extra[k]
            else:
                fail(f"missing frozen feature {k}")
        if not np.isfinite(out).all():
            fail("non-finite features")
        return out


def epochs_of(rec, first):
    return [(rec, e) for e in range(first, first + BLOCK)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True, help="filter_exp/c_handoff/candidate_blocks.csv")
    ap.add_argument("--anchor", type=Path, required=True)
    ap.add_argument("--quotas", type=Path, required=True)
    ap.add_argument("--rp15", type=Path, required=True, help="C 的 condition_A_RP_draws_1_5.csv")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--supplement", type=Path, action="append", default=[])
    ap.add_argument("--linear-src", type=Path, required=True, help="含 C 的 historical_head/train.py 的目录")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(a.linear_src))
    from train import fit as linear_fit

    anchors = defaultdict(list)
    for r in rd(a.anchor):
        anchors[int(r["roster"])].append((r["record_id"], int(r["first_epoch"])))
    rosters = sorted(anchors)
    if len(rosters) != 26 or any(len(v) != 20 for v in anchors.values()):
        fail("expect 26 rosters x 20 anchor blocks")
    quota = defaultdict(dict)
    for r in rd(a.quotas):
        quota[int(r["roster"])][int(r["night_third"])] = int(r["quota"])

    cand = rd(a.candidates)
    cells = defaultdict(lambda: defaultdict(list))
    info = {}
    for r in cand:
        ro = int(r["roster"])
        if ro not in anchors:
            continue
        cell = (r["arm"], r["subject_id"], r["record_id"], int(r["night_third"]))
        b = r["block_id"]
        cells[ro][cell].append(b)
        cur = info.setdefault((ro, b), dict(record_id=r["record_id"], subject_id=r["subject_id"],
                                            first=int(r["first_epoch"]), third=int(r["night_third"]),
                                            in_A=int(r["in_A"]), sources=set()))
        cur["sources"].add(r["arm"])

    feats = Features(a.cache, a.supplement)
    rows, comp, vtrace, uscore = [], [], [], []

    def add(ro, d, strat, chosen):
        thirds = defaultdict(int)
        for b in chosen:
            thirds[info[(ro, b)]["third"]] += 1
        if len(chosen) != 10 or len(set(chosen)) != 10 or dict(thirds) != quota[ro]:
            fail(f"roster {ro} draw {d} {strat}: quota or count violated")
        anc = {k for rec, f in anchors[ro] for k in epochs_of(rec, f)}
        for b in chosen:
            I = info[(ro, b)]
            if not I["in_A"] or set(epochs_of(I["record_id"], I["first"])) & anc:
                fail(f"roster {ro} {strat}: {b} not in A or overlaps anchor")
            rows.append(dict(roster=ro, draw=d, strategy=strat, record_id=I["record_id"], first_epoch=I["first"],
                             last_epoch=I["first"] + BLOCK - 1, block_id=b, subject_id=I["subject_id"],
                             night_third=I["third"], sources="+".join(sorted(I["sources"]))))
        src = defaultdict(int)
        for b in chosen:
            s = info[(ro, b)]["sources"]
            src["deepen" if "deepen" in s else "other_night" if "other_night" in s else "add_people"] += 1
        comp.append(dict(roster=ro, draw=d, strategy=strat, same_night_blocks=src["deepen"],
                         other_night_blocks=src["other_night"], new_person_blocks=src["add_people"],
                         subjects=len({info[(ro, b)]["subject_id"] for b in chosen}),
                         nights=len({info[(ro, b)]["record_id"] for b in chosen})))

    for ro in rosters:
        cs = cells[ro]
        keys = sorted({b for v in cs.values() for b in v})
        A = [b for b in keys if info[(ro, b)]["in_A"]]
        # ---- R / P：共享排序 ----
        for d in DRAWS:
            u1 = np.random.default_rng([SEED, ro, d, 1]).random(len(keys))
            p1 = {b: (float(u1[i]), b) for i, b in enumerate(keys)}
            ch = []
            for cell in sorted(k for k in cs if k[0] == "add_people"):
                opts = [b for b in cs[cell] if info[(ro, b)]["in_A"]]
                if not opts:
                    fail(f"roster {ro}: empty add_people cell {cell}")
                ch.append(min(opts, key=lambda b: p1[b]))
            add(ro, d, "add_people", ch)
            ch = []
            for th, q in sorted(quota[ro].items()):
                pool = sorted((b for b in A if info[(ro, b)]["third"] == th), key=lambda b: p1[b])
                ch += pool[:q]
            add(ro, d, "random", ch)

        # ---- 特征 ----
        pool_keys = sorted({k for b in A for k in epochs_of(info[(ro, b)]["record_id"], info[(ro, b)]["first"])})
        anc_keys = [k for rec, f in anchors[ro] for k in epochs_of(rec, f)]
        Xp = feats.get(pool_keys)
        mu, sd = Xp.mean(0), np.maximum(Xp.std(0), 1e-6)
        pidx = {k: i for i, k in enumerate(pool_keys)}

        def unit(x):
            z = (x - mu) / sd
            return z / np.linalg.norm(z, axis=1, keepdims=True)

        Zp = unit(Xp)
        tsel = np.arange(len(pool_keys))
        if len(tsel) > N_TARGET:
            tsel = np.sort(np.random.default_rng([SEED_T, ro]).choice(len(tsel), N_TARGET, replace=False))
        T = torch.from_numpy(Zp[tsel])
        base = (T @ torch.from_numpy(unit(feats.get(anc_keys))).T).max(1).values.numpy()
        M = np.empty((len(A), len(tsel)), np.float32)
        for j, b in enumerate(A):
            ids = [pidx[k] for k in epochs_of(info[(ro, b)]["record_id"], info[(ro, b)]["first"])]
            M[j] = (T @ torch.from_numpy(Zp[ids]).T).max(1).values.numpy()

        def greedy(rng, d, strat):
            cur, left, ch = base.copy(), dict(quota[ro]), []
            for step in range(10):
                legal = [j for j, b in enumerate(A) if b not in ch and left.get(info[(ro, b)]["third"], 0) > 0]
                gains = np.maximum(M[legal], cur).mean(1) - cur.mean()
                order = sorted(range(len(legal)), key=lambda i: (-gains[i], A[legal[i]]))
                top = order[:TOP]
                pick = top[0] if rng is None else top[int(rng.integers(len(top)))]
                j = legal[pick]
                ch.append(A[j])
                left[info[(ro, A[j])]["third"]] -= 1
                vtrace.append(dict(roster=ro, draw=d, strategy=strat, step=step + 1, block_id=A[j],
                                   gain=repr(float(gains[pick])), top_gain=repr(float(gains[order[0]])),
                                   coverage_after=repr(float(np.maximum(M[j], cur).mean()))))
                cur = np.maximum(cur, M[j])
            return ch

        for d in DRAWS:
            add(ro, d, "coverage", greedy(np.random.default_rng([SEED_V, ro, d]), d, "coverage"))
        add(ro, 1, "coverage_det", greedy(None, 1, "coverage_det"))

        # ---- U：bootstrap 线性头分歧 ----
        ay = np.array([feats.y[k] for k in anc_keys])
        if not np.isin(ay, range(5)).all():
            fail(f"roster {ro}: anchor contains unscorable epochs")
        Xa = feats.get(anc_keys)
        probs = []
        for h in HEADS:
            pick = np.random.default_rng([SEED_U, ro, h]).integers(20, size=20)
            ids = np.concatenate([np.arange(i * BLOCK, (i + 1) * BLOCK) for i in pick])
            model, m, s, _ = linear_fit(Xa[ids], ay[ids], h, HEAD_EPOCHS)
            with torch.inference_mode():
                probs.append(torch.softmax(model(torch.from_numpy((Xp - m) / s)), 1).numpy())
        P = np.stack(probs)
        ent = lambda q: -(q * np.log(np.clip(q, 1e-12, 1))).sum(-1)
        mi = ent(P.mean(0)) - ent(P).mean(0)
        score = {b: float(mi[[pidx[k] for k in epochs_of(info[(ro, b)]["record_id"], info[(ro, b)]["first"])]].mean())
                 for b in A}
        for b in A:
            uscore.append(dict(roster=ro, block_id=b, night_third=info[(ro, b)]["third"], score=repr(score[b])))
        for d in DRAWS:
            rng = np.random.default_rng([SEED_US, ro, d])
            ch = []
            for th, q in sorted(quota[ro].items()):
                rem = sorted((b for b in A if info[(ro, b)]["third"] == th), key=lambda b: (-score[b], b))
                for _ in range(q):
                    j = int(rng.integers(min(TOP, len(rem))))
                    ch.append(rem.pop(j))
            add(ro, d, "disagreement", ch)
        print(f"roster {ro:02d}: pool {len(A)} blocks / {len(pool_keys)} epochs, targets {len(tsel)}, "
              f"anchor coverage {base.mean():.4f}, mean U score {np.mean(list(score.values())):.4f}", flush=True)

    # ---- 第 1–5 次 R/P 必须与 C 交接表一致 ----
    mine = {(r["roster"], r["draw"], r["strategy"], r["block_id"]) for r in rows
            if r["strategy"] in ("random", "add_people") and r["draw"] <= 5}
    theirs = {(int(r["roster"]), int(r["draw"]), r["strategy"], r["block_id"]) for r in rd(a.rp15)}
    if mine != theirs:
        fail(f"R/P draws 1-5 differ from C handoff ({len(mine ^ theirs)} rows)")
    print("R/P draws 1-5 reproduce C handoff exactly", flush=True)

    rows.sort(key=lambda r: (r["roster"], r["strategy"], r["draw"], r["first_epoch"], r["record_id"]))
    wr(a.out / "D_blocks.csv", rows)
    wr(a.out / "composition.csv", comp)
    wr(a.out / "v_trace.csv", vtrace)
    wr(a.out / "u_scores.csv", uscore)
    lock = dict(
        purpose="C/D context experiment 2026-11-29, experiment 2 selections (label-blind for candidate blocks)",
        seeds=dict(rp_shared_order=SEED, v_targets=SEED_T, v_top3=SEED_V, u_bootstrap=SEED_U, u_top3=SEED_US),
        draws=list(DRAWS), top_k=TOP, n_target_max=N_TARGET, head_epochs=HEAD_EPOCHS, heads=list(HEADS),
        strategies=sorted({r["strategy"] for r in rows}), n_rows=len(rows),
        inputs={str(p.name): sha(p) for p in (a.candidates, a.anchor, a.quotas, a.rp15)},
        feature_info_sha256=sha(a.cache / "features_info.json"),
        code_sha256={"02_lock_selections.py": sha(Path(__file__)), "train.py": sha(a.linear_src / "train.py")},
        outputs={n: sha(a.out / n) for n in ("D_blocks.csv", "composition.csv", "v_trace.csv", "u_scores.csv")},
        checks=["10 blocks per unit", "night-third quotas", "condition A only", "no anchor overlap",
                "R/P draws 1-5 equal to C handoff", "candidate labels never read"])
    (a.out / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(f"LOCKED | D_blocks.csv sha256 {lock['outputs']['D_blocks.csv']}", flush=True)


if __name__ == "__main__":
    main()
