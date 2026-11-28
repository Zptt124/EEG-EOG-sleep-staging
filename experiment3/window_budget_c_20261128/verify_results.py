"""Verify the published Experiment C plans, scores and paired intervals."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from fit_compare import ARMS, bootstrap, macro, require
from download import MANIFEST, hash_file, read_rows

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OLD_RESULTS = HERE.parent / "week3_budget_replication_20261128/results"


def verify_plan(protocol):
    for name, digest in protocol["files"].items():
        require(hash_file(RESULTS / name) == digest, f"Changed locked file: {name}")
    require(hash_file(MANIFEST) == protocol["manifest_sha256"], "Changed source manifest")
    require(hash_file(OLD_RESULTS / "plan.json") == protocol["original_plan_sha256"],
            "Changed historical protocol")
    source = {name: digest for digest, name in
              (line.split() for line in (RESULTS / "original_runner.sha256")
               .read_text(encoding="utf-8-sig").splitlines())}
    require(all(source[name] == digest for name, digest in protocol["source_sha256"].items()),
            "Original planning source receipt differs")
    old = read_rows(OLD_RESULTS / "blocks.csv")
    require(read_rows(RESULTS / "anchor_blocks.csv") == [r for r in old if r["arm"] == "base"],
            "Historical anchors differ")

    fields = ("roster", "arm", "subject_id", "record_id", "night_third", "first_epoch", "last_epoch")
    candidates = {tuple(r[k] for k in fields) for r in read_rows(RESULTS / "candidate_blocks.csv")
                  if r["paired_roster"] == "1"}
    selected = read_rows(RESULTS / "selected_blocks.csv")
    require(all(tuple(r[k] for k in fields) in candidates for r in selected),
            "Selected block absent from paired candidate pool")
    quotas = Counter((int(r["roster"]), int(r["draw"]), r["arm"]) for r in selected)
    require(quotas == {(r, d, a): 10 for r in protocol["paired_rosters"]
                       for d in range(1, 6) for a in ARMS}, "Selected block quotas differ")


def main():
    protocol = json.loads((RESULTS / "protocol.json").read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    require(summary["protocol_sha256"] == hash_file(RESULTS / "protocol.json"), "Protocol hash mismatch")
    verify_plan(protocol)
    with np.load(RESULTS / "paired_confusions.npz", allow_pickle=False) as paired:
        counts, rosters, good = (paired[k] for k in ("counts", "rosters", "exact_600"))
        require(paired["arms"].tolist() == list(ARMS)
                and paired["windows"].tolist() == ["old", "lights_off"]
                and rosters.tolist() == protocol["paired_rosters"]
                and len(paired["evaluation_subjects"]) == summary["evaluation_people"],
                "Wrong confusion archive axes")
    require(counts.shape == (len(rosters), 5, 2, 3, 16, 25)
            and np.all(counts.sum(axis=(-2, -1)) == summary["evaluation_epochs"]),
            "Wrong evaluation size")
    with np.load(OLD_RESULTS / "confusion_counts.npz", allow_pickle=False) as historical:
        require(np.array_equal(counts[:, :, 0], historical["counts"][0, rosters - 1]),
                "Archived historical arm counts differ")

    scores = read_rows(RESULTS / "scores.csv")
    require(len(scores) == summary["fits"] == len(rosters) * 5 * len(ARMS), "Wrong fit count")
    scores_by_key = {(int(s["roster"]), int(s["draw"]), s["arm"]): s for s in scores}
    deltas = read_rows(RESULTS / "paired_deltas.csv")
    delta_by_key = {(int(row["roster"]), int(row["draw"])): row for row in deltas}
    require(len(delta_by_key) == len(deltas) == len(rosters) * 5, "Wrong paired delta count")
    f1 = macro(counts.sum(axis=-2))
    for i, roster in enumerate(rosters):
        for d in range(5):
            arms = [scores_by_key[roster, d + 1, arm] for arm in ARMS]
            require(all(int(s["attempted_additions"]) == 600 and 0 <= int(s["scorable_additions"]) <= 600
                        for s in arms), "Invalid added-label count")
            exact = all(int(s["scorable_additions"]) == 600 for s in arms)
            delta = delta_by_key[roster, d + 1]
            require(exact == good[i, d] == bool(int(delta["exact_600_all_arms"])),
                    "Exact-budget membership differs")
            expected = [[float(s[k]) for s in arms] for k in ("old_macro_f1", "lights_off_macro_f1")]
            np.testing.assert_allclose(f1[i, d], expected, atol=1e-10, rtol=0)
            old, new = f1[i, d, :, 2] - f1[i, d, :, 0]
            np.testing.assert_allclose([old, new, new - old],
                                       [float(delta[k]) for k in ("old_delta_pp", "lights_off_delta_pp",
                                                                 "difference_of_differences_pp")],
                                       atol=1e-10, rtol=0)
    boot = (protocol["bootstrap_draws"], protocol["bootstrap_seed"])
    for name, mask in (("exact_600", good), ("fixed_600_attempts", np.ones_like(good))):
        measured = bootstrap(counts, mask, *boot)
        expected = summary[name]
        for key in ("rosters", "draw_pairs"):
            require(measured[key] == expected[key], f"Wrong {name} {key}")
        for key in expected.keys() - {"rosters", "draw_pairs"}:
            np.testing.assert_allclose(measured[key]["mean_pp"], expected[key]["mean_pp"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(measured[key]["interval_95_pp"], expected[key]["interval_95_pp"],
                                       atol=1e-10, rtol=0)
    print("Verified the locked plan, 390 fits, 130 draw pairs and both archived paired analyses.")


if __name__ == "__main__":
    main()
