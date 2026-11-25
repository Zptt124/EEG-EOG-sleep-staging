"""Audit and compare C's four W4-backbone/linear-readout cells on aligned development predictions."""

import json

import numpy as np
import pandas as pd
import yaml

from run_week5_c_late_layers import (
    BUDGETS, IDS, INDEX_SHA, OUTPUTS, SEEDS, SPECS, load_evaluator, run_path, sha,
)


OUT = OUTPUTS / "W5_C_30k_fourcell_v01"
ARMS = [arm for spec in SPECS.values() for arm in spec.arms]
KEYS = ["macro_f1", "n1_f1", "n1_ap", "n1_f1_adjusted"]
CONTRASTS = {
    "plain_last3_minus_all12": (-1, 1, 0, 0),
    "selfcond_last3_minus_all12": (0, 0, -1, 1),
    "selfcond_minus_plain_all12": (-1, 0, 1, 0),
    "selfcond_minus_plain_last3": (0, -1, 0, 1),
    "selfcond_by_layer_subset_interaction": (1, -1, -1, 1),
}


def check_sources():
    receipts = {}
    for mode, spec in SPECS.items():
        features = spec.out / "features"
        receipt = json.loads((features / "receipt.json").read_text())
        if not (receipt["source_sha256"] == spec.archive_sha
                and receipt["checkpoint_sha256"] == spec.checkpoint_sha
                and receipt["source_index_sha256"] == sha(features / "index.csv") == INDEX_SHA):
            raise ValueError(f"Wrong source receipt: {mode}")
        verification = json.loads((spec.out / "analysis/verification.json").read_text())
        if verification["runs"] != 12 or verification["test_used"]:
            raise ValueError(f"Incomplete or out-of-scope probes: {mode}")
        receipts[mode] = receipt
    for key in ("protocol_file_sha256", "preprocessing", "counts"):
        if receipts["plain"][key] != receipts["selfcond"][key]:
            raise ValueError(f"Backbones do not share {key}")
    return receipts


def load_budget(evaluator, budget):
    metrics, raw, adjusted = [], {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
    for seed in SEEDS:
        identities, folds = [], []
        for mode, spec in SPECS.items():
            for arm, (first, last) in spec.arms.items():
                folder = run_path(spec, arm, budget, seed)
                config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
                feature = config["feature_config"]
                if (feature["checkpoint_sha256"] != spec.checkpoint_sha or feature["backbone_seed"] != 0
                        or feature["layers"] != list(range(first, last + 1)) or config["test_used"]
                        or config["group"] != "C_W5" or config["downstream_seed"] != seed
                        or config["nominal_budget"] != budget):
                    raise ValueError(f"Unexpected fit configuration: {folder.name}")
                path = folder / "predictions_val.csv"
                frame, fingerprint = evaluator.load(path)
                result = evaluator.run_metrics(dict(run_id=folder.name, mode=mode, arm=arm, budget=budget,
                                                    seed=seed, predictions=str(path)), frame, fingerprint)
                recorded = json.loads((folder / "metrics.json").read_text())
                alpha = recorded["n1_adjust"]["alpha"]
                adj = pd.read_csv(folder / "predictions_val_n1adj.csv")
                probabilities = frame[evaluator.ev.PROB_COLS].to_numpy(copy=True)
                probabilities[:, 1] *= alpha
                if (not adj[IDS].equals(frame[IDS])
                        or not np.array_equal(adj["y_pred"].to_numpy(), probabilities.argmax(1))):
                    raise ValueError(f"OOF threshold adjustment does not replay: {folder.name}")
                # The adjusted CSV retains original probabilities and changes only the decisions.
                adj_metrics = evaluator.ev.compute_metrics(adj["y_true"], adj["y_pred"])
                for key in ("macro_f1", "n1_ap", "n1_f1"):
                    if not np.isclose(result[key], recorded[key], rtol=0, atol=1e-12):
                        raise ValueError(f"Prediction score does not replay: {folder.name} {key}")
                result.update(macro_f1_adjusted=adj_metrics["macro_f1"], n1_f1_adjusted=adj_metrics["n1_f1"],
                              alpha=alpha, selected_epoch=recorded["selected_epoch"])
                metrics.append(result)
                raw[arm].append(frame)
                adjusted[arm].append(adj)
                identities.extend(item[IDS].sort_values("sample_id").reset_index(drop=True) for item in (frame, adj))
                oof = pd.read_csv(folder / "oof_train.csv")
                folds.append(oof[IDS + ["fold"]].sort_values("sample_id").reset_index(drop=True))
        if len(identities[0]) != 21186 or any(not df.equals(identities[0]) for df in identities[1:]):
            raise ValueError(f"Dev identities differ across four cells: {budget} seed={seed}")
        if any(not df.equals(folds[0]) for df in folds[1:]):
            raise ValueError(f"OOF labels or folds differ across four cells: {budget} seed={seed}")
    return pd.DataFrame(metrics), raw, adjusted


def paired_contrasts(evaluator, budget, metrics, raw, adjusted):
    subjects = sorted(raw[ARMS[0]][0]["subject_id"].unique())
    if len(subjects) != 8:
        raise ValueError("Expected eight development subjects")
    raw_stats = [evaluator.per_subject(raw[arm]) for arm in ARMS]
    adj_stats = [evaluator.per_subject(adjusted[arm])["cms"] for arm in ARMS]

    def scores(selected):
        result = []
        for arm_raw, arm_adj in zip(raw_stats, adj_stats):
            row = evaluator.pooled_stats(arm_raw, selected)
            cm = sum(arm_adj[subject] for subject in selected)
            row["n1_f1_adjusted"] = evaluator.ev.metrics_from_confusion(cm)["n1_f1"]
            result.append([row[key] for key in KEYS])
        return np.array(result)

    point = scores(subjects)
    draws = np.random.default_rng(0).integers(0, len(subjects), size=(evaluator.N_BOOT, len(subjects)))
    boot = np.array([scores([subjects[i] for i in draw]) for draw in draws])
    seed_rows, contrasts = [], []
    for name, weights in CONTRASTS.items():
        effect = np.array(weights) @ point
        replicates = np.array(weights) @ boot
        bounds = np.percentile(replicates, [2.5, 97.5], axis=0)
        for i, key in enumerate(KEYS):
            contrasts.append(dict(budget=budget, contrast=name, metric=key, delta=effect[i],
                                  lo=bounds[0, i], hi=bounds[1, i], n_subjects=len(subjects),
                                  n_boot=evaluator.N_BOOT))
        for seed in SEEDS:
            matched = metrics[metrics["seed"] == seed].set_index("arm").loc[ARMS, KEYS].to_numpy()
            seed_rows.append(dict(budget=budget, contrast=name, seed=seed,
                                  **dict(zip(KEYS, np.array(weights) @ matched))))
    return contrasts, seed_rows


def main():
    receipts = check_sources()
    evaluator = load_evaluator()
    OUT.mkdir(parents=True, exist_ok=True)
    tables, contrasts, seeds = [], [], []
    for budget in BUDGETS:
        metrics, raw, adjusted = load_budget(evaluator, budget)
        print(f"Verified 12 probe runs for {budget}; computing jointly paired intervals", flush=True)
        differences, seed_rows = paired_contrasts(evaluator, budget, metrics, raw, adjusted)
        tables.append(metrics)
        contrasts.extend(differences)
        seeds.extend(seed_rows)
    table = pd.concat(tables, ignore_index=True)
    compact = table.groupby(["budget", "arm"])[KEYS + ["macro_f1_adjusted"]].mean().reset_index()
    table.to_csv(OUT / "metrics_by_run.csv", index=False)
    compact.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "paired_contrasts.csv", index=False)
    pd.DataFrame(seeds).to_csv(OUT / "paired_by_seed.csv", index=False)
    costs = [pd.read_csv(spec.out / "analysis/readout_cost.csv").assign(mode=mode)
             for mode, spec in SPECS.items()]
    pd.concat(costs, ignore_index=True).to_csv(OUT / "readout_cost.csv", index=False)
    verification = dict(runs=len(table), reused_plain_runs=12, new_selfcond_runs=12, dev_rows_per_run=21186,
                        dev_subjects=8, head_seeds=SEEDS, backbone_seeds=[0], index_sha256=INDEX_SHA,
                        identities_aligned=True, oof_folds_aligned=True, test_used=False,
                        source_receipts={mode: receipt["source_sha256"] for mode, receipt in receipts.items()})
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(compact.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
