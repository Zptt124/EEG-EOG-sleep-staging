"""Fit the frozen other-night block masks with the unchanged week-one probe."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
WEEK1 = HERE.parent / "week1_budget_20261126"
sys.path.insert(0, str(WEEK1))
from protocol import KEY, SOURCES, read_csv, require, sha256
from train import BATCH, code_hashes, features, fit, labels, load_bundle, load_plan, masks, score
import numpy as np
import torch
from protocol import write_csv


def fitting_indices(plan, roster, rows, positions, old_base):
    keys = read_csv(plan/roster/"training_epochs.csv")
    ids = [positions[r["subject_id"],r["record_id"],int(r["epoch_index"])] for r in keys]
    actual = [{key: rows[i][key] for key in KEY} for i in ids]
    require(keys == actual and ids == sorted(set(ids)) and len(ids) == 1800, "Wrong training keys or row order")
    blocks = read_csv(plan/roster/"training_blocks.csv")
    from_blocks = [positions[b["subject_id"],b["record_id"],e] for b in blocks
                   for e in range(int(b["first_epoch"]), int(b["last_epoch"])+1)]
    require(ids == sorted(from_blocks) and set(old_base) <= set(ids), "Block keys or preserved starting epochs mismatch")
    return ids


def main(args):
    old_plan = load_plan(args.old_plan,args.data)
    plan = json.loads((args.plan/"manifest.json").read_text())
    require(plan["first_week_plan_sha256"]==sha256(args.old_plan/"manifest.json"), "Different original plan")
    require(plan["epoch_index_sha256"]==sha256(args.data/SOURCES["epoch"]), "Different epoch index")
    require(plan["epochs"]==old_plan["epochs"]==30 and plan["head_seed"]==old_plan["head_seed"]==0,
            "Wrong seed or fixed epochs")
    for name, digest in plan["locks"].items():
        require(sha256(args.plan/name)==digest, f"Changed matching file: {name}")
    rows, positions, arrays = load_bundle(args.bundle,args.feature_info_sha256,args.old_plan,args.data)
    torch.set_num_threads(4)
    for roster in args.roster:
        previous, scoring = masks(args.old_plan,roster,positions,rows)
        require((args.plan/roster/"evaluation_subjects.csv").read_bytes()==
                (args.old_plan/roster/"evaluation_subjects.csv").read_bytes(), "Different development populations")
        ids = fitting_indices(args.plan,roster,rows,positions,previous["base"])
        require(set(ids).isdisjoint(set(scoring["pool36"]) | set(scoring["dev8"])),
                "Development sample visited during fitting")
        dev_truth = {r["sample_id"]:int(r["y_true"]) for r in read_csv(args.data/SOURCES["dev"])}
        require(all(int(rows[i]["y_true"])==dev_truth[rows[i]["sample_id"]] for i in scoring["dev8"]), "Dev truth changed")
        output = args.output/roster/"other_night"
        output.parent.mkdir(parents=True,exist_ok=True)
        require(not output.exists(), f"Existing run: {output}")
        with tempfile.TemporaryDirectory(prefix=".other_night-",dir=output.parent) as staging:
            pending=Path(staging)/"run"
            pending.mkdir()
            model, mean, std, history = fit(features(arrays,ids),labels(rows,ids),seed=0,epochs=30)
            require(history[-1]["cumulative_updates"]==450, "Unexpected update count")
            write_csv(pending/"loss.csv", history)
            np.savez(pending/"normalization.npz",mean=mean,std=std)
            torch.save(model.state_dict(),pending/"head.pt")
            metrics = {kind:score(pending,kind,subset,rows,arrays,model,mean,std) for kind,subset in scoring.items()}
            cfg=dict(roster=roster,arm="other_night",feature_info_sha256=args.feature_info_sha256,
                     first_week_plan_sha256=plan["first_week_plan_sha256"],
                     match_plan_sha256=sha256(args.plan/"manifest.json"),
                     training_epochs_sha256=sha256(args.plan/roster/"training_epochs.csv"),
                     matched_blocks_sha256=sha256(args.plan/roster/"matched_blocks.csv"),
                     code=code_hashes(),entrypoint_sha256=sha256(Path(__file__)), seed=0, epochs=30, batch=BATCH,
                     drop_last=False,train_samples=len(ids),updates=450, development_scoring=list(scoring),
                     held_out_scored=False, weights_sha256=sha256(pending/"head.pt"),
                     normalization_sha256=sha256(pending/"normalization.npz"),torch=str(torch.__version__),
                     numpy=np.__version__,python=sys.version.split()[0])
            (pending/"config.json").write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
            (pending/"metrics_dev.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
            pending.rename(output)
        print(f"{roster}: 1800 epochs, 450 updates, pool36={metrics['pool36']:.6f}, dev8={metrics['dev8']:.6f}",flush=True)

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","old-plan","data","bundle","output"):
        cli.add_argument(f"--{name}",type=Path,required=True)
    cli.add_argument("--feature-info-sha256",required=True)
    cli.add_argument("--roster",nargs="+",choices=("01","02","03","04","06","07","08","09","10"),required=True)
    args=cli.parse_args()
    main(args)
