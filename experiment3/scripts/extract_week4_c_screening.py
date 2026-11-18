"""Cache paired W3 small-EEGDM features on the W4 C five-subject budget."""

import argparse
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch


ARCHIVE_SHA = "56a506d04f6243c395f9fb5b557c70761f0f247c374c70d40dfc706b994d6122"
BACKBONE_RUN = "W3_D_pretrain_m64x12_bs108_seed0_v01"
STEPS = {"random_init": 0, "pretrained": 30000}
POOL_SECONDS = (1, 2)


def extract_record(extractors, signal, batch_size, pool_seconds=POOL_SECONDS):
    output = {}
    for index, (modality, extractor) in enumerate(extractors.items()):
        batches = {seconds: [] for seconds in pool_seconds}
        for batch in torch.from_numpy(signal[:, index].copy()).split(batch_size):
            latent = extractor((batch[:, None].cuda(), None))[:, 0, :, 0]
            for seconds in pool_seconds:
                # Pool raw activations over 200 or 400 points, including across
                # 5-second extractor window boundaries at 2-second resolution.
                windows = latent.unflatten(2, (30 // seconds, 200 * seconds))
                reduced = windows.std(dim=3, correction=1)
                if not torch.isfinite(reduced).all():
                    raise ValueError("Nonfinite pooled features")
                batches[seconds].append(reduced.cpu().numpy())
        for seconds, parts in batches.items():
            output[f"{modality}_{seconds}s_std"] = np.concatenate(parts).astype(np.float16)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-checkout", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="extract eight real epochs per arm")
    parser.add_argument("--scope", choices=("training", "development"), default="training")
    parser.add_argument("--pool-seconds", nargs="+", type=int, choices=POOL_SECONDS,
                        default=list(POOL_SECONDS))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.d_checkout / "experiment4/week3/src"))
    import w3d_common as d

    if d.sha256_file(args.archive) != ARCHIVE_SHA:
        raise ValueError("Checkpoint archive SHA256 differs from D's release")
    d.VENDOR = root / "vendor/EEGDM"
    d.SPLIT_CSV = args.d_checkout / "manifests/split_subjects.csv"
    d.LABEL_SUBSETS_CSV = args.d_checkout / "manifests/label_subsets_v2.csv"
    d.add_vendor_to_path()
    from model.classifier import LatentActivityExtractor
    from model.diffusion_model import Wavenet

    (data_dir,) = (root / "vendor/experiment1-review/交付包").glob("*/2_*/data/processed")
    budget = {
        row["subject_id"] for row in d.read_csv(d.LABEL_SUBSETS_CSV)
        if row["sampling_seed"] == "2024" and row["label10"] == "1"
    }
    record_map = d.split_records()
    records = sorted((record, subject) for record, (subject, split) in record_map.items()
                     if split == "train" and subject in budget)
    if args.scope == "development":
        records = sorted((record, subject) for record, (subject, split) in record_map.items()
                         if split == "val")
        if len({subject for _, subject in records}) != 8:
            raise ValueError("Expected eight development subjects")
    elif len(budget) != 5 or {subject for _, subject in records} != budget:
        raise ValueError("Expected all five budget subjects in the train split")
    if args.smoke:
        records = records[:1]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if args.smoke else "screening"
    if args.scope == "development":
        mode = "development"
    plan = dict(mode=mode, archive_sha256=ARCHIVE_SHA, backbone_run=BACKBONE_RUN,
                checkpoint_steps=STEPS, split_sha256=d.SPLIT_SHA,
                label_subsets_sha256=d.LABEL_SUBSETS_SHA,
                budget="development" if args.scope == "development" else "label10",
                sampling_seed=2024, records=[list(pair) for pair in records],
                diffusion_t=1, query="gate", layers=12, hidden=64,
                pooling_seconds=list(args.pool_seconds), weights="ema_model",
                batch_size=args.batch_size, dev_used=args.scope == "development", test_used=False)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf8")) != plan:
        raise ValueError("Output directory already has a different extraction plan")
    d.write_json(plan_path, plan)
    torch.backends.cuda.matmul.allow_tf32 = True

    with ZipFile(args.archive) as source, torch.inference_mode():
        for arm, step in STEPS.items():
            arm_dir = output / arm
            arm_dir.mkdir(exist_ok=True)
            start = time.perf_counter()
            with source.open(f"{BACKBONE_RUN}/ckpt_step{step}.pt") as stream:
                checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
            if checkpoint["model_kwargs"] != d.model_kwargs("m64x12"):
                raise ValueError("Unexpected backbone architecture")
            model = Wavenet(**checkpoint["model_kwargs"])
            model.load_state_dict(checkpoint["ema_model"], strict=True)
            model.requires_grad_(False).eval().cuda()
            extractors = {
                modality: LatentActivityExtractor(model, diffusion_t=1, query=["gate"],
                                                  use_cond=[channel]).eval().cuda()
                for modality, channel in d.MODALITIES.items()
            }
            for index, (record, subject) in enumerate(records, 1):
                target = arm_dir / f"{record}.npz"
                if target.exists():
                    continue
                scope = slice(8) if args.smoke else slice(None)
                with np.load(data_dir / f"{record}.npz", allow_pickle=False) as data:
                    signal = np.stack(
                        [d.transform_uv(data["signal"][scope, ch]) for ch in d.MODALITIES.values()],
                        axis=1,
                    )
                    labels = data["labels"][scope]
                    epochs = data["epoch_index"][scope]
                values = extract_record(extractors, signal, args.batch_size, args.pool_seconds)
                partial = target.with_suffix(".partial")
                with partial.open("wb") as stream:
                    np.savez(stream, **values, y_true=labels, epoch_index=epochs, subject_id=subject)
                partial.replace(target)
                print(f"{arm} {index}/{len(records)} {record} n={len(labels)} "
                      f"elapsed={time.perf_counter() - start:.1f}s", flush=True)
            del extractors, model
    (output / "DONE").write_text("ok\n", encoding="utf8")


if __name__ == "__main__":
    main()
