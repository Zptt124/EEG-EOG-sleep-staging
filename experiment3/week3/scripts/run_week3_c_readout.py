"""Week 3 C: compare two readouts of the same frozen EEGDM features."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_s0_s1_eeg_comparison import (
    EVALUATOR, ROOT, SPLIT, SUBSETS, align_indices, audit_index, official_module, read_csv,
    require, selected_positions, sha, write_csv, write_json,
)

UPSTREAM = ROOT / "vendor/EEGDM"
sys.path.insert(0, str(UPSTREAM))
from model.classifier import LatentActivityDecoder, TransformerClassifier

SOURCES = {
    "EEG": ROOT / "features/eeg_cond14_v01",
    "EOG": ROOT / "features/eog_cond14_local_Dprotocol_v01",
}
CONFIG = ROOT / "configs/week3_c_readout.json"
MODALITIES = tuple(SOURCES)


class CachedLFT(nn.Module):
    """Reuse the upstream decoder/classifier after its extractor and reducer."""

    def __init__(self, n_layers, n_tokens, n_pool=30):
        super().__init__()
        self.decoder = LatentActivityDecoder(
            d_model=128, d_kv_embed_factor=1, n_layer=n_layers, n_query=1,
            n_pool=n_pool, pool_merge="share", multi_query_merge="seq",
            ch_order=list(MODALITIES), have_ch_pos_embed=True,
            cat_ch_pos_embed=False, ch_pos_emb_sym=None,
            clst_dim="TP", clst_pos_embed_dim="", n_clst=n_tokens,
            stack_struct="scf", num_heads=8, ff=4, have_crossnorm=False,
            across_pool_stack_struct=None,
        )
        self.classifier = TransformerClassifier(
            n_tower=1, n_pool=n_pool, n_clst=n_tokens, d_embed=128,
            have_pos_embed=True, pos_embed_dim="TPN", stack_struct="sfsfsfsf",
            num_heads=8, ff=4, final_act="pool", n_class=5,
        )

    @staticmethod
    def to_official_axes(x):
        # B,C,n,P,H -> B,n,T,P,Q,C,H (one tower and one gate query).
        return x.permute(0, 2, 3, 1, 4).unsqueeze(2).unsqueeze(4)

    def forward(self, x):
        return self.classifier(self.decoder(self.to_official_axes(x))[0])


class SegmentMeanLFT(CachedLFT):
    """Use one five-second LFT for all six segments and average their logits."""

    def __init__(self, n_layers, n_tokens):
        super().__init__(n_layers, n_tokens, n_pool=5)

    def forward(self, x):
        batch, modalities, layers, _, hidden = x.shape
        segments = x.reshape(batch, modalities, layers, 6, 5, hidden)
        segments = segments.permute(0, 3, 1, 2, 4, 5).flatten(0, 1)
        logits = super().forward(segments)
        return logits.reshape(batch, 6, 5).float().mean(dim=1)


class TemporalConvReadout(nn.Module):
    """Mix adjacent seconds before pooling, with a parameter count close to all20 LFT."""

    def __init__(self, n_layers, width, classifier_width):
        super().__init__()
        self.project = nn.Linear(2 * n_layers * 128, width)
        self.temporal = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, classifier_width), nn.GELU(),
            nn.Linear(classifier_width, 5),
        )

    def forward(self, x):
        seconds = x.permute(0, 3, 1, 2, 4).flatten(start_dim=2)
        features = F.gelu(self.project(seconds)).transpose(1, 2)
        features = features + F.gelu(self.temporal(features))
        return self.classifier(features.mean(dim=-1))


def build_heads(config, seed, device):
    torch.manual_seed(seed)
    n_layers = len(config["layers"])
    linear = nn.Sequential(nn.Flatten(), nn.Linear(2 * n_layers * 128, 5))
    if "temporal_conv" in config:
        temporal = TemporalConvReadout(n_layers, **config["temporal_conv"])
    else:
        lft_type = SegmentMeanLFT if config.get("lft_segment_seconds") == 5 else CachedLFT
        temporal = lft_type(n_layers, config["lft_tokens_per_second"])
    return {name: model.to(device) for name, model in (("C0", linear), ("C1", temporal))}


def logits_for(name, model, x):
    return model(x.mean(dim=3) if name == "C0" else x)


def optimizer_for(model, config):
    return torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )


def inspect_inputs(dev_reference):
    configs = {name: json.loads((path / "feature_config.json").read_text(encoding="utf-8"))
               for name, path in SOURCES.items()}
    protocol = dict(
        upstream_commit="003b5a2a016b2f02788ea5693d8b7cd00b6bd969",
        checkpoint_sha256="0fe2d385d074834f3c22d9458798da49b2e5a58bd817284f130aae8814f5c1ef",
        condition=14, diffusion_t=1, random_noise=False, query="gate",
        layers=list(range(20)), reduction="std, torch correction=1",
        axes=["sample", "layer", "time_window", "hidden"], feature_shape=[20, 30, 128],
        storage_dtype="float16", internal_segments=6, continuous_30s_context=False,
        target_fs=200, window_points=200, stride_points=200,
    )
    for modality, cfg in configs.items():
        require(cfg["modality"] == modality.lower(), f"Wrong modality in {modality} cache")
        require(all(cfg[k] == v for k, v in protocol.items()), f"{modality} protocol differs")
    require({k: v for k, v in configs["EEG"].items() if k != "modality"} ==
            {k: v for k, v in configs["EOG"].items() if k != "modality"},
            "EEG/EOG extraction settings differ")
    indices = {name: read_csv(path / "feature_index.csv") for name, path in SOURCES.items()}
    split = read_csv(SPLIT)
    grouped = audit_index(indices["EEG"], split)
    audit_index(indices["EOG"], split)
    align_indices(indices["EEG"], indices["EOG"])
    rows = indices["EEG"]
    subjects, train, dev = selected_positions(rows, read_csv(SUBSETS), "label25")
    keys = ("sample_id", "subject_id", "record_id", "epoch_index", "y_true")
    expected_dev = sorted(tuple(r[k] for k in keys) for r in read_csv(dev_reference))
    actual_dev = sorted(tuple(rows[i][k] for k in keys) for i in dev)
    require(actual_dev == expected_dev, "Development index differs from the shared reference")
    for record, items in grouped.items():
        for path in SOURCES.values():
            values = np.load(path / f"{record}.npy", mmap_mode="r", allow_pickle=False)
            require(values.shape == (len(items), 20, 30, 128) and values.dtype == np.float16,
                    f"Invalid feature header: {path.name}/{record}")
    audit = dict(
        samples=len(rows), records=len(grouped), split_counts=Counter(r["split"] for r in rows),
        split_subject_counts=Counter(r["split"] for r in split),
        labeled_subjects=sorted(subjects), labeled_epochs=len(train), dev_epochs=len(dev),
        feature_protocol=protocol, modality_order=MODALITIES, test_used=False,
        eog_provenance="local regeneration under prior D reference protocol",
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in [
            SPLIT, SUBSETS, EVALUATOR, UPSTREAM / "model/classifier.py", Path(__file__), CONFIG,
            *(folder / file for folder in SOURCES.values()
              for file in ("feature_index.csv", "feature_config.json")),
        ]},
        development_reference_sha256=sha(dev_reference),
    )
    return rows, train, dev, audit


class CachedEpochs:
    """Read selected rows in contiguous blocks from the two mmap caches."""

    def __init__(self, rows, positions, layers, sources=SOURCES):
        self.rows = [rows[i] for i in positions]
        self.layer_slice = slice(layers[0], layers[-1] + 1)
        self.n_layers = len(layers)
        counts = Counter(r["record_id"] for r in self.rows)
        self.maps = {
            record: [np.load(path / f"{record}.npy", mmap_mode="r", allow_pickle=False)
                     for path in sources.values()]
            for record in counts
        }
        self.blocks = []
        begin = 0
        for record, count in counts.items():
            require(all(r["record_id"] == record and int(r["row_index"]) == i
                        for i, r in enumerate(self.rows[begin:begin + count])),
                    f"Unexpected selected row order: {record}")
            self.blocks.append((record, begin, count))
            begin += count
        self.labels = np.array([int(r["y_true"]) for r in self.rows], dtype=np.int64)

    def groups(self, block_size, blocks_per_group, rng=None):
        blocks = [(record, base + offset, offset, min(count - offset, block_size))
                  for record, base, count in self.blocks for offset in range(0, count, block_size)]
        if rng is not None:
            rng.shuffle(blocks)
        for start in range(0, len(blocks), blocks_per_group):
            selected = blocks[start:start + blocks_per_group]
            x = np.empty((sum(b[3] for b in selected), 2, self.n_layers, 30, 128),
                         dtype=np.float16)
            positions = np.empty(len(x), dtype=np.int64)
            end = 0
            for record, base, offset, count in selected:
                for c, values in enumerate(self.maps[record]):
                    x[end:end + count, c] = values[offset:offset + count, self.layer_slice]
                positions[end:end + count] = np.arange(base, base + count)
                end += count
            if rng is not None:
                order = rng.permutation(len(x))
                x, positions = x[order], positions[order]
            yield x, positions


def smoke(config, rows, train, device, sources=SOURCES):
    subset = CachedEpochs(rows, train, config["layers"], sources)
    rng = np.random.default_rng(0)
    size = config["batch_size"]
    raw, positions = next(subset.groups(size, 2, rng))
    x = torch.as_tensor(raw[:size], device=device, dtype=torch.float32)
    y = torch.as_tensor(subset.labels[positions[:size]], device=device)
    require(torch.isfinite(x).all().item(), "Nonfinite cached features in smoke batch")
    tokens = CachedLFT.to_official_axes(x)
    for c in range(2):
        require(torch.equal(tokens[:, :, 0, :, 0, c], x[:, c]), "Cache axis mapping differs")
    out = dict(input_shape=list(x.shape), official_shape=list(tokens.shape),
               sample_ids=[subset.rows[i]["sample_id"] for i in positions[:size]], heads={})
    for name, model in build_heads(config, 0, device).items():
        optimizer = optimizer_for(model, config)
        layer = next(model.parameters())
        initial = layer.detach().clone()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = logits_for(name, model, x)
                loss = F.cross_entropy(logits, y)
            require(logits.shape == (len(x), 5) and torch.isfinite(logits).all().item(),
                    f"{name} output invalid")
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                        for p in model.parameters()), f"{name} missing/nonfinite gradient")
            optimizer.step()
        require(not torch.equal(layer, initial), f"{name} weights did not update")
        out["heads"][name] = dict(
            logits_shape=list(logits.shape), loss=float(loss.detach()), updates=4,
            seconds=time.perf_counter() - start, parameters=sum(p.numel() for p in model.parameters()),
            peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2**20
                                if device.type == "cuda" else None),
        )
    return out


def train_statistics(data, config, device):
    shape = (2, data.n_layers, 128)
    sums = torch.zeros(shape, dtype=torch.float64, device=device)
    squares = torch.zeros_like(sums)
    count = 0
    for raw, _ in data.groups(config["batch_size"], config["blocks_per_group"]):
        x = torch.as_tensor(raw, device=device, dtype=torch.float32)
        sums += x.sum(dim=(0, 3), dtype=torch.float64)
        squares += x.square().sum(dim=(0, 3), dtype=torch.float64)
        count += len(x) * x.shape[3]
    mean = sums / count
    std = (squares / count - mean.square()).clamp_min(1e-12).sqrt()
    require(torch.isfinite(mean).all().item() and torch.isfinite(std).all().item(),
            "Nonfinite training normalization")
    return mean.float()[None, :, :, None, :], std.float()[None, :, :, None, :]


def train_epoch(models, optimizers, data, config, device, mean, std, rng, on_progress):
    losses = torch.zeros(len(models), device=device)
    count = updates = 0
    size = config["batch_size"]
    for raw, positions in data.groups(size, config["blocks_per_group"], rng):
        x = torch.as_tensor(raw, device=device, dtype=torch.float32)
        x.sub_(mean).div_(std)
        y = torch.as_tensor(data.labels[positions], device=device)
        for batch, labels in zip(x.split(size), y.split(size)):
            for i, (name, model) in enumerate(models.items()):
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = logits_for(name, model, batch)
                    loss = F.cross_entropy(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                optimizer.step()
                losses[i] += loss.detach() * len(batch)
            count += len(batch)
            updates += 1
        if updates >= 32 and updates % 32 < config["blocks_per_group"]:
            on_progress(updates, count, losses)
    require(count == len(data.rows), "Incomplete training epoch")
    return updates, count, losses


@torch.inference_mode()
def predict_development(models, data, config, device, mean, std):
    probabilities = {name: np.full((len(data.rows), 5), np.nan, dtype=np.float32) for name in models}
    size = config["batch_size"]
    for raw, positions in data.groups(size, config["blocks_per_group"]):
        x = torch.as_tensor(raw, device=device, dtype=torch.float32)
        x.sub_(mean).div_(std)
        for start, batch in enumerate(x.split(size)):
            ids = positions[start * size:(start + 1) * size]
            for name, model in models.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = logits_for(name, model, batch)
                probabilities[name][ids] = logits.float().softmax(dim=1).cpu().numpy()
    return probabilities


def evaluate_pair(output, data, probabilities):
    official = official_module()
    frames = {}
    results = {}
    fields = ("sample_id", "subject_id", "record_id", "epoch_index", "y_true")
    for name, values in probabilities.items():
        require(np.isfinite(values).all(), f"Incomplete or nonfinite {name} predictions")
        rows = []
        for row, prob in zip(data.rows, values):
            rows.append({**{key: row[key] for key in fields}, "y_pred": int(prob.argmax()),
                         **dict(zip(official.PROB_COLS, map(float, prob)))})
        folder = output / name
        pred_path = folder / "predictions_val.csv"
        write_csv(pred_path, rows)
        frames[name] = official.load_predictions(pred_path)
        results[name] = official.compute_metrics(frames[name]["y_true"], frames[name]["y_pred"])
        write_json(folder / "metrics.json", results[name])
        official.per_subject_table(frames[name]).to_csv(folder / "per_subject_metrics.csv", index=False)
    paired = official.paired_bootstrap_delta(frames["C1"], frames["C0"], n_boot=2000, seed=0)
    subjects = [official.per_subject_table(frames[name]).set_index("subject_id")
                for name in ("C0", "C1")]
    (subjects[1][["macro_f1", "n1_f1"]] - subjects[0][["macro_f1", "n1_f1"]]).to_csv(
        output / "per_subject_delta_C1_minus_C0.csv"
    )
    report = dict(metrics=results, paired_C1_minus_C0=paired, test_used=False)
    write_json(output / "paired_results.json", report)
    return report


def run_pair(config, rows, train, dev, seed, device, output, sources=SOURCES):
    train_data = CachedEpochs(rows, train, config["layers"], sources)
    rng = np.random.default_rng(seed)
    models = build_heads(config, seed, device)
    optimizers = {name: optimizer_for(model, config) for name, model in models.items()}
    status = dict(seed=seed, epochs_planned=config["epochs"], train_samples=len(train_data.rows),
                  device=torch.cuda.get_device_name(device), torch_version=str(torch.__version__),
                  parameters={name: sum(p.numel() for p in model.parameters())
                              for name, model in models.items()}, test_used=False)

    def progress(stage, **values):
        updated = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
        write_json(output / "progress.json", dict(status, stage=stage, updated_at=updated, **values))
        print(json.dumps(dict(stage=stage, **values)), flush=True)

    start = time.perf_counter()
    try:
        progress("fitting_train_normalization")
        mean, std = train_statistics(train_data, config, device)
        status["normalization_seconds"] = time.perf_counter() - start
        history = []
        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            progress("training", epoch=epoch, updates=0, samples=0)

            def report_batch(updates, count, losses):
                avg = (losses / count).tolist()
                require(np.isfinite(avg).all(), "Nonfinite training loss")
                progress("training", epoch=epoch, updates=updates, samples=count,
                         train_loss=dict(zip(models, avg)), epoch_seconds=time.perf_counter() - epoch_start)

            updates, count, losses = train_epoch(
                models, optimizers, train_data, config, device, mean, std, rng, report_batch
            )
            avg = (losses / count).tolist()
            require(np.isfinite(avg).all(), "Nonfinite training loss")
            history.append(dict(epoch=epoch, updates=updates, samples=count,
                                train_loss=dict(zip(models, avg)),
                                epoch_seconds=time.perf_counter() - epoch_start,
                                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20))
            write_json(output / "history.json", history)
            progress("epoch_complete", **history[-1])
        status["training_seconds"] = sum(row["epoch_seconds"] for row in history)
        for name, model in models.items():
            folder = output / name
            folder.mkdir()
            require(all(torch.isfinite(p).all().item() for p in model.parameters()),
                    f"Nonfinite final {name} parameters")
            torch.save(dict(model=model.state_dict(), mean=mean.cpu(), std=std.cpu(), seed=seed,
                            epoch=config["epochs"], config=config), folder / "final_head.pt")
            model.eval()
        progress("evaluating_development")
        dev_data = CachedEpochs(rows, dev, config["layers"], sources)
        probabilities = predict_development(models, dev_data, config, device, mean, std)
        report = evaluate_pair(output, dev_data, probabilities)
        progress("complete", elapsed_seconds=time.perf_counter() - start,
                 dev_samples=len(dev_data.rows), paired_C1_minus_C0=report["paired_C1_minus_C0"])
    except Exception as error:
        progress("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "train"])
    parser.add_argument("--dev-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    layers = config["layers"]
    require(layers and layers == list(range(layers[0], layers[-1] + 1))
            and 0 <= layers[0] <= layers[-1] < 20, "Invalid layer selection")
    require(args.seed in config["seeds"], "Seed outside the planned pairs")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA GPU required")
    args.output.mkdir(parents=True, exist_ok=False)
    print("Auditing source indices and cache headers", flush=True)
    rows, train, dev, audit = inspect_inputs(args.dev_reference)
    write_json(args.output / "input_audit.json", audit)
    write_json(args.output / "config.json", config)
    if args.command == "smoke":
        result = smoke(config, rows, train, device)
        write_json(args.output / "smoke.json", result)
        print(json.dumps(result), flush=True)
    else:
        run_pair(config, rows, train, dev, args.seed, device, args.output)


if __name__ == "__main__":
    main()
