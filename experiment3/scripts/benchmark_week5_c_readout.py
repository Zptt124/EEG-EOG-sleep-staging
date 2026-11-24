"""Measure W5 C backbone extraction, pooling, copy, and cache persistence separately."""

import importlib.util
import os
from pathlib import Path
import tempfile
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from run_week5_c_late_layers import ASSETS, INDEX_SHA, OUTPUTS, ROOT, SPECS, TEAM, sha


OUT = OUTPUTS / "W5_C_readout_cost_v01"
ARMS = {"all12": slice(None), "last3": slice(9, 12)}
MODALITIES = ("eeg", "eog")
BATCH = 8
PER_SUBJECT = 16
GPU_ROUNDS = 4
DISK_ROUNDS = 3
ARCHIVE_SHA = "b17e8561136a0b249796364bfbbf1e79c6c3d62b7b25c08951e8412e543094d0"


def load_extractor():
    if "SLEEP_DATA" not in os.environ:
        (processed,) = (ROOT / "experiment3/vendor/experiment1-review/交付包").glob("*/2_*/data/processed")
        os.environ["SLEEP_DATA"] = str(processed)
    os.environ.setdefault("EEGDM_VENDOR", str(ROOT / "experiment3/vendor/EEGDM"))
    path = TEAM / "experiment4/week4/src/04_extract_features.py"
    spec = importlib.util.spec_from_file_location("w4_feature_extractor", path)
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    return extractor


def load_model(extractor, temp):
    archive = ASSETS / "w4_d_main_ckpt_0to30k_v01.zip"
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError("Unexpected checkpoint archive")
    ckpt = temp / "ckpt_step30000.pt"
    with ZipFile(archive) as source:
        ckpt.write_bytes(source.read("w4_d_main_ckpt_0to30k_v01/ckpt_step30000.pt"))
    if sha(ckpt) != SPECS["plain"].checkpoint_sha:
        raise ValueError("Checkpoint does not match the four-cell feature source")
    model, state, switches = extractor.load_model(ckpt, "ema_model")
    if (state["step"] != 30000 or state["run_config"]["run_name"] != SPECS["plain"].backbone
            or switches["self_cond_layer"] or switches["cross_modal"] != "none"):
        raise ValueError("Unexpected W4 model configuration")
    return model.cuda()


def load_dev(extractor):
    features = SPECS["plain"].out / "features"
    path = features / "index.csv"
    if sha(path) != INDEX_SHA:
        raise ValueError("Unexpected feature index")
    index = pd.read_csv(path)
    subject_groups = list(index[index.split.eq("val")].groupby("subject_id"))
    if len(subject_groups) != 8:
        raise ValueError("Expected eight dev subjects")
    chosen = pd.concat([group.iloc[np.linspace(0, len(group) - 1, PER_SUBJECT, dtype=int)]
                        for _, group in subject_groups]).sort_index()
    raw = np.empty((len(chosen), len(MODALITIES), 3000), dtype=np.float32)
    positions = {index_value: position for position, index_value in enumerate(chosen.index)}
    records = extractor.C.split_records()
    for record, group in chosen.groupby("record_id"):
        if records[record] != (group.subject_id.iloc[0], "val"):
            raise ValueError(f"Unexpected record split: {record}")
        with np.load(extractor.C.DATA / f"{record}.npz", allow_pickle=False) as source:
            lookup = {int(epoch): i for i, epoch in enumerate(source["epoch_index"])}
            rows = [lookup[epoch] for epoch in group.epoch_index]
            if not np.array_equal(source["labels"][rows], group.y_true.to_numpy()):
                raise ValueError(f"Labels do not match for {record}")
            raw[[positions[i] for i in group.index]] = source["signal"][rows]
    transformed = [torch.from_numpy(extractor.C.transform_uv(raw[:, m])[:, None]).cuda()
                   for m in range(len(MODALITIES))]
    return chosen, transformed


@torch.inference_mode()
def timed_extract(extractor, model, inputs, layers):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    vectors = []
    timing = np.zeros(3)
    torch.cuda.reset_peak_memory_stats()
    for modality, x in enumerate(inputs):
        events[0].record()
        gates = extractor.gate_activations(model, x, modality)
        events[1].record()
        gates = gates[:, layers]
        batch, width, length, channels = gates.shape
        std = gates.reshape(batch, width, length // 200, 200, channels).std(dim=3).mean(dim=2)
        events[2].record()
        events[2].synchronize()
        start = perf_counter()
        vectors.append(std.reshape(batch, -1).cpu().numpy())
        timing += [events[0].elapsed_time(events[1]), events[1].elapsed_time(events[2]),
                   (perf_counter() - start) * 1000]
    return vectors, dict(zip(("backbone_gate_ms", "pool_ms", "device_to_cpu_ms"), timing),
                         peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)


def measure_gpu(extractor, model, sample, signals):
    expected = {}
    for arm in ARMS:
        expected[arm] = []
        for mod in MODALITIES:
            path = SPECS["plain"].out / "features" / f"plain30k_{arm}_{mod}.npz"
            with np.load(path, allow_pickle=False) as source:
                if not np.array_equal(source["sample_id"][sample.index], sample.sample_id):
                    raise ValueError(f"Feature identities differ in {path.name}")
                expected[arm].append(source["x"][sample.index])
    for _ in range(3):
        for layers in ARMS.values():
            timed_extract(extractor, model, [signal[:BATCH] for signal in signals], layers)

    rows, max_error = [], 0.0
    for repeat in range(GPU_ROUNDS):
        for start in range(0, len(sample), BATCH):
            stop = start + BATCH
            inputs = [signal[start:stop] for signal in signals]
            order = list(ARMS) if (repeat + start // BATCH) % 2 == 0 else list(reversed(ARMS))
            vectors = {}
            for arm in order:
                vectors[arm], stats = timed_extract(extractor, model, inputs, ARMS[arm])
                rows.append(dict(arm=arm, repeat=repeat, batch_start=start, examples=BATCH, **stats))
            if repeat == 0:
                for all12, last3 in zip(vectors["all12"], vectors["last3"]):
                    np.testing.assert_array_equal(all12[:, 576:], last3)
                for arm in ARMS:
                    for actual, ref in zip(vectors[arm], expected[arm]):
                        wanted = ref[start:stop]
                        np.testing.assert_allclose(actual, wanted, rtol=0.02, atol=0.0005)
                        max_error = max(max_error, float(np.max(np.abs(actual - wanted))))
        print(f"GPU round {repeat + 1}/{GPU_ROUNDS} complete", flush=True)
    return pd.DataFrame(rows), max_error


def measure_disk(temp):
    rows = []
    for backbone, spec in SPECS.items():
        for repeat in range(DISK_ROUNDS):
            order = list(ARMS) if repeat % 2 == 0 else list(reversed(ARMS))
            for arm in order:
                sources = []
                for modality in MODALITIES:
                    with np.load(spec.out / "features" / f"{backbone}30k_{arm}_{modality}.npz",
                                 allow_pickle=False) as source:
                        sources.append(dict(sample_id=source["sample_id"], x=source["x"]))
                elapsed, size = 0.0, 0
                for modality, source in zip(MODALITIES, sources):
                    path = temp / f"{modality}.npz"
                    start = perf_counter()
                    with path.open("wb") as file:
                        np.savez(file, **source)
                        file.flush()
                        os.fsync(file.fileno())
                    elapsed += perf_counter() - start
                    size += path.stat().st_size
                rows.append(dict(backbone=backbone, arm=arm, repeat=repeat, samples=len(sources[0]["x"]),
                                 bytes=size, write_seconds=elapsed))
        print(f"Cache write measurements complete for {backbone}", flush=True)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    extractor = load_extractor()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with tempfile.TemporaryDirectory(dir=OUT) as folder:
        temp = Path(folder)
        model = load_model(extractor, temp)
        sample, signals = load_dev(extractor)
        print(f"Checkpoint validated; timing {len(sample)} fixed dev epochs", flush=True)
        gpu, max_error = measure_gpu(extractor, model, sample, signals)
        gpu.to_csv(OUT / "gpu_batches.csv", index=False)
        disk = measure_disk(temp)
    disk.to_csv(OUT / "cache_writes.csv", index=False)
    sample.to_csv(OUT / "dev_sample.csv", index=False)
    phases = ["backbone_gate_ms", "pool_ms", "device_to_cpu_ms"]
    stats = gpu.groupby("arm")[phases].quantile([0.25, 0.5, 0.75]).reset_index(names=["arm", "quantile"])
    stats.to_csv(OUT / "gpu_summary.csv", index=False)
    extractor.C.write_json(OUT / "verification.json", dict(
        checkpoint_sha256=SPECS["plain"].checkpoint_sha, model=SPECS["plain"].backbone,
        source_code_sha256=sha(Path(extractor.__file__)), index_sha256=INDEX_SHA,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, tf32=True, batch=BATCH,
        dev_subjects=sample.subject_id.nunique(), unique_epochs=len(sample),
        gpu_rounds=GPU_ROUNDS, disk_rounds=DISK_ROUNDS, max_abs_feature_error=max_error,
        late_slice_equal=True, test_used=False,
    ))
    print(stats.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
