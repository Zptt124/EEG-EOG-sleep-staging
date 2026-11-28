"""Validate frozen inputs shared by the extractor and the head runner."""
import json
import os
from pathlib import Path

from download import ROOT, OLD, OUT, MANIFEST, hash_file

BLOCK = 60
BUNDLE = Path(os.environ.get("BUDGET_C_BUNDLE",
                            ROOT / "experiment3/deliveries/d-w6-shared-cache-v01-20261126/full"))
FEATURE_HASH = "392122c602030d8dcfa903a8d5a5f55fd582f0e09dc6d086248c86bac4267bdf"


def load_lock():
    lock = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    expected = {OUT / name: digest for name, digest in lock["files"].items()}
    expected.update({OLD / "plan.json": lock["original_plan_sha256"],
                     OLD / "evaluation_index.npz": lock["evaluation_sha256"],
                     MANIFEST: lock["manifest_sha256"],
                     BUNDLE / "features_info.json": FEATURE_HASH})
    info = json.loads((BUNDLE / "features_info.json").read_text())
    expected[BUNDLE / "index.csv"] = info["index_sha256"]
    for name in ("eeg_std", "eog_std"):
        expected[BUNDLE / f"{name}.npy"] = info["files"][name]["sha256"]
    for path, digest in expected.items():
        if hash_file(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return lock
