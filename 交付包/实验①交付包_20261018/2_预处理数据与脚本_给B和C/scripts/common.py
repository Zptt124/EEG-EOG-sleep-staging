"""Shared constants and helpers for Experiment 1 (实验①: 数据与单模态基线).

Project: 睡眠分期课题 · 第一周工作文档
Owner:   A (实验① 数据与单模态基线)
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent          # exp1/
RAW = BASE / "data" / "raw"                             # EDF files (junction to D:\sleepedf\raw)
PROCESSED = BASE / "data" / "processed"                 # per-record npz segments (junction to D:)
SPLIT = BASE / "split"
DOCS = BASE / "docs"
RUNS = BASE / "runs"
FIG = DOCS / "figs"

# ---- data rules (from the team doc, section 二) ----
EEG_CH = "EEG Fpz-Cz"
EOG_CH = "EOG horizontal"
CHANNELS = [EEG_CH, EOG_CH]        # index 0 = EEG, 1 = EOG
SFREQ = 100
EPOCH_SEC = 30
EPOCH_PTS = SFREQ * EPOCH_SEC      # 3000
TRIM_MARGIN_MIN = 30               # keep 30 min before sleep onset .. 30 min after final wake

# label mapping: 0=W, 1=N1, 2=N2, 3=N3, 4=REM  (stages 3+4 merged into N3)
LABEL_MAP = {"W": 0, "1": 1, "2": 2, "3": 3, "4": 3, "R": 4}
DROP_STAGES = {"M", "?"}           # Movement / Unknown removed entirely
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES = 5

# subject-wise split (from the team doc): ~70% / 10% / 20% of 78 subjects
SPLIT_SEED = 42
SPLIT_COUNTS = {"train": 54, "val": 8, "test": 16}


def subject_id_of(record_id: str) -> str:
    """SC4001 -> SC400 (first 5 chars)."""
    return record_id[:5]


def night_of(record_id: str) -> str:
    """SC4001 -> '1' (6th char)."""
    return record_id[5]


def sample_id(record_id: str, epoch_index: int) -> str:
    """Stable per-epoch id, e.g. SC4001_0123.

    epoch_index = position in the 30-s epoch grid expanded from the hypnogram
    timeline (0-based, zero-padded to 4 digits). Sleep-EDF SC hypnograms are
    variable-length R&K stage runs over 24 h; the grid epoch k covers
    [30k, 30k+30) s. Indices are unchanged by trimming/dropping, so ids stay
    stable across models.
    """
    return f"{record_id}_{epoch_index:04d}"


def annotation_stage(desc: str) -> str | None:
    """'Sleep stage W' -> 'W'; 'Movement time' -> 'M'; None if unrecognized."""
    if desc == "Movement time":
        return "M"
    if desc.startswith("Sleep stage "):
        return desc[len("Sleep stage "):]
    return None
