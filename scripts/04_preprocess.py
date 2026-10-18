"""Step 4/5: preprocess Sleep-EDF SC into per-record 30-s epoch files.

Sleep-EDF SC hypnograms are variable-length R&K stage runs over 24 h
(not fixed 30-s epochs), so we expand them onto a 30-s grid:
grid epoch k covers [30k, 30k+30) s of the recording and inherits the
label of the run covering that time.

Per record:
  - read PSG (Fpz-Cz, EOG horizontal @100 Hz) and its hypnogram
  - expand runs -> 30-s grid (2880 epochs over 24 h)
  - trim to [30 min before sleep onset, 30 min after final wake]
  - map stages (3+4 -> N3), drop Movement/Unknown epochs entirely
  - save data/processed/<record_id>.npz:
      signal      float32 (n_epochs, 2, 3000)  in uV  [0]=EEG [1]=EOG
      labels      int8    (n_epochs,)          0=W 1=N1 2=N2 3=N3 4=REM
      epoch_index int32   (n_epochs,)          grid position k (stable sample_id)
  - collect per-record stats into data/processed/processed_stats.csv

Small-batch mode (first N records + alignment plots for data_checks.md):
    python 04_preprocess.py --records 8 --plots
Full run:
    python 04_preprocess.py --full --workers 8
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import mne

from common import (RAW, PROCESSED, SPLIT, FIG, DOCS,
                    CHANNELS, EPOCH_PTS, TRIM_MARGIN_MIN, LABEL_MAP, DROP_STAGES,
                    STAGE_NAMES, sample_id, annotation_stage)

STATS_CSV = PROCESSED / "processed_stats.csv"
MARGIN_EPOCHS = TRIM_MARGIN_MIN * 2          # 30 min = 60 epochs of 30 s
SLEEP_STAGES = {"1", "2", "3", "4", "R"}


def build_epoch_grid(hyp) -> np.ndarray:
    """Expand variable-length stage runs onto the 30-s grid.

    Returns an array of stage chars ('W','1','2','3','4','R','M','?') with
    one entry per 30-s slot covering [0, end of last run).
    """
    starts, durs = hyp.onset, hyp.duration
    ends = starts + durs
    n_epochs = int(np.ceil(ends[-1] / 30))
    t = np.arange(n_epochs) * 30.0
    run_idx = np.searchsorted(starts, t, side="right") - 1
    stage_runs = np.array([annotation_stage(d) or "?" for d in hyp.description])
    grid = stage_runs[run_idx]
    # a slot must lie inside its run (guards gaps / boundary slop)
    inside = t < ends[run_idx]
    grid = np.where(inside, grid, "?")
    return grid


def process_one(record_id: str, psg_file: str, hyp_file: str, plot: bool = False) -> dict:
    # pick BEFORE loading: the 1 Hz aux channels (resp/temp/event) would otherwise
    # be FFT-resampled to 100 Hz by mne at preload time -> MemoryError with 8 workers.
    raw = mne.io.read_raw_edf(RAW / psg_file, preload=False, verbose="error")
    raw.pick(CHANNELS).load_data()
    psg_len = raw.times[-1]
    if raw.info["sfreq"] != 100:
        raw = raw.resample(100, npad="auto")
    hyp = mne.read_annotations(RAW / hyp_file)
    grid = build_epoch_grid(hyp)
    n_epochs = len(grid)
    # mne reports these EDFs in V (physical units); Sleep-EDF amplitudes are uV.
    # Empirically: x1e6 gives healthy EEG (std ~15-30 uV) and EOG (std ~70 uV).
    UV = 1e6

    # sleep onset / final wake: first/last grid epoch that is actually sleep
    sleep_mask = np.isin(grid, list(SLEEP_STAGES))
    if sleep_mask.any():
        i_on = int(np.argmax(sleep_mask))
        i_wake = int(n_epochs - 1 - np.argmax(sleep_mask[::-1]))
    else:
        i_on, i_wake = 0, n_epochs - 1
    lo = max(0, i_on - MARGIN_EPOCHS)
    hi = min(n_epochs, i_wake + MARGIN_EPOCHS + 1)

    sig_list, lab_list, idx_list = [], [], []
    n_drop_stage = 0
    for k in range(lo, hi):
        s = grid[k]
        if s in DROP_STAGES:
            n_drop_stage += 1
            continue
        t = k * 30.0
        s0 = int(t * 100)                  # get_data uses sample indices
        if s0 + EPOCH_PTS > int(psg_len * 100) + 50:
            continue                       # beyond PSG end -> counted below
        seg = raw.get_data(start=s0, stop=s0 + EPOCH_PTS)
        if seg.shape[1] != EPOCH_PTS:
            continue
        sig_list.append((seg * UV).astype(np.float32))
        lab_list.append(LABEL_MAP[s])
        idx_list.append(k)

    raw.close()
    n_kept = len(sig_list)
    n_drop_short = (hi - lo) - n_drop_stage - n_kept
    stats = {
        "record_id": record_id,
        "n_grid": n_epochs,
        "n_kept": n_kept,
        "n_drop_stage": n_drop_stage,
        "n_drop_short": n_drop_short,
        "trim_lo": lo, "trim_hi": hi,
        "sleep_onset_idx": i_on, "final_wake_idx": i_wake,
        "psg_len_h": round(psg_len / 3600, 2),
    }
    if n_kept:
        signal = np.stack(sig_list)
        labels = np.array(lab_list, dtype=np.int8)
        epoch_idx = np.array(idx_list, dtype=np.int32)
        np.savez(PROCESSED / f"{record_id}.npz",
                 signal=signal, labels=labels, epoch_index=epoch_idx)
        for c, name in enumerate(STAGE_NAMES):
            stats[f"n_{name}"] = int((labels == c).sum())

        if plot:
            plot_alignment(record_id, signal[:, 0], labels, epoch_idx)
            plot_hypnogram(record_id, grid, lo, hi, i_on, i_wake)
    return stats


def plot_alignment(record_id, eeg, labels, epoch_idx):
    """40-epoch slice of EEG with stage-colored background (alignment check)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_show = 40
    x = np.arange(min(n_show, len(eeg)) * EPOCH_PTS) / 100
    y = eeg[:n_show].ravel()
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=110)
    colors = {"W": "#e0e0e0", "N1": "#a6cee3", "N2": "#b2df8a", "N3": "#1f78b4", "REM": "#fb9a99"}
    for j in range(min(n_show, len(eeg))):
        ax.axvspan(j * 30, (j + 1) * 30, color=colors[STAGE_NAMES[labels[j]]], alpha=0.55)
    ax.plot(x, y, lw=0.35, color="k")
    ax.set_title(f"{record_id}  Fpz-Cz  30-s epochs (epoch_index {epoch_idx[0]}..{epoch_idx[min(n_show, len(eeg)) - 1]})")
    ax.set_xlabel("time (s)"); ax.set_ylabel("uV")
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=c, label=s) for s, c in colors.items()],
              ncol=5, loc="upper right", fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"align_{record_id}.png")
    plt.close(fig)


def plot_hypnogram(record_id, grid, lo, hi, i_on, i_wake):
    """Full-24h stage overview with the kept window marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ymap = {"W": 4, "1": 3, "2": 2, "3": 1, "4": 0, "R": 0.5, "M": -0.5, "?": -1}
    y = np.array([ymap.get(s, -1.5) for s in grid], dtype=float)
    x = np.arange(len(y)) / 120            # hours (2 epochs per min)
    fig, ax = plt.subplots(figsize=(14, 2.6), dpi=110)
    ax.step(x, y, where="post", lw=0.5)
    ax.axvspan(lo / 120, hi / 120, color="orange", alpha=0.15, label="kept window")
    ax.axvline(i_on / 120, color="g", lw=1, label="sleep onset")
    ax.axvline(i_wake / 120, color="r", lw=1, label="final wake")
    ax.set_yticks([4, 3, 2, 1, 0, 0.5, -0.5, -1], ["W", "N1", "N2", "N3", "N4", "R", "M", "?"])
    ax.set_title(f"{record_id} hypnogram grid (trim {lo}..{hi}, kept {hi - lo} epochs)")
    ax.set_xlabel("hours since recording start"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"hypnogram_{record_id}.png")
    plt.close(fig)


def run_batch(records: list[dict], plots: bool, workers: int) -> list[dict]:
    import concurrent.futures as cf
    PROCESSED.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    if workers > 1 and len(records) > 1:
        with cf.ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(process_one, r["record_id"], r["psg_file"], r["hyp_file"], plots and i < 2): r
                    for i, r in enumerate(records)}
            for fut in cf.as_completed(futs):
                r = futs[fut]
                try:
                    results.append(fut.result())
                    print(f"  ok   {r['record_id']}: kept {results[-1]['n_kept']}/{results[-1]['n_grid']} grid epochs", flush=True)
                except Exception:
                    errors.append(r["record_id"])
                    print(f"  FAIL {r['record_id']}:\n{traceback.format_exc()}", flush=True)
    else:
        for i, r in enumerate(records):
            try:
                st = process_one(r["record_id"], r["psg_file"], r["hyp_file"], plots and i < 2)
                results.append(st)
                print(f"  ok   {r['record_id']}: kept {st['n_kept']}/{st['n_grid']} grid epochs", flush=True)
            except Exception:
                errors.append(r["record_id"])
                print(f"  FAIL {r['record_id']}:\n{traceback.format_exc()}", flush=True)
    if errors:
        print(f"\n{len(errors)} records failed: {errors}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=None, help="process first N records")
    ap.add_argument("--full", action="store_true", help="process all records")
    ap.add_argument("--missing", action="store_true", help="process only records without an npz yet")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plots", action="store_true", help="write alignment figures for data_checks.md")
    args = ap.parse_args()

    manifest = pd.read_csv(SPLIT / "dataset_manifest.csv", dtype=str)
    if args.full:
        records = manifest.to_dict("records")
    elif args.missing:
        missing = [r for r in manifest["record_id"] if not (PROCESSED / f"{r}.npz").exists()]
        records = manifest[manifest["record_id"].isin(missing)].to_dict("records")
        print(f"{len(missing)} records missing npz")
    elif args.records:
        records = manifest.head(args.records).to_dict("records")
    else:
        records = manifest.head(8).to_dict("records")
    print(f"processing {len(records)} records (workers={args.workers})")
    results = run_batch(records, args.plots, args.workers)
    if not results:
        sys.exit("no records processed successfully — aborting")
    df_new = pd.DataFrame(results).sort_values("record_id")
    if STATS_CSV.exists() and args.missing:
        df_old = pd.read_csv(STATS_CSV)
        df = pd.concat([df_old[~df_old["record_id"].isin(df_new["record_id"])], df_new],
                       ignore_index=True).sort_values("record_id")
    else:
        df = df_new
    df.to_csv(STATS_CSV, index=False)
    print(f"\nwrote {STATS_CSV}")
    cols = [c for c in df.columns if c.startswith("n_") and c[2:] in STAGE_NAMES]
    print("class counts across processed records:")
    print(df[cols].sum().to_string())
    print(f"dropped: M/?={df['n_drop_stage'].sum()}, beyond-PSG={df['n_drop_short'].sum()}")
    print(f"total kept epochs: {df['n_kept'].sum()}")


if __name__ == "__main__":
    sys.exit(main())
