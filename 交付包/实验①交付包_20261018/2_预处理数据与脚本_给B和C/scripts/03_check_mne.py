"""Step 3: read 2-3 nights with MNE, confirm channels / sampling rate / label
format, and write docs/preprocess.yaml (the group's shared preprocessing rules).

Run after the first PSG+hypnogram files have downloaded:
    python 03_check_mne.py
"""
import sys

import numpy as np
import pandas as pd
import yaml

import mne

from common import RAW, SPLIT, DOCS, EEG_CH, EOG_CH, CHANNELS, LABEL_MAP, DROP_STAGES

N_RECORDS = 3
OUT_YAML = DOCS / "preprocess.yaml"


def check_one(record_id: str, psg_file: str, hyp_file: str):
    print(f"\n===== {record_id}  (PSG {psg_file}, hyp {hyp_file}) =====")
    psg_path = RAW / psg_file
    hyp_path = RAW / hyp_file
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="error")
    print(f"sfreq: {raw.info['sfreq']:.1f} Hz, n_channels: {len(raw.ch_names)}")
    print(f"channels: {raw.ch_names}")
    print(f"duration: {raw.times[-1] / 3600:.2f} h")
    for ch in CHANNELS:
        assert ch in raw.ch_names, f"{ch} missing!"
    on = raw.annotations.onset
    print(f"PSG annotations: {len(on)}, desc set: {set(raw.annotations.description)}")

    hyp = mne.read_annotations(hyp_path)
    print(f"hypnogram annotations: {len(hyp.onset)}, duration {hyp.duration.min():.1f}-{hyp.duration.max():.1f} s")
    descs = hyp.description
    print("hypnogram desc values:", sorted(set(descs)))
    print("duration counts:", pd.Series(hyp.duration).value_counts().to_dict())
    print(f"hyp onset range: {hyp.onset.min():.1f} .. {hyp.onset.max():.1f} s")

    # stage stats (SC hypnograms are variable-length R&K runs -> expanded to a 30-s grid)
    stages = [d.replace("Sleep stage ", "") for d in descs]
    counts = pd.Series(stages).value_counts()
    print("stage-run counts:", counts.to_dict())
    print(f"duration min/median/max: {hyp.duration.min():.0f} / "
          f"{pd.Series(hyp.duration).median():.0f} / {hyp.duration.max():.0f} s "
          f"(variable-length runs, not fixed 30-s epochs)")

    # any annotation whose onset+duration exceeds PSG length?
    psg_len = raw.times[-1]
    over = ((hyp.onset + hyp.duration) > psg_len + 1.0).sum()
    print(f"annotations past PSG end: {over}")
    # hypnogram coverage vs PSG length (end of the LAST annotation, not max duration)
    print(f"hyp covers {hyp.onset.min():.0f}s..{hyp.onset[-1] + hyp.duration[-1]:.0f}s "
          f"of {psg_len:.0f}s (PSG {psg_len / 3600:.2f} h)")

    # sample a segment of Fpz-Cz to confirm plausible EEG amplitude range
    raw.pick([EEG_CH, EOG_CH])
    seg, _ = raw[: int(30 * raw.info["sfreq"])]
    seg = seg * 1e6   # mne reports V for these EDFs; uV is the natural unit
    print(f"first-30s amplitude (uV): EEG mean {np.abs(seg[0]).mean():.2f} max {np.abs(seg[0]).max():.2f}; "
          f"EOG mean {np.abs(seg[1]).mean():.2f} max {np.abs(seg[1]).max():.2f}")
    raw.close()
    return stages


def main():
    manifest = pd.read_csv(SPLIT / "dataset_manifest.csv", dtype=str)
    if not (RAW / "SHA256SUMS.txt").exists():
        sys.exit("raw data folder missing — run download_data.py first")
    rows = manifest.head(N_RECORDS)
    summaries = []
    for _, r in rows.iterrows():
        if not (RAW / r["psg_file"]).exists() or not (RAW / r["hyp_file"]).exists():
            print(f"skip {r['record_id']}: files not downloaded yet")
            continue
        stages = check_one(r["record_id"], r["psg_file"], r["hyp_file"])
        summaries.append({"record_id": r["record_id"],
                          "n_runs": len(stages),
                          "run_counts": pd.Series(stages).value_counts().to_dict()})
    if not summaries:
        sys.exit(f"ERROR: {N_RECORDS} records were to be checked but 0 were read — "
                 f"not writing preprocess.yaml on unverified data")

    yaml_rules = {
        "version": "v1",
        "author": "A (实验①)",
        "channel": {"eeg": EEG_CH, "eog": EOG_CH, "order": ["eeg", "eog"]},
        "sfreq": 100,
        "epoch_sec": 30,
        "hypnogram_note": "SC hypnograms are variable-length R&K stage runs over 24 h; "
                         "runs are expanded onto a 30-s grid, epoch k = [30k, 30k+30) s",
        "trim": {"before_sleep_onset_min": 30, "after_final_wake_min": 30,
                 "sleep_onset": "first grid epoch with stage 1/2/3/4/R",
                 "final_wake": "last grid epoch with stage 1/2/3/4/R"},
        "label_map": {k: v for k, v in LABEL_MAP.items()},
        "label_order": ["W", "N1", "N2", "N3", "REM"],
        "drop_stages": sorted(DROP_STAGES),
        "normalization": "per-channel z-score, mean/std computed on TRAIN subjects only",
        "sample_id": "record_id + '_' + 4-digit grid epoch index k (0-based)",
        "note": "3/4 merged into N3; Movement and Unknown epochs dropped entirely",
    }
    DOCS.mkdir(exist_ok=True)
    with open(OUT_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_rules, f, allow_unicode=True, sort_keys=False)
    print(f"\nwrote {OUT_YAML}")
    print("\nMNE check summary:")
    print(pd.DataFrame(summaries).to_string())


if __name__ == "__main__":
    sys.exit(main())
