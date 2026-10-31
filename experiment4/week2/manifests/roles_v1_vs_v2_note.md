# Roles equivalence: v1_provisional vs v2 (B's official label_subsets_v2.csv)

- Date: 2026-10-31
- Compared columns per subject: split, sampling_seed, sampling_rank, n_epochs, role_label10, role_label25
- Subjects compared: 78 (54 train, 8 val, 16 test)
- Differences found: 0
- budget_counts.csv identical: True
- v1_provisional roles file sha256: 6f9882e9444d544dd2b709fd325dc2cc0231ff93360738bc886dac80d1a73173
- v2 roles file sha256: ef81a35a660946f3f236c9ee6ab6bede546cb7fd0bc9e6a6be2b3a742bc47582
- The only difference between the two files is the label10_source column
  (derived_rank_le_5_PROVISIONAL vs B_v2_column); every subject's role in label10 and label25 is identical.

## Consequence for registered runs

- All R1 runs (D0/D1/D2, label10 and label25, seeds 0-4) and the two R1 calibrations were produced with
  manifests/v1_provisional/labeled_unlabeled_subjects.csv. Because the labeled/unlabeled assignment is
  identical to v2 for every subject, these results are reused as v2 results (strict match, no rerun).
- All R2 runs and the two R2 calibrations were produced directly with manifests/v2/labeled_unlabeled_subjects.csv.
- Per-subject evidence: roles_v1_vs_v2_check.csv
