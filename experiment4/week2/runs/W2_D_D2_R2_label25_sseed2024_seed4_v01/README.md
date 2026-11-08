# W2_D_D2_R2_label25_sseed2024_seed4_v01

- status: success
- last successful step: metrics.json written
- method: D2 (cross-modal self-training: D1 candidates AND EEG-only and EOG-only teachers predict the same class)
- representation: R2: R2_A_cnn_label25_seed4: A's EEG-only + EOG-only SmallCNN encoders (128 + 128), eval mode (EEG 128 + EOG 128 = 256, concat)
- budget: label25 (14 labeled subjects, 32310 labeled samples; unlabeled subjects used: 40, pseudo labels used: 34444)
- sampling_seed: 2024, downstream seed: 4
- evaluation: validation set only (8 subjects), official evaluate.py v1
