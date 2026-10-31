# W2_D_D2_R2_label10_sseed2024_seed0_v01

- status: success
- last successful step: metrics.json written
- method: D2 (cross-modal self-training: D1 candidates AND EEG-only and EOG-only teachers predict the same class)
- representation: R2: R2_A_cnn_label10_seed0: A's EEG-only + EOG-only SmallCNN encoders (128 + 128), eval mode (EEG 128 + EOG 128 = 256, concat)
- budget: label10 (5 labeled subjects, 11936 labeled samples; unlabeled subjects used: 49, pseudo labels used: 20570)
- sampling_seed: 2024, downstream seed: 0
- evaluation: validation set only (8 subjects), official evaluate.py v1
