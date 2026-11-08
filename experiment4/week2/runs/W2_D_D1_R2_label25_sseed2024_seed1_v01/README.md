# W2_D_D1_R2_label25_sseed2024_seed1_v01

- status: success
- last successful step: metrics.json written
- method: D1 (self-training: dual-modal teacher pseudo-labels, per-predicted-class top-p confidence)
- representation: R2: R2_A_cnn_label25_seed1: A's EEG-only + EOG-only SmallCNN encoders (128 + 128), eval mode (EEG 128 + EOG 128 = 256, concat)
- budget: label25 (14 labeled subjects, 32310 labeled samples; unlabeled subjects used: 40, pseudo labels used: 40252)
- sampling_seed: 2024, downstream seed: 1
- evaluation: validation set only (8 subjects), official evaluate.py v1
