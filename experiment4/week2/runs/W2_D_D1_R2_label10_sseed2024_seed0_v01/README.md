# W2_D_D1_R2_label10_sseed2024_seed0_v01

- status: success
- last successful step: metrics.json written
- method: D1 (self-training: dual-modal teacher pseudo-labels, per-predicted-class top-p confidence)
- representation: R2: R2_A_cnn_label10_seed0: A's EEG-only + EOG-only SmallCNN encoders (128 + 128), eval mode (EEG 128 + EOG 128 = 256, concat)
- budget: label10 (5 labeled subjects, 11936 labeled samples; unlabeled subjects used: 49, pseudo labels used: 24205)
- sampling_seed: 2024, downstream seed: 0
- evaluation: validation set only (8 subjects), official evaluate.py v1
