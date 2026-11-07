# B 第三周：mean pooling 与 t 扫描特征提取（请 C 在 CUDA 机器上运行）

脚本：[scripts/26_b_extract_eegdm_w3.py](../../scripts/26_b_extract_eegdm_w3.py)。复用 C 的 `experiment3/week2/scripts/extract_features.py`（预处理、权重 SHA、kernel 缓存与数据路径），只把每秒 reducer 和 t 条件改成参数。不加随机噪声，t 只作为条件编号。

## 预先固定的设置

- 官方 checkpoint（SHA256 `0fe2d385…c1ef`），通道条件 14，query `gate`，20 层，6 个 5 秒窗口 × 每秒 1 个位置 = 30 个时间位置。
- 池化与 C 的共同 NPZ 一致：每秒特征存 float16，转 float32 后对 30 个位置取均值，拉平为 2560 维。
- 数据权限：只读 `label25`（sseed2024，14 人，含 `label10` 的 5 人）训练者与 8 名验证者，共 53,496 段，不读测试集。
- t 扫描：官方 `conf/pretrain/base.yaml` 为 `num_train_timesteps: 50`（squaredcos_cap_v2、v-prediction）。脚本运行时从 checkpoint 超参读出并校验。候选固定为 **t = 1、10、25**，std pooling，14 人预算，单 seed。

## 运行顺序（仓库根目录，Windows 用 `^` 续行）

1. 冒烟与逐值核对（约 1 分钟）。std、t=1 的前 2 个记录应与 C 的共同 NPZ 一致，`max_abs_diff` 应为 0 或在 float32 舍入级别：

   ```
   python scripts/26_b_extract_eegdm_w3.py --modality eeg --reduce std --diffusion-t 1 --record-limit 2 --cache-kernels ^
       --reference-npz <路径>/EEG_S0_pooled_154124x2560_unstandardized_v01.npz
   ```

2. 主对照 mean（各约 30 分钟；`--save-full` 额外保存 `[n,20,30,128]` float16，约 8 GB/模态，供 C 的 mean + LFT 扩展）：

   ```
   python scripts/26_b_extract_eegdm_w3.py --modality eeg --reduce mean --diffusion-t 1 --save-full --cache-kernels
   python scripts/26_b_extract_eegdm_w3.py --modality eog --reduce mean --diffusion-t 1 --save-full --cache-kernels
   ```

3. t 扫描（各约 30 分钟）：

   ```
   python scripts/26_b_extract_eegdm_w3.py --modality eeg --reduce std --diffusion-t 10 --cache-kernels
   python scripts/26_b_extract_eegdm_w3.py --modality eog --reduce std --diffusion-t 10 --cache-kernels
   python scripts/26_b_extract_eegdm_w3.py --modality eeg --reduce std --diffusion-t 25 --cache-kernels
   python scripts/26_b_extract_eegdm_w3.py --modality eog --reduce std --diffusion-t 25 --cache-kernels
   ```

中断后用同一命令重跑，已完成的记录（`<rec>.done.json`）会跳过。

## 回传给 B

每个输出目录 `features/b_w3_v01/<modality>_<reduce>_t<t>_cond14_label25/` 只需回传以下文件（每个 NPZ 约 550 MB）：

- `*_pooled_53496x2560_unstandardized.npz`
- `sample_index.csv`、`feature_config.json`、`resource_profile.csv`、`summary.json`、`SHA256SUMS.txt`

冒烟目录（`_smoke2` 结尾）回传 `summary.json` 即可。`--save-full` 的逐记录 `[n,20,30,128]` 缓存留在 C 本地。
