# B 独立核对：D 公共预训练脚本的 val MSE 与错时配对表

核对对象：PR #23（`feat/d-week4-pretrain-script`，commit 1be7a82）、Release `d-pretrain-w4-main30k-v01-20261117`。
脚本：[scripts/36_b_w4_check_d_pretrain.py](../../scripts/36_b_w4_check_d_pretrain.py)。结果文件在 [d_val_recheck/](d_val_recheck/)。

## 结论

- **val MSE 复算通过。** 7 个 checkpoint × EMA/online，共 14 行、每行 23 个数：
  - stop_metric 与 D 报的最大差 5.8e-6（相对 1e-4），所有分段最大差 3.9e-5（相对 5e-4）。
  - stop_metric 的最大差比相邻两次验证之间的最小变化（25k→30k，5.5e-4）小约 100 倍。B 的数选出的最佳步与 D 相同（30k），停止规则的判断不受影响。
- **val 集与 D 的一致**：训练池行序、2,048 个窗、t、窗清单 CSV、名单 SHA 都一致。只有噪声 SHA 不同，原因是 CPU 平台差异（见第 1 节）。
- **错时配对表通过**：重算结果与 D 训练用的 npy 逐字节一致；B 自己的检查逐条核对了规则，覆盖率 100%、两个方向各半、在候选里均匀抽取。
- 三组跨模态对照可以直接用 D 的脚本、val 集和配对表。

## 1. val 集重建

B 按 `d_val_w4_v01.json` 写的配方，用自己的代码从头重建，不读 D 的缓存：

| 项目 | 做法 | 结果 |
| --- | --- | --- |
| 训练池行序 | 从 A 的 NPZ 重建（split=train 记录按编号排序，段按 NPZ 顺序） | 132,938 行，与 D 的 `pretrain_epoch_index.csv` 逐行一致 |
| held-out 名单 | 读 B 的 `pretrain_split_w4_v01.csv` | SHA256 与 D 登记的一致 |
| 2,048 个单元编号 | `default_rng(20261114)`，按受试者、模态顺序各抽 128 个 | SHA256 一致；与 `d_val_windows_w4_v01.csv` 逐行一致 |
| t（2,048×10） | 同一个 rng，`5b + integers(0,5)` | SHA256 一致 |
| 噪声（2,048×10×1000） | `torch.Generator(cpu).manual_seed(20261114)` | **SHA256 不一致**，见下 |

**噪声 SHA 不一致的原因**：PyTorch 在 CPU 上生成 float32 正态随机数时，x86 走 AVX2 向量化的 Box-Muller，ARM 走通用实现，两者的 log/cos/sin 末位舍入不同。种子、形状和顺序都一样，差异只在浮点末位。B 的机器是 macOS arm64（torch 2.14.1），D 的是 x86 笔记本。下面 MSE 的差异是否只到这个量级，就是对这一判断的检验。

检验结果：step 0（输出层零初始化，预测恒为 0，MSE 只取决于 x0 和噪声）的所有分段与 D 相差不超过 4.6e-7。训练过的 checkpoint 最大差 3.9e-5，这部分还包含 CPU（B）与 GPU（D）前向计算的累加顺序差异。

## 2. val MSE 复算

- x0：NPZ → 200 Hz `resample_poly(2,1)` → µV/100 → 官方 staged mu-law → 取 float16 精度（与 D 的缓存一致）→ 第 k 个 5 秒窗。
- 前向：用官方 `Wavenet.forward`，不经过 D 的 `w4_forward`，不给 EEG 条件。
- checkpoint：Release zip SHA256 与 `artifact_index_D.csv` 一致，7 个 `.pt` 与 `checkpoints.csv` 一致。
- 比较：7 个 checkpoint × EMA/online，每个 22 个分段值加 stop_metric，与 D 的 `val_mse.csv` 逐格相减。

| step | 权重 | stop_metric（B） | stop_metric（D） | 23 个数的最大绝对差 |
| ---: | --- | ---: | ---: | ---: |
| 0 | ema | 0.772786 | 0.772786 | 4.6e-7 |
| 5k | ema | 0.062105 | 0.062108 | 2.6e-5 |
| 10k | ema | 0.057273 | 0.057275 | 3.9e-5 |
| 15k | ema | 0.055656 | 0.055655 | 3.3e-5 |
| 20k | ema | 0.054999 | 0.054999 | 3.0e-6 |
| 25k | ema | 0.054395 | 0.054398 | 7.1e-6 |
| 30k | ema | 0.053851 | 0.053852 | 4.7e-6 |

- online 权重 7 行的情况相同，stop_metric 最大差 6e-6，见 [val_mse_recheck.csv](d_val_recheck/val_mse_recheck.csv)（`b_` 开头的列是 B 的值，D 的值在她的 `val_mse.csv`）。
- 较大的差集中在 EOG 高 t 段（[35,50)）和两个模态的 [0,5) 段。停止指标用的 [5,25) 段差异最小。
- CPU（M 系列，torch 2.14.1）跑完 14 次约 60 分钟。

## 3. 错时配对表

- 用 D 的 `build_shift_pairs`，在 B 从 NPZ 重建的窗起点上重算，`np.save` 后的 SHA256 与 D 登记的 `sha256_npy` **一致**（`2f446ba4…`）。所以 D 训练时读的 npy 就是这份代码从这份行序算出来的。
- B 自己写的规则检查（不用 D 的代码），797,628 个窗全部通过：

| 检查 | 结果 |
| --- | --- |
| 覆盖率 | 100%（797,628 / 797,628） |
| 条件窗与目标窗同一记录 | 全部是 |
| \|Δ起点\| ∈ [60, 300] 秒 | 全部是 |
| 没配上的窗确实没有候选 | 没有未配上的窗 |
| 方向：条件窗在前 / 在后 | 399,131 / 398,497，二项检验 p = 0.48 |
| 在候选里是否均匀抽取 | 随机化 PIT 20 格卡方 p = 0.22，KS p = 0.94 |
| 候选数 | 最少 1，中位数 98 |

均匀性检验：对每个目标窗，把候选按时间排序，记下被选中的名次 r（候选数 n）。抽取均匀时 (r+U)/n（U 为独立均匀数）严格服从 U(0,1)，候选少的窗也成立。

- B 的 33 号脚本用的是另一种抽法（逐窗哈希），具体选中的窗和 D 的不同，规则相同。**训练以 D 的配对表为准**，33 号脚本只用于核对。

## 4. 为 A、B 的 30k run 准备：带 EEG 条件的复算

36 号脚本按 checkpoint 的 `run_config` 自动选择前向：
- 主实验没有开关，用官方 `Wavenet.forward`。
- 同窗组、错时组带 EEG 条件分支，用 D 的 `build_model` 和 `w4_forward`，因为这部分结构不在官方代码里。这类模型在「空条件」下也会加上一个训练过的 `xc_flag[0]` 嵌入，不能用官方前向代替。
- 这两组会多复算一行 `ema, eeg_cond=1`。条件窗由 B 从 NPZ 生成，只给 EOG 目标窗：同窗组取同一位置，错时组查 B 重建的配对表，脚本会核对这份表的 SHA 与 run_config 登记的是否一致。

**测试**：造了两个测试 checkpoint，主干用 D 30k 的 EMA 权重，条件分支随机赋值，让 EEG 条件真正起作用。在同一台机器上，用 D 的 `ValSet.evaluate` 算参考值，再和 36 号的结果比：

| 模式 | 条件 | stop_metric | 最大差 |
| --- | --- | ---: | ---: |
| shifted | 空条件 | 0.053637 | 4.9e-9 |
| shifted | 给 EEG | 0.054806 | 5.8e-9 |
| same_window | 空条件 | 0.053637 | 4.9e-9 |
| same_window | 给 EEG | 0.054804 | 4.8e-9 |

同一台机器上差异在 1e-8 以下。由此可以确认，第 2 节和 D 那 1e-5 量级的差来自平台：D 用 x86 加 GPU，B 用 ARM 加 CPU。

**附带发现**：在 B 的 Mac（arm64，scipy 1.18.0）上运行 D 的 `00_build_pretrain_cache.py`，生成的训练池行序 SHA 一致，但信号缓存 SHA 不一致（`0c4ff6d8…`，D 的是 `baa935f7…`），脚本会报错退出。原因与噪声相同：resample 和 mu-law 在 ARM 上的 float32 末位不同，存成 float16 时有少数值舍入到相邻值。B 不需要在本机训练，所以不影响 B。如果有组员在 Mac 上建缓存，会碰到这个报错。

## 复现

```bash
python scripts/36_b_w4_check_d_pretrain.py pairs --vendor <vendor/EEGDM> --d-week4 <experiment4/week4>
python scripts/36_b_w4_check_d_pretrain.py mse --vendor <vendor/EEGDM> --d-week4 <experiment4/week4> \
    --ckpt-dir <解压后的 w4_d_main_ckpt_0to30k_v01>
```

需要 A 的 NPZ。PR #23 合并前，`--d-week4` 指向 D 分支导出的 `experiment4/week4`。
