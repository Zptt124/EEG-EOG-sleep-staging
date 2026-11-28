# W4 公共预训练脚本 · 交接说明（D → A、B）

这份说明讲拿到脚本后怎么跑起来。技术细节（停止规则、val 口径、开关的实现）见同目录的 `README.md`。

- **A**：同窗 EEG 30k（`--cross-modal same_window`）。
- **B**：错时 EEG 30k（`--cross-modal shifted`）。
- **无 EEG 对照**：用 D 主实验的 30k checkpoint，D 产出后发给大家。

三组的模型尺寸（m64x12）、backbone seed（0）、held-out 名单（B 的 `pretrain_split_w4_v01.csv`）、训练窗、打乱顺序、扩散噪声和 t 全部相同，只有 EEG 条件不同。所以**除了下面写到的参数，其他参数都保持默认，不要改**。

## 1. 准备（每台电脑做一次，约 10 分钟）

1. **环境**：与第三周相同的 `eegdm` conda 环境（torch 2.7.1、diffusers 0.33.1、ema_pytorch）。
2. **设置环境变量**（路径换成你电脑上的位置）：
   ```bash
   export SLEEP_DATA=<A 的 153 个 NPZ 所在目录>
   export EEGDM_VENDOR=<C 交付包里的 vendor/EEGDM>
   export SLEEP_PRETRAIN_CACHE=<放缓存的目录，约 3.2 GB>
   export SLEEP_W4_BIG=<放 checkpoint 的目录>
   ```
3. **生成训练池缓存**：
   ```bash
   cd experiment4/week4/src
   python -u 00_build_pretrain_cache.py
   ```
   最后打印 `SHA256 matches team reference` 就说明你的缓存和 D 的逐字节相同。不一致会报错退出，这时请先核对 NPZ 版本。
4. **只有 B 需要**：生成错时配对表（约 1 分钟）：
   ```bash
   python -u 02_build_shift_pairs.py
   ```
   核对打印的 `sha256_npy` 是否为 `2f446ba47673bf75db31a79ade98d4738f7aabe8ea6c3845c4e3a5c4ecb99ded`，`coverage` 是否为 1.0。

## 2. 冒烟测试（约 10 分钟，正式跑之前做一次）

```bash
bash smoke_test_v2.sh
```

默认读取仓库根目录的 `manifests/pretrain_split_w4_v01.csv`（B 的名单，PR #21）。看到 `SMOKE V2 ALL DONE` 就算通过。日志里 shifted 被强杀后续训的那一段，第 200 步的 val MSE 应该与不中断的那次完全相同。

## 3. 正式运行（30k，tmux 里跑）

先 `nvidia-smi` 确认 GPU 没有被别的程序占用。

**A：同窗 EEG**
```bash
python -u 01_pretrain.py --run-name W4_A_xmodal_same_30k_seed0_v01 \
    --heldout-csv ../../../manifests/pretrain_split_w4_v01.csv \
    --no-early-stop --max-steps 30000 --cross-modal same_window
```

**B：错时 EEG**
```bash
python -u 01_pretrain.py --run-name W4_B_xmodal_shifted_30k_seed0_v01 \
    --heldout-csv ../../../manifests/pretrain_split_w4_v01.csv \
    --no-early-stop --max-steps 30000 --cross-modal shifted
```

- `--cond-dropout` 默认是 0.2（C 的规则），不用另外写。
- 中断后用**同一条命令**重跑就会接着练。参数和上次不一致时会报错退出，这时不要换参数硬跑。
- 速度参考：D 的 RTX 4060 Laptop 约 0.44–0.55 s/step，30k 约 4–5 小时（含每 5k 步一次 val，每次约 2 分钟）。
- 如果日志出现 `WARNING SLOWDOWN`，通常是显存溢出到了共享内存，先查 `nvidia-smi`。

## 4. 跑完以后交什么

在 `exp_d_pretrain/runs/<run>/`（小文件）和 `$SLEEP_W4_BIG/runs/<run>/`（大文件）下：

| 文件 | 内容 | 交给 |
|---|---|---|
| `config.json` | 全部设置，以及名单、训练窗、配对表、val 集的 SHA256 | B（登记） |
| `val_mse.csv` | 每 5k 步的分段 val MSE；`eeg_cond=0` 是主口径（不给 EEG），`eeg_cond=1` 是给 EEG 的结果 | B（三组汇总） |
| `history.csv` | 训练 loss（含 `loss_eeg`、`loss_eog`）、速度 | B |
| `checkpoints.csv` + `ckpt_step30000.pt` | 30k 权重（online + EMA）及 SHA256；下游用 EMA | 自己做 EOG-only probe，并交给 B、C |

## 5. 下游提特征

下游一律只输入 EOG、不给 EEG。这一步**不能**直接用官方的 `LatentActivityExtractor`：它不会经过跨模态模型新增的"有没有条件"那一项，提出来的特征会和训练时的空条件不一致。D 会另外提供 W4 的提特征脚本，用同一个前向入口、空条件，并按 C 的验收要求核对"只给 EOG"和"同时给 EEG"两种输入下的 EOG 特征是否一致。在那之前请先不要提特征。

## 6. 修改代码

所有 pretraining 变体都用这一套脚本。需要改的地方，请先告诉 D，经 review 后统一改，不要各自改出不同的版本。

## 7. 已知限制

- val MSE 的全组口径就是这套脚本（B 于 11-16 确认）。固定窗清单在 `manifests/d_val_windows_w4_v01.csv`，供 B 的 32 号脚本独立复算。
- self-conditioning 开关是 D 自己的筛查用的，A、B 不需要用。
