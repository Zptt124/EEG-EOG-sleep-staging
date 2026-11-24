# W4 实验 D：公共预训练脚本

全组所有 pretraining 变体（D 主实验、self-conditioning、A 的同窗/错时跨模态、B 的错时 EEG）都用这一套脚本，通过开关区分。改动请先给 D review。

## 1. 文件

| 文件 | 作用 |
|---|---|
| `src/w4d_common.py` | 路径（可用环境变量改）、模型尺寸、扩散/优化设置、名单与 SHA 检查、held-out 名单读取、`TrainPool`（缓存读取、按受试者剔除 held-out、取 EEG 条件窗） |
| `src/w4d_model.py` | `build_model()`、`w4_forward()`：官方 Wavenet + 两个可选开关（self-conditioning、cross-modal） |
| `src/w4d_train.py` | `Trainer`（训练一步、分模态 loss、断点状态）、`ValSet`（held-out 上的分段 val MSE） |
| `src/01_pretrain.py` | 主脚本：训练、每 5k 步 val MSE + 存 ckpt、停止规则、断点续训 |
| `src/04_extract_features.py` | 下游提特征（W3 配方；跨模态模型一律空条件；自动核对去掉 EEG 后 EOG 特征逐位不变）。用法见 HANDOFF 第 5 节 |
| `src/03_export_val_manifest.py` | 导出 val MSE 的固定窗清单（`manifests/d_val_windows_w4_v01.csv` + `d_val_w4_v01.json`），供 B 独立复算 |
| `src/00_build_pretrain_cache.py` | 组员在自己电脑上生成训练池缓存（54 人）；与 D 的缓存逐字节相同（已重建核对：SHA 一致） |
| `src/02_build_shift_pairs.py` | 生成错时（shifted）EEG 条件的配对表（C 的规则：同记录、间隔 60–300 秒、固定 seed 均匀抽一个） |
| `src/smoke_test.sh`、`src/smoke_test_v2.sh` | 冒烟测试（v1：中断续训、两个开关；v2：按 C 的规则改完后的跨模态） |
| `manifests/` | split、label_subsets_v2（SHA 与第三周相同）、缓存清单、错时配对表摘要、val 固定窗清单 |
| `HANDOFF.md` | 给 A、B 的交接说明（拿到后先做什么、各自的命令） |

与第三周的关系：模型尺寸、扩散与优化设置、预处理、缓存格式完全沿用第三周。已核对：不开开关时，`build_model('m64x12', seed=0)` 的初始化与第三周 `ckpt_step0.pt` **逐位相同**，`w4_forward` 与官方 `Wavenet.forward` 输出**逐位相同**。

## 2. 环境变量（组员在自己电脑上用）

| 变量 | 默认 | 含义 |
|---|---|---|
| `SLEEP_DATA` | `~/data/sleepedf_sc_processed` | A 的 153 个 NPZ |
| `EEGDM_VENDOR` | `~/week1/C_experiment3/vendor/EEGDM` | C 交付包里的官方 EEGDM 代码（commit `003b5a2`） |
| `SLEEP_PRETRAIN_CACHE` | `~/week3-1/cache/pretrain_train_v01` | 训练池缓存（54 人，`signals_f16.npy` 约 3.2 GB） |
| `SLEEP_W4_BIG` | `~/sleep_project/week4/big` | checkpoint 输出目录；错时配对表默认在 `shift_pairs/` 下 |

缓存的 SHA256 写死在 `w4d_common.py` 里（`baa935f7…`），和第三周 D 的缓存不一致会直接报错退出，保证全组用的是同一份数据。

## 3. held-out 与 val MSE（共同规则 1、2）

- **名单**：`--heldout-csv` 用 B 的正式名单 `manifests/pretrain_split_w4_v01.csv`（仓库根目录，PR #21，SHA256 `470c23f7…`），取 `pretrain_role == heldout_val` 的 8 人：SC414、SC416、SC431、SC450、SC457、SC463、SC467、SC477。
  - 规则（B 于 11-16 收紧）：从训练池中**不在 label50（27 人，含 label25、label10）里的 27 人**中分层抽 8 人。这样 5/14/27 三档预算者都留在预训练池里。D 已独立复现这份名单。
  - 脚本会检查：正好 8 人、都在训练池、都不在 label50；B 的格式还会核对其余 46 人都是 `pretrain_train`。不是这份正式名单（SHA 不同）时会打印警告。
  - 训练时把这 8 人整个剔除（54 → 46 人，114,041 段 / 1,368,492 个单元，与 B 的数字一致），**不重建缓存**，只在 `TrainPool.train_units()` 里按受试者过滤；名单文件名和 SHA256 写进每个 run 的 `config.json`。
- **val 集**（`ValSet`，全组正式口径，B 于 11-16 确认采用）：每个 held-out 受试者、每个模态固定抽 128 个 5 秒窗（每模态 1024 个）；每个窗在 10 个 t 段（[0,5)…[45,50)）里各评一次，段内 t、噪声都由 `VAL_SEED=20261114` 固定。选窗、t、噪声只取决于名单和 seed，**与模型、训练 seed、开关无关**，所以不同 run、不同 checkpoint 可以直接比。
  - 固定窗清单：`manifests/d_val_windows_w4_v01.csv`（2,048 行），配方和 SHA256 在 `manifests/d_val_w4_v01.json`：units `cb979d0b…`、t `a2734ffb…`、噪声 `3465eadc…`。每个 run 的 `config.json` 的 `val_set` 里有同样三个 SHA，用来核对。
- 每次 val 写 `val_mse.csv` 的行：`weights`（ema / online）× `eeg_cond`（0 = 不给 EEG；跨模态 run 另有 1 = 给 EEG），列为 `{eeg,eog}_t00_05 … t45_50`、`{eeg,eog}_all`、`{eeg,eog}_t05_25`、`stop_metric`。
- 训练 loss 也按模态分开：`history.csv` 的 `loss_eeg`、`loss_eog`（用于检查两种模态有没有"拔河"）。

## 4. 停止规则（D 于 2026-11-14 定稿，之后不改）

- **指标**：EMA 权重、不给 EEG 条件；EEG 和 EOG 各自取 t ∈ [5, 25) 的 val MSE 平均，再等权平均（`stop_metric`）。
- **频率**：每 5k 步一次（step 0 也评）。
- **停止**：连续 3 次验证（15k 步）都没有**严格低于**此前最好值，就停。
- **30k 之前不停**：保证全组共用的 30k 对照一定存在。（这是在默认规则上补的一条。）
- **上限**：240k。
- **保留**：最好的那一步复制成 `best.pt`；每次 val 的 `ckpt_step<N>.pt` 也全部保留（每个约 8 MB），后面做 probe 可以任选。
- 停止判断只依赖 `val_mse.csv`，续训重跑时可以完全复现。结果写进 `stop_decision.json`。
- 新方法筛查一律固定 30k：`--no-early-stop --max-steps 30000`。

## 5. 开关

### self-conditioning（`--self-cond-layer m --self-cond-pool {mean,std,meanstd}`）
按论文 arXiv 2505.10999（Xiang et al.）的 single-shot "conditioning residual" 实现：
- 取第 m 个 block（从 1 开始数）输出的残差流 h_m，在时间维上做无参数池化，得到 s_m；
- Δe = P(s_m) ⊙ φ(t)：P 是一层**零初始化**的 Linear，φ 是一个**单独的** timestep embedder（sinusoidal → Linear → SiLU → Linear）；
- Δe 加到第 m 个 block 之后所有 block 的全局 condition 上；
- 单次前向，训练和采样流程不变；零初始化保证第 0 步与不开开关时输出完全相同（已核对）。

**和论文不一样的地方**：
1. 论文只用 mean 池化；这里多给了 std 和 mean+std 选项，因为第三周发现 std 读出更好。默认是 mean。
2. 论文的 m 是用短训练"看哪一层早期 loss 最低"挑出来的（DiT-B 12 层取第 9 层，UNet 取 6–7 层）。m 在这里先作为参数，选法另定。
3. 论文里新增参数约占 1.3%；m64x12 太小，新增约 11.6 万参数（约 +11.6%，992,961 → 1,108,417），主要来自 φ。
4. 论文没提 stop-gradient 和随机丢弃，这里也都没加。

### cross-modal（`--cross-modal {none,same_window,shifted} --cond-dropout p --shift-pairs <npy>`）
按 C 的规则实现（`experiment3/week4/cross_modal_protocol.md`，2026-11-14）。

| 组 | `--cross-modal` | EEG 条件 | 负责人 |
|---|---|---|---|
| 同窗 EEG | `same_window` | 同一记录、起止时间完全一致的 Fpz-Cz 窗 | A |
| 错时 EEG | `shifted` | 查配对表：同一记录内起点间隔 60–300 秒（两个方向）的窗，固定 seed 均匀抽一个 | B |
| 无 EEG | `none` | 始终为空，直接用 D 主实验的 30k checkpoint | D |

- 只有 EOG 目标窗接收 EEG 条件；EEG 目标窗和 modality ID 都不变（EEG 目标窗一律是空条件）。
- **接口**：`w4_forward(model, x, t, modality, condition_eeg=None, condition_mask=None)`。`condition_eeg=None` 表示整批都是空条件；`condition_mask` 是批内逐窗的有效掩码（1 = 有条件，0 = 空条件）。三组用同一个前向入口。
- **空条件只有一个分支**：condition dropout、没有错时候选、推理和提特征时只给 EOG，走的都是同一个空分支（EEG 分支整体乘 0，flag = 0）。
- **EEG 条件编码器**（A、B 共用同一个）：EEG 窗经过 Linear(1,d)–ReLU–Linear(d,d)（最后一层零初始化）后，逐点加到输入嵌入上，与目标窗在时间上对齐；另有一个零初始化的"有没有条件"flag 嵌入，加到全局 condition 上，用来区分"空条件"和"EEG 全为 0"。
- **condition dropout**：逐窗进行，默认 `p=0.2`（C 的默认值），必须满足 `0 < p < 1`。
- **随机流分开**：打乱顺序、扩散噪声和 t、dropout、错时配对各用各的随机流。所以三组的目标窗、打乱顺序、噪声和 t **完全相同**。已核对：三组第 1 步的 loss 逐位相同。
- **错时配对表**（`02_build_shift_pairs.py`，seed 20261114）：
  - 按真实时间戳计算间隔（窗起点 = epoch_index×30 + 5k 秒，缓存里的缺口也算在内）；配对表覆盖全部 54 人，共 797,628 个窗位置。
  - 覆盖率 100%，没有找不到候选的窗。负向 399,131 个，正向 398,497 个，|Δ| 中位数 180 秒。|Δ| 刚好等于 60 秒的有 16,338 个，刚好等于 300 秒的有 16,281 个。
  - 每个方向按 |Δ| 分 4 段（[60,120)、[120,180)、[180,240)、[240,300]）：负向 98,280 / 97,970 / 97,318 / 105,563，正向 97,450 / 98,156 / 97,761 / 105,130。最后一段两端都包含，所以略多。
  - npy 的 SHA256 为 `2f446ba4…`；完整配对清单（目标窗 ID、条件窗 ID、带正负号的间隔）在 `shift_pairs_60_300_v01.csv.gz`，摘要在 `manifests/shift_pairs_60_300_v01_summary.json`。
- **登记**：`config.json` 记录名单 SHA、训练窗清单 SHA（清单本身是 big 目录下的 `train_units.npy`）、配对表 SHA；`checkpoints.csv` 记录每个 checkpoint 的 SHA256。
- val 的主口径是**不给 EEG**（`eeg_cond=0`），这样三组和其他 run 可以直接比；给 EEG 的结果另记为 `eeg_cond=1`（只有 EOG 目标窗带条件，错时组按配对表取 EEG）。
- C 规则里的验收条件（只给 EOG 和同时给 EEG 两种输入下，EOG 特征逐位一致）已在 `04_extract_features.py` 里落实，每次提特征时自动检查。

## 6. 用法

主实验（tmux 里跑）：
```bash
python -u 01_pretrain.py --run-name W4_D_main_m64x12_seed0_v01 --heldout-csv ../../../manifests/pretrain_split_w4_v01.csv
```
30k 筛查（例如 self-cond）：
```bash
python -u 01_pretrain.py --run-name W4_D_selfcond_l8_mean_30k_v01 --heldout-csv ../../../manifests/pretrain_split_w4_v01.csv --no-early-stop --max-steps 30000 --self-cond-layer 8
```
中断之后，用同一条命令重跑就会接着练；`config.json` 和原来不一致时会报错退出，防止参数被悄悄改掉。

启动时会检查 GPU：已被占用超过 2 GiB 就退出（本进程自己的 CUDA context 在 WSL 上约占 1.1 GiB）（`--allow-shared-gpu` 可以跳过这个检查）。显存上限设为 92%，超了会真正报 OOM，而不是悄悄溢出到共享内存。训练中如果 s/step 超过开头 5 次记录中位数的 3 倍，日志会打印 `WARNING SLOWDOWN`。

## 7. 冒烟测试

脚本是 `src/smoke_test.sh`（v1：中断续训、两个开关）和 `src/smoke_test_v2.sh`（v2：按 C 的规则改完后的跨模态、中断续训）。现在两个脚本默认都用 B 的正式名单（可以用环境变量 `HELDOUT_CSV` 换成别的）。v2 用了 `--smoke-small-val`（每人每模态只取 16 个窗），val MSE 不能和正式 run 比。日志留在 D 本地，没有入库。结果记在第 8 节。

## 8. 冒烟测试结果

**v1**（2026-11-14，RTX 4060 Laptop，D 自己抽的**暂定**名单：SC400、SC414、SC418、SC425、SC432、SC457、SC465、SC467；这份名单不符合后来收紧的 label50 规则，已删除），m64x12，seed 0。这些数字只能说明脚本能正常运行，**不能当实验结果用**。

| 检查项 | 结果 |
|---|---|
| held-out 剔除 | 训练 46 人，1,352,076 个单元（54 人时是 1,595,256） |
| 初始化 / 前向与第三周一致 | 逐位相同（不开开关；用第三周 30k 的 EMA 权重比对） |
| 开关零初始化 | 加载同一份权重后，开 self-cond（m=8）或 cross-modal，第 0 步输出与不开开关的差为 0 |
| loss 下降 | plain：0.442（第 50 步）→ 0.101（第 300 步）；EEG 0.411 → 0.089，EOG 0.475 → 0.113 |
| val MSE（EMA，stop_metric） | step 0：0.773 → 100：0.158 → 200：0.103 → 300：0.091 |
| 断点续训 | 在第 150 步强杀（`kill -9`）后续上；第 200、300 步的 val MSE 与不中断的那次**逐位相同** |
| self-cond（m=8，mean） | 参数 1,108,417；200 步正常，loss 与 plain 接近 |
| cross-modal shifted（v1 旧实现，p=0.5） | 参数 997,761；`eeg_cond=0/1` 两种 val 都能写出；给不给 EEG，val 差别很小（才训练 200 步，在意料之中）。这个实现已按 C 的规则替换，见下面 v2 |
| 停止规则的单元测试 | 15k 最好、之后连续 3 次没改善 → 在 30k 停；没到 30k 不停；和最好值相等不算改善 |
| 速度 | 0.51–0.60 s/step（第三周是 0.459 s/step；这次每 50 步记一次日志，测得不够精确）；每次 val 约 35 s × 变体数 |
| 存储 | 每个 `ckpt_step<N>.pt` 约 8 MB；`latest.pt` 约 22 MB（含 perm 和优化器状态） |

按 0.5 s/step 估算，主实验跑满 240k 约 33 小时，加上 48 次 val（每次约 1 分钟），总共约 34 小时。

**v2（按 C 的跨模态规则改完后，2026-11-14 17:35–17:43，`--smoke-small-val`，暂定名单）**

| 检查项 | 结果 |
|---|---|
| 三组随机流一致 | none / same_window / shifted 第 1、2 步的训练 loss **逐位相同**（0.541408、0.542272）；第 3 步才开始因 EEG 条件分支更新而出现差异（第 6 位小数） |
| 错时取窗 | `gather_eeg_cond` 取到的 EEG 窗与配对表指向的窗逐位一致（抽 3 个窗核对） |
| same_window、shifted（p=0.2） | 200 步正常；loss 0.442 → 0.113；`eeg_cond=0/1` 两种 val 都能写出 |
| 断点续训（含 dropout 随机流） | shifted 在第 150 步强杀后续上；history 前 6 列、第 100/200 步各项 val MSE 与不中断的那次**逐位相同** |
| 登记 | `checkpoints.csv` 正常写出。两次 run 的 checkpoint SHA 不一样，是因为文件里嵌了 run 名称；权重本身一致，这一点由 val MSE 逐位相同间接证明 |
| 小 val 用时 | 每次 val 13–16 秒（3 个变体），正式 val 约 1–2 分钟 |

**v3（接入 B 的正式名单，2026-11-17，`--smoke-small-val`，100 步）**

| 检查项 | 结果 |
|---|---|
| 读取 B 的 `pretrain_split_w4_v01.csv` | 取出 8 人，`official_w4_v01=True`，SHA 已写进 config |
| 训练池 | 46 人、1,368,492 个单元 = 114,041 段，与 B 的数字一致 |
| 错误名单会被拒绝 | 旧的暂定名单（含 label50 的 SC418、SC425、SC432、SC465）报错退出 |
| 训练 | loss 0.441 → 0.238（100 步），0.44 s/step |
