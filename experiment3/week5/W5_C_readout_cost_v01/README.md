# W5 C 全层与后三层读出开销

普通 W4 30k EMA 用同一批开发集信号实测。每名 dev 受试者均匀抽 16 段，共 128 段。RTX 4060 Laptop GPU 上每次处理 8 段，EEG、EOG 逐模态前向，预热后交错测量两种读出，各 64 组。GPU 数字为双模态耗时中位数。输入提前传入 GPU，以 CUDA event 分开测 backbone 前向（含收集 gate）和 `std` pooling。两种读出都先算完 12 层，第 10–12 层方案只对选定层做 pooling。两种读出的后三层数值逐元素一致，对照已发布特征的最大绝对误差是 `0.000171`。封存的 16 人测试集未使用。

| 分项 | 全 12 层 | 第 10–12 层 | 测量口径 |
| --- | ---: | ---: | --- |
| Backbone 前向及 gate 收集 | 149.04 ms | 149.26 ms | 普通 30k，GPU，8 段 EEG+EOG |
| 特征 pooling | 1.39 ms | 0.39 ms | 普通 30k，GPU，8 段 EEG+EOG |
| 特征传回 CPU | 0.381 ms | 0.379 ms | 普通 30k，8 段 EEG+EOG |
| 双模态缓存大小 | 333.39 MB | 86.88 MB | 普通及 self-cond 各自的全部 53,496 段，未压缩 NPZ，含 ID |
| 缓存写盘 | 3.45 s | 1.03 s | 普通 30k，3 次中位数，含 `fsync` |
| 缓存写盘 | 3.50 s | 1.03 s | Self-cond 30k，3 次中位数，含 `fsync` |
| CPU linear head 前向 | 580 µs | 160 µs | 普通 30k，512 段、4 CPU 线程，复用四格实验测量 |
| CPU linear head 前向 | 560 µs | 155 µs | Self-cond 30k，512 段、4 CPU 线程，复用四格实验测量 |

普通 30k 的 backbone 前向时间基本相同。已保存的后三层特征减少约 74% 的缓存体积，写盘时间缩短约 70%。GPU 与 CPU 计时的批大小不同，分别报告。GPU 计时使用本机 PyTorch 回退 kernel，绝对耗时对应本机环境。

四格结果支持先用**普通 30k 的第 10–12 层**作为节省缓存的默认读出。相对普通全层，Macro-F1 配对差在 5 人档为 −0.25 个百分点，95% CI [−2.79, +2.82]，在 14 人档为 +0.54，95% CI [−0.81, +1.51]。目前是固定 backbone seed 的 dev 对照。[四格性能与配对结果](../W5_C_30k_fourcell_v01/README.md)

Self-cond Release 目前提供已做 30 秒内平均的分层特征。要在同一 self-cond 30k 上执行 GPU 前向计时，需对应 EMA checkpoint。继续做同一 checkpoint 的逐秒 LFT，则需要该 checkpoint 或保留 30 个逐秒位置的分层特征。已有 LFT 计时来自 W3 普通模型。[W3 普通模型 LFT 对照](../W5_C_lft_layer_screen_5ep_v01/README.md)

[GPU 时延分位数](gpu_summary.csv) · [GPU 逐批明细](gpu_batches.csv) · [写盘明细](cache_writes.csv) · [核验记录](verification.json)
