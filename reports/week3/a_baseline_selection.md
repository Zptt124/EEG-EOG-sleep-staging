# 实验A 第三周:额外基线选择与核查(ContraWR / ST-Transformer)

- 负责人/日期:A / 2026-11-08
- 依据:第三周工作文档(修订版 2026-11-06,`docs/week3/work_plan_W3.pdf`)实验A
  "选择并核查基线:核对现有 CNN 的配置和结果。检查 ContraWR 与 ST-Transformer 的
  实现、输入模态、划分和预处理,按可运行性选择一个额外模型完成主比较。另一个模型
  完成协议和实现核查。"
- 结论:**选择 ContraWR 完成主比较**;ST-Transformer(SleepTransformer)完成
  协议与实现核查(不可直接运行,详见下)。

## 1. 现有 CNN 配置与结果核对

- 现有基线 = 团队文档参考结构 SmallCNN([scripts/cnn_model.py](../../scripts/cnn_model.py),
  W2 v1.1,code commit f9ba515),dual 仅首层输入通道 1→2,86,437 参数;
  Adam lr=1e-3、batch=128、上限 60 epoch、验证 Macro-F1 早停 patience=10。
- 三档预算(5/14/54 人 = label10/25/100,名单 v2 seed2024)dual × seed0-4 的
  48 个 W2 run 已全部完成并通过 B 官方 evaluate.py --reference
  (run_registry.csv 50 行;无效 run 隔离在本地 `_w2_invalid_unseeded_init/`,
  不入登记)。第三周 18 格中 CNN 一侧(9 格)与 EOG 对照 6 格全部可复用,
  逐项复用核查见 [a_w3_reuse_check.md](a_w3_reuse_check.md)。

## 2. ContraWR(选定,完成主比较)

| 项 | 内容 |
| --- | --- |
| 论文 | Chaoqi Yang, Danica Xiao, M. Brandon Westover, Jimeng Sun, "Self-Supervised Electroencephalogram Representation Learning for Automatic Sleep Staging: Model Development and Evaluation Study", JMIR AI 2023;2(1):e46769(arXiv:2110.15278,DOI 10.2196/46769) |
| 上游仓库 | https://github.com/ycq091044/ContraWR ,固定 commit `217753a`(2023-06-02) |
| 输入模态 | **双通道**原始信号 [B,2,3000],100 Hz,30 s 段(V 单位,mne 读取);上游 Sleep-EDF 用 **Fpz-Cz + Pz-Oz 双 EEG**(EDF 前两通道,非 EOG);组内已核验 NPZ 只有 Fpz-Cz + 水平 EOG,按第三周"同样的分期输入"要求以组内 dual 代入(形状不变),该模态差异记录在案 |
| 预处理 | 段内 torch.stft(n_fft=256, hop=64, center=False, 矩形窗)→ log(|real|+1e-8) / log(|imag|+1e-8) 按通道拼接 → [4,129,43] 谱图;无滤波/加窗/裁剪(wake trimming 未做,上游 W 占 68.8%,其报告 accuracy 受此影响,数字不可直接对照) |
| 监督架构 | conv1(4→6) + ResBlock(6→8,stride2) + ResBlock(8→16,stride2+pool) + ResBlock(16→32,stride2+pool),flatten 128 → Linear(128,32)→(32,5);Dropout 0.5 ×3(上游固定,训练时生效);**29,511 参数**(重实现实测) |
| 上游训练配方 | Adam lr=2e-4、batch=128、100 epoch、单 seed=1234、90/5/5 随机划分、accuracy 观察 |
| 上游 Sleep-EDF 结果 | 论文报告在 SHHS/Sleep-EDF 上优于 MoCo/SimCLR/BYOL/SimSiam(自监督后下游);监督分支为对照 |
| 许可 | **上游仓库无 LICENSE 文件**;按团队规则("第三方代码须核查许可并保留来源")不逐字复制,做架构重实现并保留来源记录(见 scripts/contrawr_model.py 头注释与本表) |
| 可运行性 | PyTorch 纯 torch 依赖,本机(Python 3.13 / torch 2.13.0+cu126 / RTX 4060 8GB)当天可跑;形状自检通过(谱图 [4,129,43],展平 128,前后向+确定性 OK) |

**适配决定(记录在案,config.yaml 逐项登记):**

1. 划分/名单/标签:全部换用组内协议(54/8/16 划分 v1 seed42、名单 v2 seed2024、
   标签 0-4 M/? 剔除、N3+N4 合并)——上游 90/5/5 未播种随机划分**不用**;
   上游 '?'/M→Wake 的回退映射也不用(组内预处理已剔除)。
2. 输入通道:上游为 Fpz-Cz+Pz-Oz 双 EEG;组内 dual 为 Fpz-Cz+水平 EOG。
   按第三周"同样的分期输入和标签预算"的研究问题,用组内 dual 代入
   (同 [B,2,3000] 形状,模型无改动);文献模态差异记入第 4 节对照表。
3. 标准化:上游无标准化;按协议"标准化只拟合本预算有标签者"扩展为
   **每 (谱图分量, 频率 bin) 标量 z-score**(4×129 个统计量,训练侧拟合,
   验证用同一统计;normalization.json 记录方法与拟合对象)。
4. 训练规则与组内统一:种子在模型构造前设定、Adam、batch=128、上限 60 epoch、
   验证 Macro-F1 早停 patience=10、cudnn 确定性;lr 保留上游文献值 2e-4
   (CNN 为 1e-3,两者记录在 config.yaml/resource 表)。
5. 评估:一律 B 官方 `src/evaluation/evaluate.py --reference manifests/val_reference_v1.csv`。
6. 自监督预训练**不跑**(第三周协议为监督基线比较;上游 SSL 分支仅记录;
   若日后复现 SSL,按论文值 100 epoch/lr 2e-4/T=2 而非仓库 argparse 默认)。
7. 上限 60 epoch(上游 100)与早停 patience=10(上游无 Macro-F1 早停)为组内
   统一训练规则的一部分,记录于 config.yaml。

## 3. ST-Transformer / SleepTransformer(协议与实现核查,不运行)

| 项 | 内容 |
| --- | --- |
| 论文 | Huy Phan et al., "SleepTransformer: Automatic Sleep Staging With Interpretability and Uncertainty Quantification", IEEE TBME 69(8), 2022(arXiv:2105.11043) |
| 上游仓库 | https://github.com/pquochuy/SleepTransformer(注:常见写法 `pquochuy/st_transformer` 不存在;同名概念另见 Eldele et al. AttnSleep, TNSRE 2021,PyTorch——第三周文档所指按字面为 SleepTransformer,如有出入请组长确认) |
| 输入模态 | 单通道 EEG(Fpz-Cz),但**不读原始波形**:2 s 帧/1 s 步长、Hamming 窗、256 点 STFT → 29×128 log 幅度时频图(去 DC bin);序列上下文 L=21 段(seq-to-seq,对序列中每段出 5 类) |
| 预处理 | 100 Hz 重采样、100 抽头 FIR 0.3–40 Hz 零相位带通;MATLAB 脚本只给 SHHS(须 sleepdata.org 授权);无 Sleep-EDF 现成管线 |
| 架构/参数 | 两级 Transformer encoder(4 blocks, d=128, 8 heads, d_ff=1024, dropout 0.1)+ attention pooling + FC(128→1024→1024→5),约 3.8M 参数 |
| 上游配方 | Adam lr=1e-4、batch=32 序列、序列级 CE + L2 1e-4、无早停 patience 对应项(200 校验步)+ 1000 步下限 |
| 上游 Sleep-EDF 结果 | SleepEDF-78 十折 CV:从零 acc 81.4% / κ 0.743 / MF1 74.3(SHHS 预训练后 84.9%/0.789/78.8;N1 F1 48.5%) |
| 框架 | **TensorFlow 1.x**(placeholder/Session,TF2 不可运行),Python 3.7;数据准备与评估在 MATLAB;h5(MAT v7.3)文件 + txt 列表 |
| 许可 | LICENSE 文件 = MIT,GitHub API 报 MIT;README "License" 节写 CC-BY-NC-4.0(两处冲突;学术使用两者均可,商业需谨慎) |
| 可运行性 | **本机不可直接运行**:TF 1.x 不支持 Python 3.13 与 RTX 40 系 GPU;MATLAB/SHHS 授权另需。PyTorch 重实现估计 1–2 天(预处理 2-4 h + 模型 2-4 h + 数据/集成 1 天) |
| 协议冲突 | 序列上下文 L=21 段/逐段 seq-to-seq 预测,与组内 context_length=1 逐段协议、逐段预测表与 sample_set 核对不一致;纳入主比较需先与 B 对齐上下文与样本集规则 |

**核查结论:** SleepTransformer 作为文献参照可行(参数与预处理在论文/config.py 中
完整可复现),但 TF 1.x 实现是死路、移植成本 1–2 天且引入 21 段上下文协议变更;
按第三周"按可运行性选择一个"的要求,本轮只完成上述协议与实现核查,不运行。
若后续轮次需要,可按表中 PyTorch 移植步骤推进(单通道、29×128 谱图、
L=21 序列需全组先定上下文规则)。

## 4. 文献协议与资源成本对照(预填,实测值见 resource 表)

| 项 | 组内协议(W3) | ContraWR 文献 | SleepTransformer 文献 |
| --- | --- | --- | --- |
| 数据版本 | Sleep-EDF Expanded SC,153 夜已核验 | Sleep-EDF(SC) | SleepEDF-78 |
| 划分 | 54/8/16 受试者固定(seed42) | 90/5/5 随机(代码) | 10 折 CV |
| 通道 | Fpz-Cz EEG + 水平 EOG | 上游 Fpz-Cz + Pz-Oz 双 EEG(本组代入组内 dual,见第 2 节) | Fpz-Cz 单通道 |
| 输入 | 原始波形 [B,C,3000] | 原始波形→STFT 4×129×43 | 29×128 时频图 |
| wake trimming | ±30 min(第一周预处理) | 代码未做 | 未明示 |
| 标签预算 | 5/14/54 人嵌套名单 | 无(全量) | 无(全量) |
| 数据权限 | 监督只用预算内有标签者 | 全量 | 全量 |
| 标准化 | 每通道/每 bin 标量 z-score,训练侧拟合 | 无 | 每频 bin z-score,训练集拟合 |
| 优化 | Adam lr=1e-3(CNN)/2e-4(ContraWR) | Adam 2e-4 | Adam 1e-4 |
| 批大小 | 128 | 128 | 32 序列 |
| 选模 | 8 人验证 Macro-F1,patience=10 | accuracy | 校验步早停 |
| 种子 | 配对种子 0-2 | 1234 单种子 | 未报告多种子 |

资源实测(本机 RTX 4060 Laptop 8GB,同设备同 batch=128)见
`reports/week3/resource_profile.csv` 与 [a_w3_results.md](a_w3_results.md)。

## 5. 结论与待办

- [x] 核对现有 CNN 配置与结果(W2 48 run 全部官方评估通过)
- [x] ContraWR 实现核查(上游 commit/许可/输入/预处理/划分)并完成 PyTorch 重实现
- [x] ST-Transformer 实现与协议核查(TF1 死路 + L=21 上下文冲突,本轮不运行)
- [ ] 9 个 ContraWR run(label10/25/100 × seed0-2)与资源记录
- [ ] 18 格配对比较表、标签效率曲线、EOG 对照与逐段预测
