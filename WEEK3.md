# 第三周工作安排

[四天方案 PDF](docs/week3/work_plan_W3.pdf) · [第三周交接](HANDOFF.md#第三周交接)

本轮在 Sleep-EDF Expanded 的 Sleep Cassette 子集上完成四项跨受试者实验。使用 Fpz-Cz EEG、水平 EOG 和 30 秒分期 epoch。按[固定受试者划分](manifests/split_subjects.csv)使用 54 人训练、8 人开发验证、16 人封存测试。沿用[固定标签预算名单](manifests/label_subsets_v2.csv)，以 14 人为核心比较，使用 5 人复验，A 补 54 人全标签参考。预训练只读取训练池的无标签信号。选参、标准化和 checkpoint 选择使用训练侧数据，使用的分期标签计入对应预算。配置锁定后在开发集统一汇总。

主指标为 Macro-F1 和 N1 F1，补充 N1 precision/recall、各阶段 F1、混淆矩阵与逐人差值。以 `sample_id` 对齐预测，按受试者配对计算区间，记录开发集反复使用和样本规模的限制。运行需记录配置、名单、seed、模型来源和逐段预测。第二周继续适配和严格一致性筛选暂停追加。跨数据集预训练待成本明确后再排期。

## A 睡眠分期基线与标签效率

- 核验现有 CNN，检查 ContraWR 和 ST-Transformer 的实现与协议，选一个额外模型完成主比较。
- CNN 与新增基线在 5、14、54 人三档预算各跑 3 个配对 seeds，共 18 格。先完成 5/14 人，复用符合协议的既有结果。额外模型只支持 EEG 时，使用 EEG-only CNN 配对。
- 14 人预算下，用同一个 CNN 对比 EEG-only 与 EEG+EOG，复用主矩阵匹配的一侧，补另一侧 3 格。汇总 Macro-F1/N1 F1 标签效率曲线、N1/REM 错误及逐人差异。
- 交付基线协议、结果曲线、逐段预测和资源表。使用同设备和 batch size 测推理吞吐，记录训练时间、峰值显存和参数量。

## B 线性读出与统一评估

- 固定官方 checkpoint、t=1、通道条件与层组。比较每秒 std 和 mean pooling，两组均在 30 个时间位置上取均值，再训练相同 linear head。14 人和 5 人预算各跑 3 个配对 seeds，共 12 格。mean 特征从原始 activations 计算或重新提取。
- 用 14 人预算、std pooling 和单个 seed 探索 t=1 与两个较大的 t。核对官方总步数，主比较使用干净输入。追加加噪时单独登记 noise policy 和 noise seed。
- 按预算报告 N1 F1、precision、recall 和 AP，结合逐人结果分析 N1→W 与 N1→N2。14 人预算固定一个 seed，在训练名单内按受试者留出生成预测，选定 N1 分数缩放系数后固定用于开发集。原始分数的 AP 用于比较排序。
- 交付特征索引、pooling 对照、t 扫描、N1 分数调整对照和统一评估脚本。向 C 提供特征轴、索引和匹配的线性参考，向 D 提供固定线性探针。

## C Pooling 与 LFT 读出

- 接入官方 LFT。核对现有单模态缓存的 20 层 × 30 个时间位置 × 128 维，以及官方输入的各轴、层选择和位置编码。验证每个 30 秒 epoch 输出一组五分类分数。
- **C0** 为 temporal mean + linear head，**C1** 为保留 30 个时间位置的 LFT + classifier。两组使用同一官方 TUEV checkpoint、t=1、每秒 std、无随机输入加噪，以及相同 EEG/EOG 输入、层组和 14 人名单。首个配对跑通后补到 3 个配对 seeds。若统一取末层组，记录 EEG/EOG 的实际层选择。
- 交付输入核验、配对结果、逐段预测、N1/REM 错误及训练成本。先以 C1−C0 看 readout 收益，再以 A 的匹配基线看任务竞争力。主比较完成后，按资源在 mean + LFT 或 t=2/3 中择一扩展。

## D 从零扩散预训练

- 从训练池 EEG/EOG 各自切出 5 秒单通道片段，按模态条件编号共享小 backbone。保留合理深度并优先缩小 hidden size。记录采样、幅度缩放、staged mu-law、timestep sampling、noise schedule 和 denoising target。
- 先做稳态测速，再估算 50/100/200 个 training epochs。按四天内可用 GPU 时间固定更新数和 checkpoint，为第 4 天配对探针预留时段。若缩减数据，在训练受试者内部按固定规则抽取窗口，记录比例及实际样本数。
- 保存同一初始化的 random-init 与 pilot-pretrained 权重，冻结后使用相同模态、pooling、训练侧标准化和 linear head 配方，在 14 人预算下做配对。目标 3 个 head seeds，另记 backbone seed。单套预训练按 pilot 报告。
- 交付预训练配置、样本清单、测速与成本、短训 checkpoint 和配对探针，按实际更新步数说明结果范围。

## 四天执行安排

| 日期 | A / B | C / D |
| --- | --- | --- |
| 第 1 天 | A 选择额外 baseline、核查旧结果。B 锁定名单与接口，启动 mean 特征提取。 | C 核验 LFT 输入和训练流程。D 跑通小 backbone、测速并定预算。 |
| 第 2 天 | A 跑 5/14 人基线。B 做 14 人 std/mean 对照，准备 N1 分数调整。 | C 跑 14 人首个 linear/LFT 配对。D 启动固定预算短训。 |
| 第 3 天 | A 补 54 人参考与模态对照。B 做 5 人复验、有限 t 扫描和训练侧分数调整。 | C 补齐配对 seeds 并汇总 N1/REM 错误。D 保存短训权重并提取配对特征。 |
| 第 4 天 | A 汇总基线、错误与成本。B 完成 N1 分析和统一复评。 | C 核对同特征比较。D 完成随机权重/短训权重的配对探针。各组交结果表和简短结论。 |

如需共享 GPU，第 1 天根据实测耗时锁定时段。A 优先 5/14 人比较，B 优先 14 人 pooling 对照，C 优先 t=1 配对，D 为第 4 天探针预留时间。扩展任务按剩余预算安排。本文为计划，实际运行与接收状态以交付记录为准。
