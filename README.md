# EEG–EOG 睡眠分期研究

第一周协作仓库：少标注基线、EEGDM 扩散表征迁移与跨模态融合。

**当前状态：A 的数据管线与两个小 CNN 基线已在分支 `feat/a-data-baseline` 落地**（EEG/EOG 验证集 Macro-F1 0.7645 / 0.6827，暂用 A 临时评估脚本，待 B 官方 evaluate.py 复核；详见 `reports/run_index.md` 与 `reports/结果报告.md`）。B/C/D 的实现与真实成绩尚未提交。

## 第一周目标与分工

| 成员 | 负责范围 | 优先交接 |
| --- | --- | --- |
| A | Sleep-EDF SC 数据、按人划分、预处理、小 CNN 的 EEG/EOG 基线 | 首先发布真实 `split_subjects.csv`，再交读取/训练代码及验证集预测 |
| B | 公共 `evaluate.py`、25% 标签子集、少标注 CNN | 先用假数据验证评估脚本；收到 A 划分后发布标签名单 |
| C | 冻结 EEGDM 主干、EEG 输入适配、特征提取、线性分类 | 先跑小样本；把通用提取脚本与设置交给 D |
| D | CNN 概率融合、EEGDM 的 EOG 探索和特征融合、N1 错误分析 | 先用假预测验证对齐/翻转统计，真实结果到达后分析 |

具体步骤见 [WEEK1.md](WEEK1.md)，交接要求见 [HANDOFF.md](HANDOFF.md)，协作方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 共同实验协议

- 第一周只使用 Sleep-EDF Expanded 的 SC 部分。EEG：Fpz-Cz；EOG：水平 EOG；基线按 100 Hz、30 秒片段处理。
- 按受试者划分训练/验证/测试，约 70%/10%/20%；以实际清单为准。同一人的所有夜晚必须属于同一集合。
- A 发布的 `split_subjects.csv` 是唯一依据，种子与版本写入记录。模板只含表头，不是真实划分。
- 标签：`0=W, 1=N1, 2=N2, 3=N3, 4=REM`；旧3/4期合并N3，Movement/Unknown按协议排除。
- 基线的归一化统计量只从训练集计算。本周只在验证集调参和分析，测试集封存。
- 主指标 Macro-F1；另存每类 precision/recall/F1/support、N1 指标、混淆矩阵（行真实、列预测）。
- 25%标签按训练受试者抽样，名单嵌套于100%；训练种子和抽样种子分开记录。
- EEGDM 输入缩放、重采样、通道条件、长度、取层及噪声设置必须单独记录并核查。EOG 迁移是待验证假设，不能预设有效。

## 现有文件

- `protocol.example.yaml`：待团队确认的配置模板，不能直接当作完成的训练配置。
- `predictions_val.template.csv`：统一验证集预测表头。
- `split_subjects.template.csv`：受试者划分表头。
- `RUN_TEMPLATE.md`：每次运行的记录模板。
- `文件清单.md`：全仓库每个文件的用途与收件人速查。
- `交付包/实验①交付包_20261018/`：A 的完整分发版（含预处理数据、运行记录、预测表，
  每个子文件夹带说明），供无网盘环境直接获取；团队开发仍按下方目录约定工作。
- `.gitignore`：排除原始数据、权重、缓存、运行大文件与密钥（交付包内成品除外）。

## 开发时采用的目录约定

成员提交实现时按以下路径新增文件；这些是约定，不表示已经实现：

```text
src/data/          # A：数据读取和预处理
src/models/        # A：SmallCNN；C/D：线性分类器
src/evaluation/    # B：公共指标
src/features/      # C：EEGDM适配；与D共享
src/fusion/        # D：融合与错误分析
scripts/          # 各实验运行入口
configs/          # 可复现配置（不用绝对私人路径）
manifests/        # 版本化划分与标签名单
reports/          # 小体积汇总和研究假设
data/             # 本地原始/处理数据，不入Git
checkpoints/      # 本地模型，不入Git
features/         # 本地特征缓存，不入Git
runs/             # 本地完整运行记录，不入Git
```

`runs/` 虽被Git忽略，仍须本地完整保存并通过团队约定的存储方式交接。小体积配置、指标及运行索引整理到 `reports/`，逐片段大表通过运行编号及存储路径交接。

## 开始协作

```bash
git clone https://github.com/Zptt124/eeg-eog-sleep-staging.git
cd eeg-eog-sleep-staging
git switch -c feat/a-data-baseline
```

私有仓库需要先获得访问权限。当前没有安装命令和训练命令；环境、依赖版本和运行入口由成员在实现后补充并验证。C 可单独维护 EEGDM 环境，避免把所有依赖混在一起。

## 资料

- 数据：https://physionet.org/content/sleep-edfx/1.0.0/
- EEGDM：https://github.com/jhpuah/EEGDM
- 权重：https://huggingface.co/jhpuah/eegdm
- 后续参考：https://github.com/Ben1001409/SleepDIFFormer

基于团队《睡眠分期课题·第一周工作文档》整理。本文规定协作流程，不声称文档中每项 EEGDM 技术细节已实测。引用外部代码前核查其许可，保留来源；当前未指定开源许可证。
