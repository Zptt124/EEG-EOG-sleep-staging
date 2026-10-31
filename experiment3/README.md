# 实验③：第一周档案与第二周交付

**2026-10-31 C三个补充实验已完成：** [结论与原始证据](week2/SUPPLEMENTS.md)包含固定50轮40头、类别加权40头、取层消融80新头，合计160份完整验证预测；全部审计通过。加权40/40配对的N1 F1改善，末5层有一般性分类收益，整体仍未证明稳定的适配收益。均为同一开发验证集上的探索性补充，测试集封存。

**C→B 特征下载补充（2026-10-31）：** [EEG/EOG 两份完整池化 NPZ](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-pooled-features-v01-20261031)已发布到 GitHub Release；各154124×2560、float32、未标准化，键为sample_id和x，附样本索引、SHA和B接口核验。EEG为原始冻结S0，EOG为C按D同协议本地重建。

2026-10-31：最新结果见[第二周交付入口](week2/README.md)。本地实验已完成，B/D团队验收待确认。下文保留第一周发布记录。

# 实验③：冻结EEGDM的EEG表征迁移

C负责的第一周实验已完成：输入适配、全量EEG特征、100%/25%线性分类、官方评估、资源核对及复盘。EOG与融合由D负责。

## 结果入口

| 验证集Macro-F1 | EEGDM＋线性头 | EEG CNN |
|---|---:|---:|
| 100%标签 | 74.97% | 76.45% |
| 25%，名单种子2024（主） | 69.71% | 71.71% |
| 25%，名单种子2025 | 70.18% | 71.58% |
| 25%，名单种子2026 | 72.07% | 69.15% |
| 三份25%名单均值 | 70.66% | 70.81% |

全部结果来自同一验证集8人、21186段；训练seed=0。N1-F1在三份25%名单中都高于CNN，但没有稳定的整体Macro-F1优势。主名单固定为2024，2026只是重复结果，不能替换主结果。测试集封存。

- [最终复盘、完成清单与两个表征假设](reports/final_week1/实验3_第一周最终复盘.md)
- [100%/25%正式评估及配对区间](reports/b_followup_20261023/实验3_少标签统一评估报告.md)
- [模型与输入理解笔记](reports/final_week1/EEGDM理解笔记.md)
- [运行索引](reports/final_week1/run_index.csv)
- [时间均值与时间展开对比](reports/time_pooling_comparison/时间池化对比报告.md)

## 给D的接口

完整特征为float16的`[N,20,30,128]`，按记录保存。使用`sample_id`与EOG取对应关系，按`feature_path`和`row_index`读取，核对标签和split；不能仅按文件行序拼接。

- [特征索引及完整提取元数据](manifests/eeg_cond14_v01)
- [提取配置](configs/feature_config.yaml)：200Hz、条件14、gate、20层、1秒std(correction=1)、t=1、不向输入添加随机噪声。
- [五个模型的验证预测](reports/predictions_E3)：每个CSV均为完整21186段、五类概率和真实/预测标签。
- [小体积运行记录](reports/runs_E3)：README、配置、逐轮日志、正式指标及完成记录。
- [大文件交接清单](manifests/external_artifacts.json)：完整特征约23.67GB，主干权重和分类头权重通过C的两个交付包提供；仓库中记录文件大小、模型哈希和包名。尚无对外发布的网盘分享链接。

官方代码、原始信号、大特征和模型权重不在本次新增提交中。根仓库原有A交付包不属于C本次新增内容。原`runs/`目录继续保存在本地；仓库内`reports/runs_E3`是可查阅的文本记录，不冒充包含模型权重的完整run。

30秒输入会被官方主干内部拆为6个5秒片段，不能视为连续30秒上下文。主方案只在分类头前对时间取均值，交付的完整特征保留30个时间位置。数据下标0是EEG，模型条件14是FP1-F3代理编号；提EOG需使用数据下标1并独立检查跨模态适配。

## 环境及只复核已发布结果

Python3.11。原环境为PyTorch2.7.1+cu126，主干使用RTX4060 Laptop，分类头CPU；依赖版本见[环境记录](environment/requirements_freeze_wsl.txt)。在仓库根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 pandas==1.5.3
python experiment3/scripts/prepare_sources.py
python experiment3/scripts/verify_published_results.py
```

`prepare_sources.py`获取B固定提交的评估脚本、名单和必要记录，不安装软件，不下载主干或特征。它将特征索引等元数据放到忽略目录；只有拿到真正完整的特征目录后，才能进行分类头训练。`verify_published_results.py`用B官方定义逐一复核五张预测表的ID、划分、标签、概率和保存指标，无需特征或权重。

## 重新提取或训练

在上述环境继续安装（CUDA安装方式需与自己的显卡匹配）：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r experiment3/environment/requirements-core.txt
python experiment3/scripts/prepare_sources.py --backbone
python experiment3/scripts/verify_backbone.py --device cuda --forward
```

主干源码固定到`003b5a2a016b2f02788ea5693d8b7cd00b6bd969`，来源[官方EEGDM](https://github.com/jhpuah/EEGDM)；权重由[官方Hugging Face页面](https://huggingface.co/jhpuah/eegdm)下载并按[来源记录](backbone_source.md)验证SHA256。本提交没有复制第三方源码或替它声明许可证；使用前查看上游条款。

A的原始预处理NPZ按根仓库交付包路径读取；不是z-score后的CNN输入。若没有这些文件，先从A取得同版本数据。然后：

```bash
python experiment3/scripts/extract_features.py --batch-size 4 --cache-kernels
python experiment3/scripts/train_linear100.py
python experiment3/scripts/run_b_followup.py
```

也可从C交付包将完整`features/eeg_cond14_v01`放入`experiment3/features/`，跳过主干提取。实际训练产物在`experiment3/runs/`（已忽略）。代码会拒绝覆盖已完成的同名run；重训可按各run说明选择新输出目录。

B分支未合并时，准备脚本从同一仓库提取固定提交`6e7df9978c48c84bbef4d2f7e9d66e9ccd93b923`，不会自行改动B脚本。A原数据版本为`02f120ee80c1c7a7822c33746ce92017632f1faa`，准备时核对A/B划分一致。

`reports/final_week1/code_sha256.json`是本地实际实验源代码的历史哈希；[发布代码记录](reports/published_code_provenance.json)同时保存本地与GitHub版本哈希，公开版本只将A路径改为仓库根目录、缓存改为用户缓存目录，保持实验算法。训练结果没有因上传重新训练或修改。

## 证据和限制

296段训练集小样本对三条件及两种调用长度的检查已保存；长度调用数值一致，三条件的分类排行只来自六名训练受试者的留一人小样本检查，不证明全量最优。五组正式模型的预测已从原始特征与最佳分类头重放；主干提取的显存、耗时、存储及半精度误差有记录。

验证集用于早停和配置比较；只有一个训练seed、三份25%名单，只有8名验证受试者。CNN与EEGDM的架构、外部数据、输入、优化器及设备不同，不能将现象归因为扩散目标，不能当作独立测试成绩。EOG迁移和融合没有在C范围内完成。
