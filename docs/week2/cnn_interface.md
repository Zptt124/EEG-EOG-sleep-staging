# cnn_interface.md — A 公平监督 CNN 编码器接口(W2 v1.1)

实验① 负责人 A · 交付对象:B(时间上下文)、D(跨模态半监督)
对应文档:第二周工作文档 v1.1 三.实验①;公共协议 [protocol_v2.md](protocol_v2.md)。
代码:`scripts/cnn_model.py`(唯一模型实现)、`scripts/train_cnn_w2.py`(统一训练入口)、
`scripts/export_encoder.py`(编码器导出/特征提取)、`scripts/load_encoder_example.py`(加载例子)。

## 1. 编码器来源(每个编码器必须可追溯)

| 字段 | 取值/说明 |
| --- | --- |
| 输入模态 | `eeg`(EEG Fpz-Cz)/ `eog`(EOG horizontal)/ `dual`(EEG+EOG 双通道,通道顺序 [EEG, EOG]) |
| 标签预算 | `label10`/`label25`/`label50`/`label100`(5/14/27/54 人,名单 seed2024,`manifests/label_subsets_v2.csv`) |
| 名单版本 | v2(每 run 记录 `subset_sha256`) |
| 训练种子 | 下游 seed 0–4;训练、采样、适配种子分列,不混写 |
| 代码/权重 SHA | `config.yaml` 的 `code_commit`;`encoder_export/SHA256SUMS.txt` 的权重与标准化 sha256 |

索引文件:`reports/week2/encoder_index.csv`(字段见 `reports/week2/README.md`)。

## 2. 数据规则

- 输入 `[B, C, 3000]`,`C=1` 单模态、`C=2` 双通道;**100 Hz**、30 秒一段;
  信号为 A 第一周已核验 NPZ(`signal` 键),单位 **µV**(第一周 mne 读数按 V 报告,预处理已 ×1e6)。
- 双通道两列来自同一 npz 行(同一 record_id、epoch_index),没有独立打乱。
- **标准化**:每通道标量 z-score `(x - mean) / std`,mean/std **只拟合本预算有标签者**,
  随编码器一起交付(`normalization.json`,含拟合对象名单)。使用时逐通道应用:
  ```python
  x = (x - mean[None, :, None]) / std[None, :, None]   # x: (B, C, 3000), mean/std: (C,)
  ```
- 原始数据不被修改;z-score 后的信号不能当作原始 µV 输入误用(协议v2)。

## 3. 输出规则

- 编码器 = `SmallCNN.features`(**去掉分类层**),逐段输出 **128 维**、dtype float32
  (第一周结构 GAP 前通道数 128;以代码检查结果为准,`model.encoder(x).shape == (B, 128)`)。
- 对应索引:`sample_id = <record_id>_<epoch_index:04d>`(30 秒网格序号,稳定不变);
  参考样本清单 `manifests/val_reference_v1.csv`。
- **前向必须 eval 模式 + no_grad**(无标签池前向防止 BatchNorm 统计被悄悄更新,协议v2)。

## 4. 加载例子(B/D 直接照用)

```python
import torch
from cnn_model import load_encoder          # 文件: scripts/cnn_model.py

model, norm = load_encoder("W2_A_EEG_label25_sseed2024_seed0_v01/encoder_export")
model.to("cuda")
# model 已 eval;norm["channels"] 含每通道 name/mean/std,fit_on 记录拟合对象

x = torch.randn(4, 1, 3000, device="cuda")  # 或 (4, 2, 3000) 双通道
with torch.no_grad():
    feats = model.encoder(x)                # (4, 128) float32
```

命令行批量提取逐段特征(自带 sample_id 索引):

```bash
python scripts/load_encoder_example.py \
    --encoder-dir <runs>/W2_A_EEG_label25_sseed2024_seed0_v01/encoder_export \
    --data-dir <A已核验NPZ目录> --records SC4061,SC4062 --out-feats feats.npz
```

## 5. 使用范围与注意事项

- 每个编码器**只用对应预算有标签者的标签训练**,不读其他训练者标签、不用无标签信号。
- **两单模态拼接**(EEG 编码器 + EOG 编码器各 128 维 → 256 维)与**双通道编码器**(128 维)
  是不同表征,不能混称;D 的 CNN 表征用前者,不要用 A 双通道编码器替代(W2 文档 实验④)。
- 内折验证/校准需要重训编码器(D 在自己设备上用 `scripts/train_cnn_w2.py` 完成,
  不在 A 的 50 次矩阵内);见过被留出者标签的编码器不能复用于该折(协议v2 第4条)。
- 100% 预算 seed0 的 EEG/EOG run 为第一周复用,逐项匹配证据见
  [a_reuse_check_w1.md](../../reports/week2/a_reuse_check_w1.md)。

## 6. 训练与导出入口

```bash
# 训练(六件套 + B 官方 metrics.json + resource.json)
python scripts/train_cnn_w2.py --modality dual --budget label25 --seed 0 \
    --data-dir <processed NPZ> --runs-dir <本地 runs 目录>
# D 内折重训:--subjects 覆盖 --budget 的名单(仍须全部在训练集,拼写错误会报错;
# --budget 仍必填,仅作标注;默认 run_id 带名单哈希避免折叠冲突;--owner D 可标注负责人)
python scripts/train_cnn_w2.py --modality eeg --budget label25 --seed 0 \
    --subjects SC403,SC410 --owner D --data-dir <processed NPZ> --runs-dir <本地 runs 目录>
# 导出编码器(权重 + 标准化 + SHA256SUMS)
python scripts/export_encoder.py --run-dir <runs>/W2_A_EEG_label25_sseed2024_seed0_v01
# 登记 run/资源/比较/编码器各表(--storage-prefix 必填,团队存储 URI)
python scripts/a_register_runs.py --storage-prefix team-storage://w2-a \
    --run-dir <runs>/W2_A_EEG_label25_sseed2024_seed0_v01
```

注:自定义名单(fold)run 的早停与选模仍按本脚本行为使用外部 8 人验证集;若 D 的
内折协议要求只在折内选模,需在 D 侧自行封装(本脚本保持单一行为,便于与主矩阵对照)。

训练固定设置:SmallCNN(团队文档结构,dual 仅首层输入通道 2)、Adam lr=1e-3、
batch=128、最多 60 epoch、验证 Macro-F1 早停 patience=10、cudnn 确定性模式。
