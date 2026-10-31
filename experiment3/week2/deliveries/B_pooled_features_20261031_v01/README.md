# C → B：原始冻结 EEG/EOG 池化特征

两份 NPZ 均覆盖 **154124 段 × 2560 维**，未进行特征标准化。它们是 B 的共同冻结表征输入，不是三个适配 S1 表征。

## 下载文件

- [EEG：原始冻结 S0](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/c-pooled-features-v01-20261031/EEG_S0_pooled_154124x2560_unstandardized_v01.npz)
- [EOG：C 按 D 原始冻结协议在本地重建](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/c-pooled-features-v01-20261031/EOG_C_rebuilt_S0_pooled_154124x2560_unstandardized_v01.npz)
- [样本索引 sample_index.csv](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/c-pooled-features-v01-20261031/sample_index.csv)
- [文件与来源清单 manifest.json](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/c-pooled-features-v01-20261031/manifest.json)、[SHA256SUMS.txt](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/download/c-pooled-features-v01-20261031/SHA256SUMS.txt)

大文件放在 Release 附件，不包含在 Git 仓库源码 ZIP/普通 clone 内。此次补充解决此前只有索引、没有实际特征下载地址的问题。

## NPZ 内容

| 键 | 形状 | 类型 | 含义 |
| --- | --- | --- | --- |
| `sample_id` | `(154124,)` | NumPy Unicode 字符串，非 object | 与 A/B 原始片段 ID 一致且唯一 |
| `x` | `(154124, 2560)` | `float32` | 未标准化的池化特征 |

两份文件的 ID 和行序完全一致。池化操作为原始 float16 `[N,20,30,128]` 特征先转 float32，对 30 个时间窗取均值，再按层/隐藏维拉平；打包未重新提取或改变数值，解包结果与既有未标准化池化数组逐值一致。

覆盖训练 54 人/132938 段、开发验证 8 人/21186 段，不含 16 人测试集。`sample_index.csv` 包含 `sample_id,subject_id,record_id,epoch_index,split,y_true`，可直接作为 B 池化模式的 `--eeg-index`，EOG 共用此索引。训练仅使用所选预算允许的训练者，验证数据不能用于拟合标准化。

输入信号仍沿用既有 EEGDM 缩放、重采样与 mu-law 预处理；“未标准化”指输出特征未拟合或应用 z-score。来源原始 EMA SHA256 为 `0fe2d385d074834f3c22d9458798da49b2e5a58bd817284f130aae8814f5c1ef`。

EOG 文件由 C 在本地按 D 的相同原始冻结协议重建。D 原始完整缓存未收到，未声称与其逐字节相同；同一份本地 EOG 已用于 C 的双模态比较。本次上传不代表 B 已下载、接收或验收。

## 读取及接入

```python
import numpy as np

with np.load('EEG_S0_pooled_154124x2560_unstandardized_v01.npz', allow_pickle=False) as data:
    sample_id = data['sample_id']
    x = data['x']
    assert set(data.files) == {'sample_id', 'x'}
    assert x.shape == (154124, 2560) and x.dtype == np.float32
    assert sample_id.shape == (154124,)
```

B 的 `scripts/w2_common.py` 已支持该格式。下载两份 NPZ 和样本索引后，可使用其现有命令：

```bash
python scripts/12_eegdm_budget_head.py \
  --eeg-pooled EEG_S0_pooled_154124x2560_unstandardized_v01.npz \
  --eog-pooled EOG_C_rebuilt_S0_pooled_154124x2560_unstandardized_v01.npz \
  --eeg-index sample_index.csv --budget label10 --seed 0
```

SHA256 与精确字节数以 `manifest.json` 和 `SHA256SUMS.txt` 为准。文件已在本机解包校验完整形状、唯一 ID、有限值、行序和原始数值。
