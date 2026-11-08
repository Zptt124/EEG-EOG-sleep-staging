# 第三周执行清单(实验A 视角)

依据[第三周工作文档](docs/week3/work_plan_W3_v1.pdf)(修订版 2026-11-06,四天实验安排)。
本文件先记录 A(实验A)的推进状态;B/C/D 状态由各成员补齐。

## 实验A:睡眠分期基线与标签效率

| 步骤 | 内容 | 状态 |
| --- | --- | --- |
| A1 | 核对现有 CNN 配置与结果 | 完成:W2 48 run 全部官方评估通过;无效 run 已隔离 |
| A2 | 核查 ContraWR 与 ST-Transformer 实现/模态/划分/预处理,按可运行性选一个 | 完成:选 ContraWR;ST-Transformer 协议与实现核查见 [reports/week3/a_baseline_selection.md](reports/week3/a_baseline_selection.md) |
| A3 | 5/14/54 人 CNN vs ContraWR 配对比较(每种方法 3 配对 seeds,18 格) | 进行中:CNN 侧 9 格复用 W2([核查](reports/week3/a_w3_reuse_check.md));ContraWR 9 格训练 |
| A4 | 14 人 EEG-only vs EEG+EOG 对照(3 配对 seeds) | 复用 W2 双侧(0 新 run),见 [a_w3_reuse_check.md](reports/week3/a_w3_reuse_check.md) |
| A5 | 5/14/54 人 Macro-F1 与 N1 F1 曲线 | [budget_curves_w3.png](reports/week3/budget_curves_w3.png) |
| A6 | 文献协议与资源成本表 | 见 [a_baseline_selection.md](reports/week3/a_baseline_selection.md) 第 4 节 + [resource_profile.csv](reports/week3/resource_profile.csv) |
| A7 | 结果表与逐段预测 | [a_w3_results.md](reports/week3/a_w3_results.md)、run 六件套(Release [a-contrawr-w3-v01-20261108](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/a-contrawr-w3-v01-20261108)) |

## 关键记录

- ContraWR 移植:[scripts/contrawr_model.py](scripts/contrawr_model.py)(上游重实现,许可与来源见文件头)
- 训练入口:[scripts/train_contrawr_w3.py](scripts/train_contrawr_w3.py)
- 复用核查:[scripts/a_w3_reuse_check.py](scripts/a_w3_reuse_check.py) → [a_w3_reuse_check.md](reports/week3/a_w3_reuse_check.md)
- 汇总:[scripts/a_w3_results.py](scripts/a_w3_results.py) → reports/week3/*.csv + png
- 指标口径:B 官方 [src/evaluation/evaluate.py](src/evaluation/evaluate.py) --reference manifests/val_reference_v1.csv

## 与 B/C/D 的交接

- B → A:B 官方评估器与 sample_set 核对已沿用(复用核查通过)。
- A → B:第三周 18 格预测与登记表见 reports/week3/;W2 四档曲线 CNN 部分已包含在
  A 主矩阵(5/14/54 三档;27 人 label50 未列入第三周矩阵,如需 W2 四档曲线
  可直接用 W2 label50 结果)。
- 上游文献口径差异(ContraWR 用 Fpz-Cz+Pz-Oz、无 wake trimming、90/5/5 随机划分)
  见 [a_baseline_selection.md](reports/week3/a_baseline_selection.md),不混入组内比较数字。
