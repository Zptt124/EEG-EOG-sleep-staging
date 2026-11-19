# 实验 C · Week 4 已完成结果

完成四格实验和跨模态三组判读，封存测试集未使用。

| 产物 | 内容 | 入口 |
| --- | --- | --- |
| 四格实验 | linear/LFT × 随机/预训练30k，5人预算、head seed0/1/2，共12个run | [结果报告](fourcell_v01/README.md) |
| 跨模态判读 | C统一前向与环境下的3组×2预算、head seed0，共6个run，补充D的三head seed表 | [结果报告](crossmodal_final_v01/README.md) |
| EEG/EOG诊断 | 对A的46人、20,000窗作按人配对复算 | [诊断报告](crossmodal_diagnostics_v01/README.md) |

跨模态主指标N1 AP的配对区间均跨零。D的三head seed汇总中，5人档同窗−无EEG的Macro-F1为−0.63 [−1.11,−0.07] 个百分点，方向为下降。该结果仍只有单个backbone seed，当前未确认稳定跨模态收益。

[全部18个C run登记](../../reports/week4/c_eval_registry_w4_all.csv)供B复评。其中四格12个与C独立跨模态6个均含实际预测；D的18份三seed探针作为引用汇总材料单列。
