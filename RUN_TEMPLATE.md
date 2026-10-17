# 运行记录（复制到本地runs/<run_id>/README.md）

- run_id：
- 负责人/日期：
- 状态：未开始 / 流程检查 / 未完成 / 成功 / 失败
- 实验问题：
- Git提交号与未提交修改说明：
- 数据版本、划分版本、预处理版本：
- 输入模态、模型、权重来源与hash：
- 标签预算与实际有标签人数：
- 训练种子/标签抽样种子：
- 完整命令、Python及依赖版本：
- GPU、峰值显存、耗时、缓存大小：
- 最佳权重选择规则：验证集Macro-F1
- 结果文件位置：
- 依赖的其他run_id：
- 初步观察与限制：
- 失败时最后成功步骤、报错与下一步：

本地保存config.yaml、history.csv、best.ckpt（若训练）、metrics.json、predictions_val.csv。metrics应包含各类support、precision/recall/F1和固定类别顺序。纯概率融合保存配方和源run，不需要伪造模型权重。未完成的实验不编造指标。
