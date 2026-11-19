# 复核与登记

C直接交付的6份跨模态预测均为head seed0，在1s std、t=1、EOG-only及B五折协议下完成。四格12份预测另有独立目录。全部18份C预测入口为reports/week4/c_eval_registry_w4_all.csv。

从仓库根运行以下命令，复算6份预测、选轮、N1 AP、α调整和三seed补充表的来源哈希。

    python experiment3/week4/scripts/verify_crossmodal.py

B运行35号脚本时，可使用reports/week4/c_crossmodal_registry_w4.csv；所有预测路径相对仓库根。D的三head seed数据在external/d_three_head_seed/中作为引用材料，保留其原始登记表。该登记表指向D分支中的运行位置，C没有把D的18份预测计为自己的交付，也没有独立核验未随表入库的预测。

原始预测和OOF按字节保留。JSON/YAML中的本机路径改为仓库根相对路径或带SHA的本地特征占位符，原始文件SHA见source_artifact_sha256.json，发布文件SHA见artifact_sha256.json。模型权重与原始信号保存在原实验/Release位置，本次提交的是报告、预测、配置和审计材料。
