# E3_EEG_label25_sseed2025_seed0_v01

负责人：C（实验③）；状态：成功，验证集实验完成。

原始计算日期：100%及时间展开为2026-10-20；25%为2026-10-23。记录与正式评估整理日期：2026-10-23。

标签预算：25%；训练人数：14；训练段数：33775；验证：8人、21186段。

训练seed=0；名单抽样种子=2025；B名单与划分版本见config.yaml。

最佳第4轮，Macro-F1=0.7018091031，N1-F1=0.4144569560。最佳权重选择依据为验证Macro-F1。

主干固定为官方EMA权重，配置与SHA见config.yaml；分类头在CPU训练。E3上游代码commit=003b5a2a016b2f02788ea5693d8b7cd00b6bd969，A数据commit=02f120ee80c1c7a7822c33746ce92017632f1faa，B评估commit=6e7df9978c48c84bbef4d2f7e9d66e9ccd93b923。本机胶水脚本未提交Git，文件哈希见reports/final_week1/code_sha256.json。

文件：config.yaml、history.csv、best.ckpt、metrics.json、predictions_val.csv为团队约定格式；原run_config.json、history.json、best_head.pt及原summary.json保留。best.ckpt不是新训练权重，和best_head.pt字节一致。

Python3.11.17、PyTorch2.7.1+cu126；完整依赖见environment/requirements_freeze_wsl.txt。主干使用GPU，batch4，资源汇总见reports/final_week1/resource_summary.json；分类头计时见原summary.json。

原始运行与复现：进入experiment3后执行下方命令，复现必须选新输出目录，避免覆盖已完成结果。

```bash
python scripts/train_linear_budget.py --sampling-seed 2025 --output runs/reproduce_E3_EEG_label25_sseed2025_seed0_v01
```

正式评估命令（把RUN目录替换为对应复现输出目录）：

```bash
python vendor/experiment2-review-6e7df99/src/evaluation/evaluate.py runs/RUN/predictions_val.csv --out runs/RUN/metrics.json --bootstrap 1000 --seed 0
```

每组均保留原始验证预测；未使用测试集。验证集用于选模，只有单训练种子；EEGDM和CNN不是等架构/等预训练数据的严格对照。
