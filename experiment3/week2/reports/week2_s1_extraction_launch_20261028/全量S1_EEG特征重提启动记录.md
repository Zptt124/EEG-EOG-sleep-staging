# 全量S1 EEG特征重提：已启动
+
+启动时间：2026-10-28T13:48:00+08:00。核验快照：2026-10-28T13:49:48+08:00（Asia/Shanghai）。
+
+**正在WSL后台运行，PID 397，已核验独立后台session。当前快照为2/122晚、1968/154124段；全量尚未完成。**
+
+实时进度见[S1 summary.json](../../features/eeg_cond14_S1_aseed0_steps10000_v01/summary.json)，每晚完成后更新。逐晚耗时和显存见[resource_profile.csv](../../features/eeg_cond14_S1_aseed0_steps10000_v01/resource_profile.csv)，控制台见[extraction_console.log](extraction_console.log)。本记录为启动快照，不替代实时状态。
+
+## 本次执行
+
+- 使用固定10000步的新EMA，权重SHA256：`4bf8d441e2c061db5662e42740d8ff742827c07062cb7965d7961ab1c502d385`。配置、旧/新提取代码、源索引和准备绑定的SHA已复核。
+- Batch=4，FP32推理、float16落盘，200Hz、条件14、t=1且不加随机输入噪声，保留20层gate的30个1秒std窗口，每段shape为`[20,30,128]`。
+- 只处理原训练/开发验证122晚，预期132938+21186=154124段，不处理测试信号。模型不接收睡眠分期标签；输出索引标签仅从已冻结的S0元数据原样复制。
+- S1写入独立新目录`features/eeg_cond14_S1_aseed0_steps10000_v01`，预计约23.67GB。S0目录保留，新模型冻结后建立自己的S4推理缓存，不复用旧S0核缓存。
+
+## 首晚落盘核验
+
+`SC4001`：841段，shape=`[841, 20, 30, 128]`，dtype=float16。完整文件有限性、文件SHA、权重来源、唯一sample_id和新版本索引均通过。S0配置与索引SHA仍与准备绑定一致。
+
+证据：[launch_audit.json](launch_audit.json)。按当时已完成两晚速度，剩余计算约81.1分钟；模型初始化、后续磁盘/文件校验和收尾会增加少量时间，实际速度以新日志为准。
+
+## 完成与恢复规则
+
+逐晚写`.partial.npy`，完成后原子改为正式`.npy`并写对应`.done.json`，记录文件SHA、权重SHA、shape、样本数、时间和显存。中断时保留已完成记录，未完成一晚会在恢复时重新提取；同权重/代码/配置才能恢复，已完成文件需重新核验SHA、shape和dtype。
+
+从experiment3目录恢复时使用原命令并加`--resume`：
+
+```bash
+/opt/eegdm/venv/bin/python -u scripts/extract_adapted_features.py --mode extract --config configs/features_W2_C_S1_pilot_v01.yaml --binding reports/week2_next_stage_20261028/extraction_binding.json --resume
+```
+
+当前进程运行中不要重复启动；排他锁会拒绝并发。准备绑定中的`ready_not_extracted`是不可改写的准备快照，实际状态以S1 summary为准。只有全部122晚/154124段以及与S0相同sample_id集合均核验通过，状态才写`complete`。
+
+本次没有启动池化缓存或下游分类头训练，未产生S0/S1分期性能结论。特征全部完成后，再进入按5/14人预算拟合标准化、五seed线性头及统一评价；双模态主比较仍需D原始EOG完整缓存。
+