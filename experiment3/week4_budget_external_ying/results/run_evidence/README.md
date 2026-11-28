# C → B：Ying 原始运行证据

三份协议快照已在提交 [`c48160e`](https://github.com/Zptt124/EEG-EOG-sleep-staging/commit/c48160e) 归档。本次把它们与本地原件逐字节比对，并从原文件读取创建时间、最后修改时间和 SHA-256。时间见[原文件时间表](source_file_times.csv)，命令、退出状态和相关输出见[当时的终端记录摘录](terminal_excerpt.json)。

| 2026-11-28，北京时间 | 事件 | 证据 |
| --- | --- | --- |
| 01:16:50–01:17:07 | 调用 `confirm.py --freeze`，进程退出码为 0。初版协议文件的最后修改时间是 01:17:04。 | [对齐核查前原件](../protocol_before_alignment_review.json)、终端记录 |
| 01:18:54–01:19:02 | 保留初版，写入对齐修订，处理 `NotScored` 和 SLP010 第 1 晚的专家标注共同前缀。预算预检查无法为 SLP010 与 SLP003 抽到匹配的固定预算。 | [资格核查前原件](../protocol_before_eligibility_review.json)、终端输出 |
| 01:19:30–01:19:37 | 保留对齐修订版，写入训练资格门槛。前两晚各需至少 6 个可用块并覆盖至少两个时段。重新检查后确认 38 晚、39,505 个可评分片段。 | [最终协议原件](../protocol.json)、终端输出 |
| 01:16:03–01:32:43 | 38 个特征 NPZ 的本地最后修改时间范围。提取日志保留两次早期异常和重试，最终记录完成 38 晚。 | [特征时间与散列](source_file_times.csv)、[原始提取日志](extract.log) |
| 01:32:45–01:35:49 | 提特征结束后，排队命令启动正式拟合。先写入块表和输入锁，随后完成四折 480 次拟合。01:35:49 的终端轮询确认退出码为 0。 | [原始拟合日志](confirm.log)、[输入锁](../data_lock.json)、[结果](../summary.json)、终端记录 |

文件系统时间是在交接时从本地原件读出的元数据。终端时间来自当时的本地工具记录，日志正文用的是相对耗时。Git 记录可核对三版快照共同入库的时间。

- [实际运行版 `extract.py`](extract.py) 与原始提取器的 SHA-256 一致。[正式 `confirm.py`](../../confirm.py) 的 SHA-256 与输入锁一致。
- [原始下载日志](download.log)、[原始提取日志](extract.log) 和[原始拟合日志](confirm.log) 均按原字节归档。拟合日志末尾的 JSON 与结果文件一致。
- 38 个特征文件的 SHA-256 逐项与输入锁一致。时间表中 `archive_path` 为空的行对应这些特征文件。需取得冻结特征时，使用 [Ying 前置交接 Release](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/cd-ying-exp3-handoff-v01-20261128) 中的前置材料 ZIP。
