# 交接登记

交付时在PR或团队沟通中写明commit、文件版本、run_id、存储位置及接收人。不要将聊天中的临时文件当作唯一版本。

| 优先级 | 交付人 | 交付物 | 接收人 | 状态 |
| --- | --- | --- | --- | --- |
| 最先 | A | split_subjects.csv | B/C/D | ✅ 已发布 v1(2026-10-18 00:52;种子42;54/8/16 按年龄分层),`manifests/split_subjects.csv` |
| 基础 | B | evaluate.py与假数据检查 | 全组 | 待实现 |
| 基础 | B | 25%名单、抽样种子、类别数 | A/C/D | 等A划分（已就绪） |
| 基础 | A | 预处理数据位置、读取与CNN训练代码 | B/C/D | ✅ 已交付:`scripts/`(01–05、common、metrics、download、evaluate 临时版);数据153个npz在交付包`2_预处理数据与脚本_给B和C\`,格式见 `reports/run_index.md` |
| 基础 | C | EEGDM提取脚本、输入设置 | D | 待验证 |
| 结果 | A/B/C | 逐段验证预测、指标、run_id | D | ✅ A部分:run_id `E1_EEG_label100_seed0_v01` / `E1_EOG_label100_seed0_v01`;预测表在交付包`3_验证集预测_给D\`(sha256见 `reports/run_index.md`);指标 `reports/metrics_E1_*.json` |
| 结果 | C | EEG特征索引与存储位置 | D | 待提取 |
| 复盘 | D | 融合、N1分析与假设 | 全组 | 待真实结果 |

受试者CSV中多条record_ids建议用分号连接，整字段按CSV规则转义。划分和预测模板只包含表头，禁止当作真实数据。
