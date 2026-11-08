# 两项追加验证：发布前复核

30项原始单元测试在本发布目录中通过。prediction_audit.json由verify_additional_validation.py使用本发布目录与待上传ZIP直接产生，逐文件SHA、96份最终预测、24份OOF合并、B固定v1点指标、36组配对bootstrap、314份分类器NPZ与9份EMA元数据/隔离检查通过。未重新训练或读原始信号，未读取封存测试。大池化特征数组不在发布包内，不能将本次复核说成已经重算所有特征或梯度。

原始文件与Release成员清单见../../additional_source_snapshot.json；当前入口见../../ADDITIONAL_VALIDATION.md。发布后GitHub端SHA、公开下载和main提交核验收据保存在操作者本地GitHub_additional_publication_20261102目录。
