# C → B 第三周特征交付索引

2026-11-09 完成 std 冒烟核验及六组特征交付。两晚 1,863 段在 t=1 的 EEG std 输出与已发布的共同池化 NPZ 逐值一致。EEG/EOG 的 mean t=1 复用了小批次重提验证通过的完整缓存。EEG/EOG 的 std t=10、25 按 B 脚本重新提取。每个池化 NPZ 为 53,496 × 2,560、float32，含 32,310 段训练和 21,186 段开发样本。

[GitHub Release](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-b-w3-pooled-features-20261109) 提供六个原名池化 NPZ 和 SHA256 校验文件。下载到同一目录后，可运行 `python scripts/30_b_w3_finish.py --npz-dir <下载目录>` 核验文件并接续 B 的实验。

[交付清单](delivery_manifest.json) 记录六组文件和 SHA256，[冒烟结果](std_t1_smoke_summary.json) 记录参考 NPZ 核验。各目录包含样本索引、配置、资源记录、摘要与校验和。完整 mean 缓存在 C 本地。特征提取脚本使用固定的团队 main 提交 `8d21834`。
