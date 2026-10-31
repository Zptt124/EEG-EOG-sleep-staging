# EEGDM 来源记录

- 准备日期：2026-10-19（Asia/Shanghai）
- 官方代码：https://github.com/jhpuah/EEGDM
- 已下载 commit：003b5a2a016b2f02788ea5693d8b7cd00b6bd969
- 本地路径：vendor/EEGDM
- 权重页：https://huggingface.co/jhpuah/eegdm
- 下载地址：https://huggingface.co/jhpuah/eegdm/resolve/main/backbone.ckpt
- 本地文件：checkpoints/backbone.ckpt
- 实际大小：95,515,027 字节
- SHA256：0fe2d385d074834f3c22d9458798da49b2e5a58bd817284f130aae8814f5c1ef
- ZIP内部CRC检查：通过，1674个条目，没有损坏条目。
- GPU严格加载已通过：缺失和多余参数均为空，可训练参数为0；5秒与30秒随机输入前向均通过。结果见 reports/backbone_verification.json。

## 已核对的适配事项

- conf/cache/base.yaml 默认标签 minus_one 必须移除，睡眠标签已是0～4。
- 通道候选的零基编号：FP1-F3=14，FP2-F4=18，C3-CZ=10；见 src/preprocessing.py。
- 默认 diffusion_t=1，fwd_with_noise=null。src/caching.py 的 null 路径不加随机噪声；False路径调用 noiseless=True。不能把 t=1 写成已经随机加噪。
- 官方提取器将超过1000点的输入内部折叠成短段，6000点输入不等于连续30秒上下文。
- src/caching.py 验证循环命名使用 train_loader.dataset.files，而非 val_loader.dataset.files。后续独立提取脚本必须使用验证集自身索引，尚未修改上游仓库。
- 官方缓存主干取 diffusion_model.ema.ema_model；验证脚本严格加载该EMA参数。
