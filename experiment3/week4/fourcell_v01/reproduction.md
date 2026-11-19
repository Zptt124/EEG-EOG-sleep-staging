# 复评与重跑

12 个 run 的仓库根相对路径在 [登记表](../../../reports/week4/c_eval_registry_w4.csv) 中。每个 run 提供开发集 11 列预测、指标、OOF、配置、按人指标；LFT 另含全名单重训历史。screening/ 保存三个 seed 的五折划分、50 轮曲线、选轮和 LFT 逐折历史。

从仓库根目录检查 12 份预测、选轮、训练侧 α 和文件 SHA。

```bash
python experiment3/week4/scripts/verify_fourcell.py
```

B 可将登记表交给其统一评估脚本。以已在 main 的 W3 评估器为例，--extra 接受相同的六列接口。

```bash
python scripts/29_b_w3_evaluate.py --extra reports/week4/c_eval_registry_w4.csv
```

脚本还会合并其默认运行目录的其他 run，输出位置沿用 B 原脚本。独立核验当前 12 份预测时，使用上方 verify_fourcell.py。

完整文字结果包和 12 份最终模型见 [Release](https://github.com/Zptt124/EEG-EOG-sleep-staging/releases/tag/c-fourcell-w4-v01-20261119)，SHA 见 release_artifacts.json 和 model_index.json。压缩包内路径以 experiment3/week4/fourcell_v01 或 runs/ 为前缀。开发集预测与本地完成版逐字节一致。为可搬移性，发布的 JSON 中路径已改为仓库根相对路径，原始文件 SHA 另有登记。

训练入口存于 experiment3/scripts/。使用 Linux/WSL、Python 3.11、torch 2.7.1+cu126、NumPy 1.26.4 和上游 EEGDM 003b5a2 的依赖。将上游代码置于 experiment3/vendor/EEGDM，并准备 A 的预处理 NPZ 及 D W3 代码/manifest 检出目录。D W3 权重包须匹配 56a506d04f6243c395f9fb5b557c70761f0f247c374c70d40dfc706b994d6122。

```bash
python experiment3/scripts/extract_week4_c_screening.py --d-checkout <D_W3_checkout> --archive <D_W3_checkpoint_zip> --output <cache>
python experiment3/scripts/run_week4_c_screening.py --b-checkout . --cache <cache> --output <training_root>/screening_1s_seed0 --seed 0 --pooling 1
# seed 1/2 分别使用各自 output 和 --seed。
python experiment3/scripts/run_week4_c_refit.py --b-checkout . --root <training_root> --output <training_root>/refit_1s_v01
```

特征缓存以 /features 放在 training_root 下，示例 cache 对应此目录。预处理 NPZ 沿用 A 交付包的 experiment3/vendor/experiment1-review/交付包/*/2_*/data/processed 布局。运行目录中的已有 plan、配置和哈希必须与输入匹配，恢复训练使用同一目录。所有从本机导出的计划位于 screening/ 下以供核对。最终评分脚本记录本次固定的 D W3 权重来源和开发集提取入口，重跑时应在新的输出目录执行，先锁定配置。

本次 LFT 重训在 seed0 完成后遇到线性训练遗留的全局 deterministic 开关，已恢复 LFT 原有设置后续跑。已交付源码包含修复。三个 head seed 共用一个 backbone seed，结果范围为五人预算、单一预训练初始化。
