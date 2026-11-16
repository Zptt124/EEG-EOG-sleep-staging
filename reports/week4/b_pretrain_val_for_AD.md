# B → A、D：第四周预训练 held-out 名单与分段 MSE

第四周第 1 天交付。新一轮小 EEGDM 预训练（D 主线、A 同窗 EEG、B 错时 EEG）都按这份名单划分训练池，用同一个脚本报告验证 MSE。

## 1. held-out 名单（已锁定，`w4_v01`）

| 层 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 受试者 | SC414 | SC416 | SC431 | SC450 | SC457 | SC463 | SC467 | SC477 |

- **规则**：`split=train` 共 54 人，去掉 `label_subsets_v2`（sampling_seed 2024）所有标签预算名单里的人。三档名单是嵌套的（5 ⊂ 14 ⊂ 27），所以去掉的就是 label50 的 27 人，剩 27 人。
  - 协议最低要求只是排除 14 人。这里把 27 人档也排除，三档预算者就都留在预训练池里。以后在第四周 backbone 上画 5/14/27 预算曲线时，每档条件一致。
  - 这 27 人按编号排序，用 `np.array_split` 分成 8 层（4,4,4,3,3,3,3,3 人），每层用 `default_rng(2026)` 依次抽 1 人。全程不读标签，也不看模型结果。
- **预训练池**：剩下 46 人、90 晚、114,041 段。留出的 8 人共 18,897 段，两者合计等于第三周 D 的 132,938 段。开发集 8 人和测试集 16 人照旧不进入预训练。
- 8 名留出者都有两晚记录。
- 文件：
  - [manifests/pretrain_split_w4_v01.csv](../../manifests/pretrain_split_w4_v01.csv)：54 人逐行列出，`pretrain_role` 为 `pretrain_train` 或 `heldout_val`。
  - [manifests/pretrain_val_w4_v01.json](../../manifests/pretrain_val_w4_v01.json)：规则、种子、计数和各文件 SHA256。
- **D 建缓存时**：只保留 `pretrain_role == pretrain_train` 的记录，并在 config 里登记 `pretrain_split_w4_v01.csv` 的 SHA256：

```python
import csv
pool = {rec for r in csv.DictReader(open('manifests/pretrain_split_w4_v01.csv'))
        if r['pretrain_role'] == 'pretrain_train' for rec in r['record_ids'].split(';')}
```

## 2. 固定验证窗

[manifests/pretrain_val_windows_w4_v01.csv](../../manifests/pretrain_val_windows_w4_v01.csv) 共 4,000 行（每行一个窗×模态）：

- 每名留出者抽 250 个 5 秒窗位置，EEG、EOG 各评估一次。窗位置用 `record_id`、`epoch_index` 和 `window_k` 表示，`window_k` 是 30 秒段里的第几个 5 秒窗（0–5）。
- 两个模态共用同一位置和同一个 t，噪声各自独立。每名留出者内，t=0…49 每个值各出现 5 次，所以每个模态×段有 200 个窗。
- 预处理和第三周 D 的 `transform_uv` 一致：100→200 Hz `resample_poly(2,1)`，µV/100，再做官方 staged mu-law。变换后取 `[1000k, 1000(k+1))`。
- 噪声不入库，由 `torch.Generator(cpu).manual_seed(20261117)` 按行序生成，脚本会核对 SHA256。

## 3. 分段 MSE 脚本

[scripts/32_b_w4_pretrain_mse.py](../../scripts/32_b_w4_pretrain_mse.py)：v-prediction MSE，每 5 个扩散步一段（10 段），EEG/EOG 分开报告，另给全 t 平均。默认用 EMA 权重。可以一次传入多个 checkpoint 比较停止点。

```bash
python scripts/32_b_w4_pretrain_mse.py --vendor <vendor/EEGDM> \
    --ckpt step0=<dir>/step0.pt --ckpt step15k=<dir>/step15000.pt --ckpt step30k=<dir>/step30000.pt \
    --out reports/week4/pretrain_mse/<run_name>
```

- **checkpoint 格式**：直接读第三周 `Trainer.weights()` 保存的 dict（`model_kwargs`、`model`、`ema_model`、`step`）。其他格式用 `--model-kwargs`、`--weights-key` 补充。
- **条件状态**：默认按空条件评估，即 `net(x_t, t, modality)`，跟下游 EOG-only 的推理方式一致，三组先统一按这种方式比较。
  - D 的公共脚本如果给前向加了条件参数，用 `--forward adapter.py:fn` 接上，函数签名是 `fn(net, x_t, t, modality, rows)`，`rows` 是该 batch 的窗表行。
  - 带 EEG 条件的 MSE 另外加 `--condition-tag sync` 或 `--condition-tag shifted`，单独成表，不和空条件混在一起。
- **输出**：
  - `mse_by_segment.csv`：主表，列出 n_windows、均值，以及 8 人人均值的标准误。
  - `mse_by_subject.csv`、`mse_by_window.csv`。
  - `mse_run.json`：checkpoint SHA256、窗表和噪声 SHA256、设备与版本。
- **自检**：`--self-test` 用随机初始化的 m64x12 跑两遍，逐窗 MSE 必须完全一致。
  - B 在 CPU 上实测两遍逐窗最大差为 0，单个 checkpoint 约 50 秒，GPU 上应该更快。
  - 随机初始化的全 t 平均 MSE 为 EEG 0.502、EOG 0.533。官方输出层零初始化，预测为 0，所以这里的 MSE 等于 E[v²]，与第三周 D 在训练池 512 窗上的初始诊断 loss 0.542 同一量级。

## 复现与核对

```bash
python scripts/31_b_w4_pretrain_heldout.py --check   # 需要 A 的 153 个 NPZ；三份文件逐字节一致即通过
pytest tests/test_w4_pretrain_heldout.py
```

名单如需改动，升级为 `w4_v02` 并通知全组，不覆盖 `w4_v01`。
