"""Step W4-A3: LFT 每轮耗时单独校准(五配置同口径)。

主对照(1s/2s)是 2 进程并行测时,提速方法②三个变体是 3 进程并行测时——并行度不同,
每轮耗时不可直接跨配置比。本脚本在 GPU 独占、无其他训练任务时,对五个配置各跑
2 个计时轮(同一份 label25 训练数据、同一 seed、同一配方),给出同口径的单独校准值。

用法(仓库根目录,确保无其他训练任务占用 GPU):
  python scripts/40_a_w4_timing_calib.py

输出:experiment3/week4/pool_compare/timing_calibration.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "experiment3/week3/scripts"))
sys.path.insert(0, str(REPO / "experiment3/week4/pool_compare/_compat"))
import run_week3_c_readout as c  # noqa: E402
import a_w4_readout as a  # noqa: E402

CONFIG = REPO / "experiment3/week4/pool_compare/configs/w4_lft_compare.json"
OUT = REPO / "experiment3/week4/pool_compare/timing_calibration.json"
CONFIGS = [  # (window_sec, signal_rate, no_classifier, tag)
    (1, 200, False, "1s"),
    (2, 200, False, "2s"),
    (3, 200, False, "3s"),
    (2, 200, True, "2s_nocls"),
    (2, 100, False, "2s_ds100"),
]


def calibrate(ws, sr, no_cls, device, timed_epochs):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config.update(window_sec=ws, signal_rate=sr, no_classifier=no_cls,
                  epochs=50, batch_size=32)
    n_pool = a.n_positions(ws)
    sources = a.w4_sources(ws, sr)
    rows = c.read_csv(sources["EEG"] / "feature_index.csv")
    subjects, train, dev = c.selected_positions(rows, c.read_csv(c.SUBSETS), "label25")
    data = a.W4CachedEpochs(rows, train, config["layers"], sources, n_pool)
    t0 = time.perf_counter()
    mean, std = c.train_statistics(data, config, device)
    norm_seconds = time.perf_counter() - t0
    model = a.w4_build_heads(config, 0, device, n_pool, no_cls)["C1"]
    optimizer = c.optimizer_for(model, config)
    rng = np.random.default_rng(0)
    epochs = []

    def on_batch(updates, samples, losses, epoch_seconds=None):
        pass

    for epoch in range(1, timed_epochs + 2):  # 第 1 轮为预热(内核/缓存编译),不计时
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        c.train_epoch({"C1": model}, {"C1": optimizer}, data, config, device, mean, std, rng, on_batch)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        if epoch > 1:
            epochs.append(dict(epoch=epoch - 1, seconds=round(elapsed, 3),
                               peak_allocated_mib=round(peak, 1)))
        print(json.dumps(dict(epoch=epoch, seconds=round(elapsed, 1)), ensure_ascii=False), flush=True)
    return dict(normalization_seconds=round(norm_seconds, 1),
                timed_epochs=epochs,
                parameters=sum(p.numel() for p in model.parameters()),
                n_time_positions=n_pool)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timed-epochs", type=int, default=2)
    args = p.parse_args()
    device = torch.device("cuda")
    c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    torch.set_num_threads(4)
    results = dict(device=str(device), torch=torch.__version__,
                   batch_size=32, timed_epochs=args.timed_epochs,
                   note="GPU 独占、同一 label25 训练数据、seed0;第 1 轮预热不计时",
                   configs={})
    for ws, sr, no_cls, tag in CONFIGS:
        print(f"=== {tag} ===", flush=True)
        results["configs"][tag] = calibrate(ws, sr, no_cls, device, args.timed_epochs)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
