"""Step W4-A2: 1 秒 vs 2 秒 std pooling 的 LFT 对照(训练侧五折选轮)。

基于 C 的 experiment3/week3/scripts/run_week3_c_inner_cv.py,协议完全相同:
  1. 预算内 14 位有标签训练者按受试者分 5 折(B 的 subject_folds,fold_seed=0,与窗口无关);
  2. 每折标准化只拟合折内训练者,训练固定 50 轮,每轮对留出受试者预测;
  3. 拼接折外预测取 OOF Macro-F1 最高的轮次 E*(并列取最早);
  4. 全部 14 人重训 E* 轮,8 名开发受试者只最终评分一次。
唯一变化:窗口秒数(1 → 30 个时间位置,2 → 15 个),LFT 的 n_pool 相应为 30/15,
其余(LFT 结构、tokens、lr/wd/clip、batch、bf16、数据顺序)完全一致。
两种窗口的 pooled 特征同为 2560 维,线性头与窗口无关。

与 C 脚本的机器差异(已核对并在 plan.json 记录):
- 无 fcntl(Windows);B 的 27 号脚本以 LF 行尾固定,内容与 C 的 CRLF pin 相同。
- 特征索引覆盖 label25(14 人)+ 8 名验证者共 44 条记录(53,496 段),与 B 第三周
  提取口径一致;C 机器上的全量索引核验改为同范围的显式核验。
- 默认不做特征暂存(staging),直接 mmap 读取特征目录。

用法(仓库根目录):
  python scripts/37_a_w4_lft_compare.py --window-sec 1 --output experiment3/week4/pool_compare/runs/lft_w1s_seed0_v01
  python scripts/37_a_w4_lft_compare.py --window-sec 2 --output experiment3/week4/pool_compare/runs/lft_w2s_seed0_v01
中断后用同一命令重跑即可续训(逐 epoch 断点)。
"""
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import runpy
import shutil
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "experiment3/week3/scripts"))
sys.path.insert(0, str(REPO / "experiment3/week4/pool_compare/_compat"))  # fcntl 垫片(仅 Windows)
import run_week3_c_readout as c  # noqa: E402  C 的 W3 readout 入口
import a_w4_readout as a  # noqa: E402  W4 窗口参数化变体

# C 的 pin(c7e8ce01…)是 B 27 号脚本的 CRLF 行尾字节;本机 LF 行尾的同一内容 SHA:
B_PROBE_SHA = "47c5e135583d892bd45eb2d5a5085d754ff739273e428de5c0581873a0de531e"
B_PROBE_SHA_NOTE = "CRLF variant of C's pin c7e8ce01…, identical content (line endings only)"
W4_CONFIG = REPO / "experiment3/week4/pool_compare/configs/w4_lft_compare.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path, state):
    partial = path.with_suffix(".tmp")
    torch.save(state, partial)
    partial.replace(path)


def build_feature_index(folder):
    """把 B 口径的 sample_index.csv 转成 C 口径的 feature_index.csv(补 feature_path/row_index)。"""
    source = folder / "sample_index.csv"
    rows = c.read_csv(source)
    order = {}
    for row in rows:
        order.setdefault(row["record_id"], []).append(row)
    index = []
    for record, items in order.items():
        for position, row in enumerate(items):
            row["feature_path"] = f"{record}.npy"
            row["row_index"] = position
            index.append(row)
    out = folder / "feature_index.csv"
    c.write_csv(out, index)
    return index


def stage_training_features(rows, positions, folder, sources):
    """把重复读取移出慢挂载(与 C 的版本一致,只是 sources 可传)。默认不启用。"""
    folder.mkdir(parents=True, exist_ok=True)
    staged = {name: folder / name for name in c.MODALITIES}
    paths = sorted({rows[i]["feature_path"] for i in positions})
    receipt = folder / "copies.json"
    copies = load_json(receipt) if receipt.exists() else {}
    for name, output in staged.items():
        output.mkdir(exist_ok=True)
        for filename in paths:
            original = sources[name] / filename
            target = output / filename
            stat = original.stat()
            key = f"{name}/{filename}"
            saved = copies.get(key, {})
            if (target.is_file() and saved.get("size") == stat.st_size
                    and saved.get("mtime_ns") == stat.st_mtime_ns
                    and target.stat().st_size == stat.st_size
                    and c.sha(target) == saved.get("sha256")):
                continue
            partial = target.with_suffix(".partial")
            digest = hashlib.sha256()
            with original.open("rb") as incoming, partial.open("wb") as outgoing:
                while block := incoming.read(16 * 2**20):
                    digest.update(block)
                    outgoing.write(block)
            c.require(c.sha(partial) == digest.hexdigest()
                      and original.stat().st_mtime_ns == stat.st_mtime_ns,
                      f"Staged feature differs from its source: {original}")
            partial.replace(target)
            copies[key] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest.hexdigest())
            c.write_json(receipt, copies)
    return staged, copies


def fit_phase(config, seed, device, data, normalization, output, epochs, progress, n_pool, held=None,
              no_classifier=False):
    """训练一折或最终重训,逐 epoch 落盘(与 C 的 fit_phase 相同,仅 build_heads 参数化)。"""
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    model = a.w4_build_heads(config, seed, device, n_pool, no_classifier)["C1"]
    optimizer = c.optimizer_for(model, config)
    rng = np.random.default_rng(seed)
    if normalization.exists():
        stats = torch.load(normalization, map_location=device, weights_only=True)
        mean, std = stats["mean"], stats["std"]
    else:
        progress("fitting_train_normalization")
        mean, std = c.train_statistics(data, config, device)
        save_checkpoint(normalization, dict(mean=mean.cpu(), std=std.cpu()))
    history = []
    fold_predictions = torch.empty((epochs, len(held.rows), 5)) if held is not None else None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        c.require(state["seed"] == seed and state["epochs_planned"] == epochs,
                  f"Checkpoint protocol differs in {output}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["shuffle_rng"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device)
        history = state["history"]
        if held is not None:
            fold_predictions[:len(history)] = state["fold_predictions"]
    for epoch in range(len(history) + 1, epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        progress("training", epoch=epoch, epochs_planned=epochs, updates=0)

        def on_batch(updates, samples, losses):
            loss = float(losses.item() / samples)
            c.require(np.isfinite(loss), "Nonfinite training loss")
            progress("training", epoch=epoch, epochs_planned=epochs, updates=updates,
                     samples=samples, train_loss=loss, epoch_seconds=time.perf_counter() - start)

        updates, samples, losses = c.train_epoch(
            {"C1": model}, {"C1": optimizer}, data, config, device, mean, std, rng, on_batch
        )
        train_seconds = time.perf_counter() - start
        loss = float(losses.item() / samples)
        c.require(np.isfinite(loss), "Nonfinite training loss")
        if held is not None:
            progress("evaluating_inner_holdout", epoch=epoch, epochs_planned=epochs)
            model.eval()
            probabilities = c.predict_development({"C1": model}, held, config, device, mean, std)["C1"]
            c.require(np.isfinite(probabilities).all(), "Incomplete inner holdout predictions")
            fold_predictions[epoch - 1] = torch.from_numpy(probabilities)
        history.append(dict(epoch=epoch, train_samples=samples, updates=updates, train_loss=loss,
                            train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - start,
                            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20))
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), seed=seed,
                     epochs_planned=epochs, history=history, shuffle_rng=rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                     fold_predictions=fold_predictions[:epoch] if held is not None else None)
        save_checkpoint(checkpoint, state)
        c.write_json(output / "history.json", history)
    c.write_json(output / "history.json", history)
    model.eval()
    return model, mean, std, fold_predictions, history


def write_selection(output, train_rows, folds, fold_index, oof, evaluator):
    c.require(np.all(fold_index >= 0) and np.isfinite(oof).all(), "Incomplete training-side predictions")
    truth = np.array([int(row["y_true"]) for row in train_rows])
    curve = []
    for epoch, probabilities in enumerate(oof, 1):
        metrics = evaluator.compute_metrics(truth, probabilities.argmax(axis=1))
        curve.append(dict(epoch=epoch, oof_macro_f1=metrics["macro_f1"], oof_n1_f1=metrics["n1_f1"]))
    best = max(curve, key=lambda row: (row["oof_macro_f1"], -row["epoch"]))
    c.write_csv(output / "inner_cv.csv", curve)
    predictions = []
    for row, fold, probability in zip(train_rows, fold_index, oof[best["epoch"] - 1]):
        predictions.append(dict(
            **{key: row[key] for key in ("sample_id", "subject_id", "record_id", "epoch_index", "y_true")},
            fold=int(fold), y_pred=int(probability.argmax()),
            **dict(zip(evaluator.PROB_COLS, map(float, probability))),
        ))
    c.write_csv(output / "oof_train.csv", predictions)
    c.write_json(output / "selection.json", dict(best, folds=folds, dev_used=False))
    return best


def run(args):
    b_script = args.b_checkout / "scripts/27_b_w3_linear_probe.py"
    c.require(c.sha(b_script) == B_PROBE_SHA,
              f"B selection source differs from the pinned version ({B_PROBE_SHA_NOTE})")
    sys.path.insert(0, str(b_script.parent))
    b = runpy.run_path(str(b_script))
    b_common = b["w"]
    evaluator = b["ev"]
    dev_reference = b_common.VAL_REFERENCE
    config = load_json(W4_CONFIG)
    c.require(config["budget"] == "label25" and config["layers"] == list(range(20))
              and b["MAX_EPOCHS"] == 50 and b["FOLD_SEED"] == 0, "Unexpected main experiment protocol")
    config.update(epochs=b["MAX_EPOCHS"], window_sec=args.window_sec,
                  signal_rate=args.signal_rate, no_classifier=args.no_classifier,
                  checkpoint_selection="B subject-wise five-fold pooled OOF macro-F1; ties choose earliest",
                  normalization="per modality/layer/hidden shared across time; fitted on each fold's training subjects, then all 14 for the final refit")
    n_pool = a.n_positions(args.window_sec)
    device = torch.device("cuda")
    c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    torch.set_num_threads(4)
    output = args.output
    status = dict(budget="label25", seeds=config["seeds"], completed_seeds=[],
                  folds_per_seed=5, epochs_per_fold=50, window_sec=args.window_sec,
                  signal_rate=args.signal_rate, no_classifier=args.no_classifier,
                  n_time_positions=n_pool, test_used=False)
    context = {}

    def progress(stage, **values):
        updated = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
        value = dict(status, stage=stage, updated_at=updated, **context, **values)
        c.write_json(output / "progress.json", value)
        print(json.dumps(value), flush=True)

    try:
        progress("checking_inputs")
        sources = a.w4_sources(args.window_sec, args.signal_rate)
        for folder in sources.values():
            if not (folder / "feature_index.csv").exists():
                build_feature_index(folder)
        rows, train, dev, audit = a.w4_inspect_inputs(dev_reference, sources, args.window_sec,
                                                      args.signal_rate)
        subject_ids = audit["labeled_subjects"]
        b_subjects, _ = b_common.selected_subjects(args.b_checkout / "manifests/label_subsets_v2.csv", "label25")
        c.require(set(b_subjects) == set(subject_ids), "B and C training subject lists differ")
        folds = b["subject_folds"](subject_ids, min(5, len(subject_ids)))
        c.require(sorted(s for fold in folds for s in fold) == sorted(subject_ids),
                  "Fold assignment does not cover the labeled subjects once")
        train_rows = [rows[i] for i in train]
        train_subjects = np.array([row["subject_id"] for row in train_rows])
        c.require(not (set(train_subjects) & {rows[i]["subject_id"] for i in dev}),
                  "Outer development subjects overlap training")
        staged = sources
        copies = None
        if args.staging:
            progress("staging_training_features")
            required = sum((folder / name).stat().st_size for folder in sources.values()
                           for name in {row["feature_path"] for row in train_rows})
            c.require(shutil.disk_usage(args.staging.parent).free > required + 2**31,
                      "Insufficient space on the training cache volume")
            staged, copies = stage_training_features(rows, train, args.staging, sources)
        plan = dict(config=config, folds=folds, input_audit=audit, b_probe_sha256=B_PROBE_SHA,
                    b_probe_sha256_note=B_PROBE_SHA_NOTE,
                    b_evaluator_sha256=c.sha(args.b_checkout / "src/evaluation/evaluate.py"),
                    cv_runner_sha256=c.sha(Path(__file__)), staged_features=copies,
                    w4_readout_sha256=c.sha(REPO / "scripts/a_w4_readout.py"))
        plan_path = output / "plan.json"
        if plan_path.exists():
            c.require(load_json(plan_path) == json.loads(json.dumps(plan)), "CV inputs changed since launch")
        else:
            c.write_json(plan_path, plan)
        norm = output / "normalization"
        norm.mkdir(exist_ok=True)
        smoke_path = output / "smoke.json"
        if not smoke_path.exists():
            c.write_json(smoke_path, a.w4_smoke(config, rows, train, device, staged, n_pool,
                                                args.no_classifier))
        for seed in config["seeds"]:
            seed_dir = output / f"seed{seed}_v01"
            seed_dir.mkdir(exist_ok=True)
            if (seed_dir / "complete.json").exists():
                status["completed_seeds"].append(seed)
                continue
            oof = np.full((config["epochs"], len(train), 5), np.nan, dtype=np.float32)
            fold_index = np.full(len(train), -1, dtype=np.int8)
            for k, held_subjects in enumerate(folds):
                context.update(seed=seed, phase="inner_cv", fold=k + 1)
                held_mask = np.isin(train_subjects, held_subjects)
                c.require(held_mask.any() and (~held_mask).any() and np.all(fold_index[held_mask] == -1),
                          "Invalid held-out fold")
                fit_data = a.W4CachedEpochs(rows, train[~held_mask], config["layers"], staged, n_pool)
                held_data = a.W4CachedEpochs(rows, train[held_mask], config["layers"], staged, n_pool)
                _, _, _, predictions, _ = fit_phase(
                    config, seed, device, fit_data, norm / f"fold{k + 1}.pt",
                    seed_dir / f"fold{k + 1}", config["epochs"], progress, n_pool, held_data,
                    args.no_classifier
                )
                oof[:, held_mask] = predictions.numpy()
                fold_index[held_mask] = k
                progress("fold_complete")
            selected = write_selection(seed_dir, train_rows, folds, fold_index, oof, evaluator)
            context.clear()
            context.update(seed=seed, phase="final_refit", selected_epoch=selected["epoch"])
            fit_data = a.W4CachedEpochs(rows, train, config["layers"], staged, n_pool)
            model, mean, std, _, history = fit_phase(
                config, seed, device, fit_data, norm / "all14.pt", seed_dir / "refit",
                selected["epoch"], progress, n_pool, no_classifier=args.no_classifier
            )
            save_checkpoint(seed_dir / "final_head.pt",
                            dict(model=model.state_dict(), mean=mean.cpu(), std=std.cpu(),
                                 seed=seed, epoch=selected["epoch"], config=config))
            progress("evaluating_outer_development")
            dev_data = a.W4CachedEpochs(rows, dev, config["layers"], staged, n_pool)
            probabilities = c.predict_development({"C1": model}, dev_data, config, device, mean, std)["C1"]
            c.require(np.isfinite(probabilities).all(), "Nonfinite outer development predictions")
            path = seed_dir / "predictions_val.csv"
            b_common.write_prediction_table(path, dev_data.rows, probabilities)
            metrics = evaluator.evaluate(path, reference=dev_reference, n_boot=2000, seed=0)
            metrics.update(selected_epoch=selected["epoch"], oof_macro_f1=selected["oof_macro_f1"],
                           final_updates=sum(epoch["updates"] for epoch in history), test_used=False)
            c.write_json(seed_dir / "metrics.json", metrics)
            evaluator.per_subject_table(evaluator.load_predictions(path)).to_csv(
                seed_dir / "per_subject_metrics.csv", index=False)
            c.write_json(seed_dir / "complete.json", dict(seed=seed, selected_epoch=selected["epoch"],
                                                         predictions_sha256=c.sha(path)))
            status["completed_seeds"].append(seed)
            progress("seed_complete")
        context.clear()
        progress("complete")
    except Exception as error:
        progress("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-sec", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--signal-rate", type=int, choices=[100, 200], default=200,
                        help="100 = downsampling 变体特征(信号原生 100 Hz 进主干)")
    parser.add_argument("--no-classifier", action="store_true",
                        help="去掉 LFT 的 classifier module(decoder + 平均池化 + 单层 Linear)")
    parser.add_argument("--b-checkout", type=Path,
                        default=REPO / "experiment3/week4/pool_compare/b_checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path, help="可选:把特征复制到该目录再训练(C 机器的做法)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / "run.lock"
    if lock.exists():
        print(f"另一个进程可能正在写 {args.output},退出。若确认没有,先删除 {lock}。", flush=True)
        sys.exit(1)
    lock.write_text(f"pid={os.getpid()} started={time.time()}", encoding="utf-8")
    try:
        run(args)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
