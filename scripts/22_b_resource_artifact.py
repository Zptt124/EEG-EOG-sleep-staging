"""Step W2-B14: 从 B 的 run 目录生成资源表行、权重包与大产物索引。

- resource_profile.csv:替换 B 的行,保留他人行。耗时、更新数、设备、样本数来自 config.yaml,参数量从 best.ckpt 数;
  训练时没有记录内存峰值,peak_memory_mib 写 NA,不估算。
- <zip-out>:每个 run 的 best.ckpt 与 normalization.npz。打包前按 config.yaml 校验 best.ckpt 的 SHA256。
- artifact_index.csv:只有给了 --release-url 才写(权重包上传之后),每个 run 的权重一行,加整包一行;
  替换 B 的行,保留他人行。没上传就不写,索引里不放不存在的地址。

用法:
  python scripts/22_b_resource_artifact.py --runs-root runs --zip-out <zip> [--release-url <Release 资产地址>]
"""
import argparse
import csv
import zipfile
from pathlib import Path

import pandas as pd
import torch
import yaml

from common import BASE, RUNS
import w2_common as w

REPORT = BASE / "reports" / "week2"


def read_table(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_table(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def trainable_parameters(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state"]   # B 自己生成的文件,含 numpy 数组,SHA 已校验
    return sum(v.numel() for v in state.values())


def resource_row(run_dir, cfg):
    epochs = len(pd.read_csv(run_dir / "history.csv"))
    n = cfg.get("train_samples", cfg.get("train_windows"))
    return dict(run_id=cfg["run_id"], phase="train", device=cfg["device"], gpu_memory_gib="NA", driver="NA", pytorch=cfg["torch"],
                cuda="NA", precision="float32", determinism_settings="seed_everything before model init; deterministic_algorithms=False",
                trainable_parameters=trainable_parameters(run_dir / "best.ckpt"), actual_updates=cfg["actual_updates"],
                actual_epochs=epochs, elapsed_seconds=cfg["elapsed_seconds"], peak_memory_mib="NA",
                examples_per_second=f"{n * epochs / float(cfg['elapsed_seconds']):.1f}", labeled_examples=n, pseudo_candidates=0,
                pseudo_accepted=0, pseudo_training_examples=0,
                notes="Mac CPU 头部训练,特征已预先提取;未记录内存峰值;耗时含每轮验证")


def artifact_rows(runs, configs, zip_path, url):
    rows = [dict(artifact_id=zip_path.name, artifact_type="run_weights_bundle", owner="B", source_run_id="NA", adaptation_version="NA",
                 adaptation_seed="NA", teacher_id="NA", storage_uri=url, sha256=w.sha256_file(zip_path),
                 size_bytes=zip_path.stat().st_size, code_commit="NA", config_path="NA", index_path="NA",
                 normalization_path="NA", notes=f"{len(runs)} 个 run 的 best.ckpt 与 normalization.npz")]
    for d, c in zip(runs, configs):
        rows.append(dict(artifact_id=f"{d.name}_best_ckpt", artifact_type="head_weights", owner="B", source_run_id=d.name,
                         adaptation_version="NA", adaptation_seed="NA", teacher_id="NA", storage_uri=f"{url}#/{d.name}/best.ckpt",
                         sha256=c["best_ckpt_sha256"], size_bytes=(d / "best.ckpt").stat().st_size, code_commit=c["code_commit"],
                         config_path=f"reports/week2/runs_B/{d.name}/config.yaml", index_path="NA",
                         normalization_path=f"{url}#/{d.name}/normalization.npz", notes=""))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs-root", default=str(RUNS))
    ap.add_argument("--zip-out", required=True)
    ap.add_argument("--release-url", help="权重包已上传后的地址;不给就不写 artifact_index.csv")
    args = ap.parse_args(argv)

    runs = sorted(Path(args.runs_root).glob("W2_B_*"))
    configs = [yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) for d in runs]
    for d, cfg in zip(runs, configs):
        if w.sha256_file(d / "best.ckpt") != cfg["best_ckpt_sha256"]:
            raise ValueError(f"{d.name}: best.ckpt 与 config.yaml 记录的 SHA256 不一致")

    zip_path = Path(args.zip_out)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for d in runs:
            for name in ("best.ckpt", "normalization.npz"):
                z.write(d / name, f"{d.name}/{name}")

    fields, rows = read_table(REPORT / "resource_profile.csv")
    keep = [r for r in rows if not r["run_id"].startswith("W2_B_")]
    write_table(REPORT / "resource_profile.csv", fields, keep + [resource_row(d, c) for d, c in zip(runs, configs)])

    if args.release_url:
        fields, rows = read_table(REPORT / "artifact_index.csv")
        write_table(REPORT / "artifact_index.csv", fields,
                    [r for r in rows if r["owner"] != "B"] + artifact_rows(runs, configs, zip_path, args.release_url))
    print(f"{len(runs)} 个 run -> {zip_path} ({zip_path.stat().st_size} 字节),sha256={w.sha256_file(zip_path)}")


if __name__ == "__main__":
    main()
