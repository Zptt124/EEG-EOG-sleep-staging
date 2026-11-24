#!/bin/bash
# W4 公共训练脚本冒烟测试：plain 中断续训 vs 不中断、self-cond、cross-modal。
set -u
cd "$(dirname "$0")"
HO=${HELDOUT_CSV:-../../../manifests/pretrain_split_w4_v01.csv}   # B 的正式名单（仓库根目录 manifests/，PR 21）
COMMON="--heldout-csv $HO --max-steps 300 --val-every 100 --min-steps 0 --log-every 50 --save-every 50"
echo "=== [1] plain, uninterrupted"
python -u 01_pretrain.py --run-name SMOKE_plain_full $COMMON || exit 1
echo "=== [2] plain, interrupted at ~170 then resumed"
python -u 01_pretrain.py --run-name SMOKE_plain_resume $COMMON &
PID=$!
while ! grep -q '^150,' ../runs/SMOKE_plain_resume/history.csv 2>/dev/null; do sleep 2; done
sleep 8; kill -9 $PID; wait $PID 2>/dev/null
echo "killed; last history row: $(tail -1 ../runs/SMOKE_plain_resume/history.csv)"
python -u 01_pretrain.py --run-name SMOKE_plain_resume $COMMON || exit 1
echo "=== [3] self-cond layer 8"
python -u 01_pretrain.py --run-name SMOKE_selfcond_l8 --heldout-csv $HO --max-steps 200 --val-every 100 --no-early-stop --log-every 50 --self-cond-layer 8 || exit 1
echo "=== [4] cross-modal shifted"
python -u 01_pretrain.py --run-name SMOKE_xmodal_shifted --heldout-csv $HO --max-steps 200 --val-every 100 --no-early-stop --log-every 50 --cross-modal shifted --cond-dropout 0.5 || exit 1
echo "SMOKE ALL DONE"
