#!/bin/bash
# W5 冒烟测试：--window-sec 30（连续 30 秒输入）。A = 一口气跑 200 步；B = 跑到约 150 步强杀后续跑到 200 步，val 应与 A 逐位相同。
set -u -o pipefail
cd "$(dirname "$0")"
HO=../../../manifests/pretrain_split_w4_v01.csv
COMMON="--heldout-csv $HO --window-sec 30 --batch-size 12 --max-steps 200 --val-every 100 --save-every 50 --log-every 50 --no-early-stop --smoke-small-val"
python -u 01_pretrain.py --run-name SMOKE_w30_full $COMMON || exit 1
python -u 01_pretrain.py --run-name SMOKE_w30_resume $COMMON &
PID=$!
until grep -q "^step 150/" ../runs/SMOKE_w30_resume/history.csv 2>/dev/null || grep -q "^150," ../runs/SMOKE_w30_resume/history.csv 2>/dev/null; do sleep 2; done
sleep 5; kill $PID; wait $PID 2>/dev/null
echo "killed resume run"
python -u 01_pretrain.py --run-name SMOKE_w30_resume $COMMON || exit 1
echo "SMOKE W30 DONE"
