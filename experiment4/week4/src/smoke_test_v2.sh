#!/bin/bash
# 冒烟测试 v2（按 C 的跨模态规则改完之后）：same_window 正常跑；shifted 中断续训 vs 不中断（检查 dropout 随机流的续训）。
# 用 --smoke-small-val 缩短 val 时间（每人每模态 16 窗）。
set -u
cd "$(dirname "$0")"
HO=${HELDOUT_CSV:-../../../manifests/pretrain_split_w4_v01.csv}   # B 的正式名单（仓库根目录 manifests/，PR 21）
COMMON="--heldout-csv $HO --max-steps 200 --val-every 100 --no-early-stop --log-every 50 --save-every 50 --smoke-small-val"
echo "=== [1] cross-modal same_window"
python -u 01_pretrain.py --run-name SMOKE2_xmodal_same $COMMON --cross-modal same_window || exit 1
echo "=== [2] cross-modal shifted, uninterrupted"
python -u 01_pretrain.py --run-name SMOKE2_xmodal_shifted_full $COMMON --cross-modal shifted || exit 1
echo "=== [3] cross-modal shifted, killed after step 150 then resumed"
python -u 01_pretrain.py --run-name SMOKE2_xmodal_shifted_resume $COMMON --cross-modal shifted &
PID=$!
while ! grep -q '^150,' ../runs/SMOKE2_xmodal_shifted_resume/history.csv 2>/dev/null; do sleep 2; done
sleep 3; kill -9 $PID; wait $PID 2>/dev/null
echo "killed; last history row: $(tail -1 ../runs/SMOKE2_xmodal_shifted_resume/history.csv)"
python -u 01_pretrain.py --run-name SMOKE2_xmodal_shifted_resume $COMMON --cross-modal shifted || exit 1
echo "SMOKE V2 ALL DONE"
