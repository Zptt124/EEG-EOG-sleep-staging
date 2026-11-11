#!/usr/bin/env bash
# 等预训练结束后自动跑完：主比较（step240000）-> 中间保存点（探索性学习曲线）
# 用法（Ubuntu，conda activate eegdm 之后）：bash 本文件的完整路径 [先等待的小时数，可省略，默认 0]
set -o pipefail
RUN=W3_D_pretrain_m64x12_bs108_seed0_v01
SRC=<D_ROOT>/week3-1/exp_d_pretrain/src
LOG=<D_ROOT>/week3-1/logs
FINAL=$HOME/week3-1/runs/$RUN/ckpt_step240000.pt

DELAY_H=${1:-0}
echo "start: $(date '+%F %T'); sleeping ${DELAY_H} h before checking"
sleep "${DELAY_H}h"
echo "waiting for $FINAL (check every 5 min) ..."
while [ ! -f "$FINAL" ]; do sleep 300; done
echo "final checkpoint found at $(date '+%F %T'); waiting 3 min for training to exit"
sleep 180

# 1) 主比较：step 240000 vs step 0（random-init）
python -u $SRC/05_extract_features.py --run-name $RUN --step 240000 2>&1 | tee $LOG/07_features_step240000.log || exit 1
python -u $SRC/06_probe.py --run-name $RUN --tags step240000_ema_model 2>&1 | tee $LOG/08_probe_step240000.log || exit 1
python -u $SRC/07_compare.py --run-name $RUN --pretrained-tag step240000_ema_model 2>&1 | tee $LOG/09_compare_step240000.log || exit 1

# 2) 探索性：中间保存点（不用于主结论）
for S in 120000 60000 30000 15000; do
  python -u $SRC/05_extract_features.py --run-name $RUN --step $S 2>&1 | tee $LOG/10_features_step$S.log || exit 1
done
python -u $SRC/06_probe.py --run-name $RUN --tags step15000_ema_model,step30000_ema_model,step60000_ema_model,step120000_ema_model 2>&1 | tee $LOG/11_probe_milestones.log || exit 1
for S in 15000 30000 60000 120000; do
  python -u $SRC/07_compare.py --run-name $RUN --pretrained-tag step${S}_ema_model 2>&1 | tee $LOG/12_compare_step$S.log || exit 1
done
echo "ALL AFTER-PRETRAIN STEPS DONE at $(date '+%F %T')"
