#!/bin/bash
# W5 D：30 秒 45k 跑完后自动提特征 + B 协议 probe，与 5 秒主实验 30k 配对比较（相同信号量 = 45k，相同步数 = 30k）。
set -u -o pipefail
cd "$(dirname "$0")"
BIG=${SLEEP_W4_BIG:-$HOME/sleep_project/week4/big}
R=W5_D_w30_m64x12_seed0_45k_v01
M=W4_D_main_m64x12_seed0_v01
SMALL=../runs/$R
log() { echo "[$(date '+%F %T')] $*"; }
until [ -f $SMALL/stop_decision.json ]; do sleep 60; done
log "pretrain finished: $(cat $SMALL/stop_decision.json)"
for s in 30000 45000; do
  python -u 04_extract_features.py --run-name $R --step $s || { log "ERROR: extract step $s failed"; exit 1; }
done
python -u 05_probes.py --tag w5_w30 \
  --arm d_pre030k=$BIG/features/$M/step30000_ema_model_pool1s:eeg,eog:std \
  --arm w30_pre030k=$BIG/features/$R/step30000_ema_model_pool1s:eeg,eog:std \
  --arm w30_pre045k=$BIG/features/$R/step45000_ema_model_pool1s:eeg,eog:std \
  --pair w30_pre030k:d_pre030k --pair w30_pre045k:d_pre030k --pair w30_pre045k:w30_pre030k || exit 1
log "W5 W30 PROBES DONE"
