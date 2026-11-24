#!/bin/bash
# 11-19 晚：WSL 重启后原队列（queue_after_shifted.sh）丢失，这里接着做剩下的步骤：
#   14（续）主实验其余各点提特征 → 18 主实验各点 probe → 19 self-cond probe（全部顺序执行）
set -u -o pipefail
cd "$(dirname "$0")"
LOGS=$HOME/sleep_project/week4/logs
BIG=${SLEEP_W4_BIG:-$HOME/sleep_project/week4/big}
FEAT=$BIG/features
MAIN=W4_D_main_m64x12_seed0_v01
SC=W4_D_selfcond_l9_mean_30k_v01
log() { echo "[$(date '+%F %T')] $*"; }
fdir() { echo "$FEAT/$1/step$2_ema_model_pool1s"; }
STEPS="0 15000 30000 60000 120000 240000"   # 240k = best = stop（stop_decision.json）

# ---- 14（续）提特征 ----
for s in $STEPS; do
  if [ ! -f "$(fdir $MAIN $s)/DONE" ]; then
    log "14 extract main step $s"
    python -u 04_extract_features.py --run-name $MAIN --step $s 2>&1 | tee -a "$LOGS/14_extract_main_ckpts.log" \
      || log "WARNING: extract main step $s failed"
  fi
done

# ---- 18 主实验 probe ----
ARMS=""; PAIRS=""
for s in $STEPS; do
  if [ "$s" = "0" ]; then name=d_rand; else name=d_pre$(printf %03d $((s / 1000)))k; fi
  ARMS="$ARMS --arm $name=$(fdir $MAIN $s):eeg,eog:std"
  [ "$s" != "0" ] && PAIRS="$PAIRS --pair $name:d_rand"
done
ARMS="$ARMS --arm d_pre030k_eog=$(fdir $MAIN 30000):eog:std --arm d_rand_eog=$(fdir $MAIN 0):eog:std"
PAIRS="$PAIRS --pair d_pre030k_eog:d_rand_eog"
log "18 main probes"
python -u 05_probes.py --tag w4_main_ckpts --skip-missing $ARMS $PAIRS 2>&1 | tee -a "$LOGS/18_probes_main.log" \
  || log "WARNING: step 18 failed"

# ---- 19 self-cond probe ----
log "19 self-cond probes"
python -u 05_probes.py --tag w4_selfcond_30k --skip-missing \
  --arm d_pre030k=$(fdir $MAIN 30000):eeg,eog:std \
  --arm sc030k=$(fdir $SC 30000):eeg,eog:std \
  --arm sc030k_fb=$(fdir $SC 30000):eeg,eog:feedback \
  --pair sc030k:d_pre030k --pair sc030k_fb:d_pre030k 2>&1 | tee -a "$LOGS/19_probes_selfcond.log" \
  || log "WARNING: step 19 failed"
log "QUEUE DONE"
