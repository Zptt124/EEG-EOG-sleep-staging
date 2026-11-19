#!/bin/bash
# 实际执行：25a、25b 完成；11-19 22:18 C 锁定 1 秒 pooling，25c、25d（2 秒版本）中途取消，未产出结果。
# 11-19 晚：GPU 收尾队列（顺序执行）
#   25a 06 号自检：重算错时 30k 的 val MSE，和训练时的 val_mse.csv 比较（应逐位相同）
#   25b 用统一噪声重算 A 同窗 30k 的 val MSE，并和 A 自己报的值比较
#   25c 三组 30k 的 EOG 特征再提一份 2 秒 pooling（C 还没锁定 1 秒 / 2 秒，两种都备好）
#   25d 三组 2 秒 pooling 的 EOG-only probe 与配对比较
set -u -o pipefail
cd "$(dirname "$0")"
BIG=${SLEEP_W4_BIG:-$HOME/sleep_project/week4/big}
FEAT=$BIG/features
HO=../../../manifests/pretrain_split_w4_v01.csv
OUT=../analysis/val_recompute
A_CK=$BIG/downloads/a_same30k/x/w4_a_xmodal_same_30k_v01
SHIFT=W4_B_xmodal_shifted_30k_seed0_v01
MAIN=W4_D_main_m64x12_seed0_v01
SAME=W4_A_xmodal_same_30k_seed0_v01
log() { echo "[$(date '+%F %T')] $*"; }

log "25a self-check on shifted 30k"
python -u 06_val_mse_ckpt.py --ckpt $BIG/runs/$SHIFT/ckpt_step30000.pt --heldout-csv $HO \
  --out $OUT/${SHIFT}_step30000.csv --check-csv ../runs/$SHIFT/val_mse.csv || log "WARNING: 25a failed"
log "25b A same-window 30k with team noise"
python -u 06_val_mse_ckpt.py --ckpt $A_CK/ckpt_step30000.pt --heldout-csv $HO \
  --out $OUT/${SAME}_step30000.csv --check-csv $A_CK/val_mse.csv || log "WARNING: 25b failed"

log "25c EOG features with 2 s pooling"
python -u 04_extract_features.py --run-name $MAIN --step 30000 --modalities eog --pool-sec 2 || log "WARNING: main p2 failed"
python -u 04_extract_features.py --run-name $SHIFT --step 30000 --modalities eog --pool-sec 2 || log "WARNING: shifted p2 failed"
python -u 04_extract_features.py --ckpt $A_CK/ckpt_step30000.pt --out-name $SAME --modalities eog --pool-sec 2 || log "WARNING: same p2 failed"

log "25d probes, 2 s pooling"
python -u 05_probes.py --tag w4_xmodal_recheck_pool2s --skip-missing \
  --arm a_same030k_eog_p2=$FEAT/$SAME/step30000_ema_model_pool2s:eog:std \
  --arm d_pre030k_eog_p2=$FEAT/$MAIN/step30000_ema_model_pool2s:eog:std \
  --arm b_shift030k_eog_p2=$FEAT/$SHIFT/step30000_ema_model_pool2s:eog:std \
  --pair a_same030k_eog_p2:d_pre030k_eog_p2 --pair b_shift030k_eog_p2:d_pre030k_eog_p2 \
  --pair a_same030k_eog_p2:b_shift030k_eog_p2 || log "WARNING: 25d failed"
log "QUEUE DONE"
