#!/usr/bin/env bash
# backbone seed 1 轻量复现：30k 步预训练 -> 提 step0 / step30000 特征 -> B 协议探针与对比
# 用法（Ubuntu，conda activate eegdm 之后）：bash 本文件的完整路径
set -o pipefail
RUN=W3_D_pretrain_m64x12_bs108_seed1_v01
SRC=<D_ROOT>/week3-1/exp_d_pretrain/src
LOG=<D_ROOT>/week3-1/logs
echo "start: $(date '+%F %T')"
python -u $SRC/04_pretrain.py --config m64x12 --max-steps 30000 --milestones 0,15000,30000 --seed 1 --run-name $RUN 2>&1 | tee -a $LOG/14_pretrain_seed1.log || exit 1
python -u $SRC/05_extract_features.py --run-name $RUN --step 0 2>&1 | tee $LOG/15_features_seed1_step0.log || exit 1
python -u $SRC/05_extract_features.py --run-name $RUN --step 30000 2>&1 | tee $LOG/16_features_seed1_step30000.log || exit 1
python -u $SRC/09_seed1_probes.py 2>&1 | tee $LOG/17_seed1_probes.log || exit 1
echo "ALL SEED1 STEPS DONE at $(date '+%F %T')"
