#!/bin/bash
# W4 D 的 GPU 队列（11-17 排定）：错时组 30k 跑完后依次自动执行，GPU 同一时间只跑一个任务。
#   13 提错时组 30k 的 EOG 特征（给 B）
#   14 主实验 30k 提特征（EEG+EOG；三组对照急用，先做）
#   15 self-cond 选层：m ∈ {3, 6, 9} 各训 2k 步，按 held-out val（online 权重）stop_metric 最低选 m
#   16 self-cond 30k（选出的 m，mean 池化）
#   17 self-cond 30k 提特征（EEG+EOG，含 feedback）
#   14（续）主实验其余各点提特征（最好点 / 停止点 / 0 / 15k / 60k / 120k）
#   18 主实验各点 probe（CPU，后台 tmux）
#   19 self-cond probe（等 18 结束后，CPU）
# 某一步失败时打印 WARNING 并继续不依赖它的后续步骤。
set -u -o pipefail
cd "$(dirname "$0")"
mkdir -p ../analysis
LOGS=$HOME/sleep_project/week4/logs
BIG=${SLEEP_W4_BIG:-$HOME/sleep_project/week4/big}
FEAT=$BIG/features
HO=../../../manifests/pretrain_split_w4_v01.csv
MAIN=W4_D_main_m64x12_seed0_v01
SHIFT=W4_B_xmodal_shifted_30k_seed0_v01
log() { echo "[$(date '+%F %T')] $*"; }
fdir() { echo "$FEAT/$1/step$2_ema_model_pool1s"; }   # 04_extract_features.py 的输出目录

log "waiting for w4_11_b_shifted_30k (which itself waits for the main run) ..."
while tmux has-session -t w4_11_b_shifted_30k 2>/dev/null; do sleep 60; done
grep -q "PRETRAIN DONE" "$LOGS/11_b_shifted_30k.log" 2>/dev/null || log "WARNING: shifted 30k did not finish normally"
[ -f "$HOME/sleep_project/week4/exp_d_pretrain/runs/$MAIN/stop_decision.json" ] || { log "ERROR: main run has no stop_decision.json; stop queue"; exit 1; }

# ---- 13 错时组 EOG 特征 ----
if [ -f "$BIG/runs/$SHIFT/ckpt_step30000.pt" ]; then
  log "13 extract shifted 30k EOG features"
  python -u 04_extract_features.py --run-name $SHIFT --step 30000 --modalities eog 2>&1 | tee -a "$LOGS/13_extract_b_shifted_30k.log" \
    || log "WARNING: step 13 failed"
fi

# ---- 14 主实验各点提特征 ----
STEPS=$(python3 - "$HOME/sleep_project/week4/exp_d_pretrain/runs/$MAIN/stop_decision.json" "$BIG/runs/$MAIN" <<'EOF'
import json, sys, os
d = json.load(open(sys.argv[1]))
stop, best = int(d['stopped_at_step']), int(d['best_step'])
want = [30000, best, stop, 0, 15000, 60000, 120000]          # 先提最急用的
seen = []
for s in want:
    if s <= stop and s not in seen and os.path.exists(f'{sys.argv[2]}/ckpt_step{s}.pt'):
        seen.append(s)
print(' '.join(map(str, seen)))
EOF
)
log "14 main checkpoints to extract: $STEPS (30k now, the rest after self-cond)"
extract_main() {
  for s in "$@"; do
    if [ ! -f "$(fdir $MAIN $s)/DONE" ]; then
      python -u 04_extract_features.py --run-name $MAIN --step $s 2>&1 | tee -a "$LOGS/14_extract_main_ckpts.log" \
        || log "WARNING: extract main step $s failed"
    fi
  done
}
extract_main 30000

# ---- 15 self-cond 选层 ----
for m in 3 6 9; do
  log "15 self-cond profile m=$m"
  python -u 01_pretrain.py --run-name W4_D_scprofile_l${m}_2k_v01 --heldout-csv $HO --no-early-stop \
    --max-steps 2000 --val-every 1000 --self-cond-layer $m 2>&1 | tee -a "$LOGS/15_sc_profile.log" \
    || log "WARNING: profile m=$m failed"
done
M=$(python3 - "$HOME/sleep_project/week4/exp_d_pretrain" <<'EOF'
import csv, json, sys
root = sys.argv[1]
res = {}
for m in (3, 6, 9):
    try:
        rows = list(csv.DictReader(open(f'{root}/runs/W4_D_scprofile_l{m}_2k_v01/val_mse.csv')))
        r = [x for x in rows if x['step'] == '2000' and x['weights'] == 'online' and x['eeg_cond'] == '0']
        res[m] = float(r[0]['stop_metric'])
    except Exception:
        pass
if not res:
    print(''); sys.exit()
best = min(res, key=lambda k: (res[k], k))
json.dump(dict(rule='lowest online held-out stop_metric at step 2000 (ties -> shallower layer); '
                    'paper (arXiv 2505.10999) picks the feedback depth by short profiling runs',
               candidates=res, chosen_layer=best),
          open(f'{root}/analysis/sc_profile_choice.json', 'w'), indent=2)
print(best)
EOF
)
if [ -z "$M" ]; then
  log "ERROR: no self-cond profile result; skip 16/17/19 (main extraction and probes still run)"
else
  log "15 chosen self-cond layer m=$M"
  SC=W4_D_selfcond_l${M}_mean_30k_v01
  # ---- 16 self-cond 30k ----
  log "16 self-cond 30k ($SC)"
  python -u 01_pretrain.py --run-name $SC --heldout-csv $HO --no-early-stop --max-steps 30000 \
    --self-cond-layer $M 2>&1 | tee -a "$LOGS/16_selfcond_30k.log" || log "WARNING: self-cond 30k failed"
  # ---- 17 self-cond 提特征 ----
  log "17 extract self-cond 30k features"
  python -u 04_extract_features.py --run-name $SC --step 30000 2>&1 | tee -a "$LOGS/17_extract_selfcond_30k.log" \
    || log "WARNING: step 17 failed"
fi

# ---- 14（续）主实验其余各点提特征 ----
extract_main $STEPS

# ---- 18 主实验 probe（CPU，后台并行） ----
ARMS=""; PAIRS=""
for s in $STEPS; do
  if [ "$s" = "0" ]; then name=d_rand; else name=d_pre$(printf %03d $((s / 1000)))k; fi
  ARMS="$ARMS --arm $name=$(fdir $MAIN $s):eeg,eog:std"
  [ "$s" != "0" ] && PAIRS="$PAIRS --pair $name:d_rand"
done
ARMS="$ARMS --arm d_pre030k_eog=$(fdir $MAIN 30000):eog:std --arm d_rand_eog=$(fdir $MAIN 0):eog:std"
PAIRS="$PAIRS --pair d_pre030k_eog:d_rand_eog"
log "18 start main probes in background tmux"
tmux new -d -s w4_18_probes_main "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate eegdm && cd $PWD && python -u 05_probes.py --tag w4_main_ckpts --skip-missing $ARMS $PAIRS 2>&1 | tee -a $LOGS/18_probes_main.log'"


# ---- 19 self-cond probe（等 18 结束） ----
while tmux has-session -t w4_18_probes_main 2>/dev/null; do sleep 60; done
[ -z "$M" ] && { log "QUEUE DONE (without self-cond)"; exit 0; }
log "19 self-cond probes"
python -u 05_probes.py --tag w4_selfcond_30k --skip-missing \
  --arm d_pre030k=$(fdir $MAIN 30000):eeg,eog:std \
  --arm sc030k=$(fdir $SC 30000):eeg,eog:std \
  --arm sc030k_fb=$(fdir $SC 30000):eeg,eog:feedback \
  --pair sc030k:d_pre030k --pair sc030k_fb:d_pre030k 2>&1 | tee -a "$LOGS/19_probes_selfcond.log"
log "QUEUE DONE"
