#!/bin/bash
# 等主实验过了指定步数的 val + 断点保存后，停掉训练的 tmux 会话（之后用同一条命令可续跑）。
# 用法：stop_at_step.sh <run_name> <tmux_session> <step>
set -u
RUN=$1; SESS=$2; STEP=$3
SMALL="$(dirname "$0")/../runs/$RUN"
BIG="${SLEEP_W4_BIG:-$HOME/sleep_project/week4/big}/runs/$RUN"
echo "waiting for step $STEP checkpoint + val in $RUN ..."
# history 里出现 > STEP 的行，说明 STEP 步的 val、ckpt、latest.pt 都已写完
until [ -f "$BIG/ckpt_step$STEP.pt" ] && grep -q "^$STEP,ema,0," "$SMALL/val_mse.csv" 2>/dev/null \
      && awk -F, -v s="$STEP" 'NR>1 && $1+0>s+0 {f=1} END{exit !f}' "$SMALL/history.csv" 2>/dev/null; do
  if ! tmux has-session -t "$SESS" 2>/dev/null; then echo "training session $SESS ended before step $STEP"; exit 1; fi
  sleep 30
done
tmux kill-session -t "$SESS"
echo "STOPPED $SESS after step $STEP at $(date '+%Y-%m-%d %H:%M:%S'); last history row: $(tail -1 "$SMALL/history.csv")"
