#!/bin/bash
# 等前一个训练（tmux 会话）正常结束后，再启动下一个训练。前一个若没有正常结束（日志里没有 PRETRAIN DONE），不启动。
# 用法：run_after.sh <prev_tmux_session> <prev_log> <log> <01_pretrain.py 的参数...>
set -u
PREV=$1; PREV_LOG=$2; LOG=$3; shift 3
echo "[$(date '+%F %T')] waiting for $PREV to finish ..."
while tmux has-session -t "$PREV" 2>/dev/null; do sleep 60; done
if ! tail -n 50 "$PREV_LOG" | grep -q "PRETRAIN DONE"; then
  echo "[$(date '+%F %T')] $PREV ended WITHOUT 'PRETRAIN DONE'; not starting the next run"; exit 1
fi
echo "[$(date '+%F %T')] $PREV done; starting: python -u 01_pretrain.py $*"
cd "$(dirname "$0")"
python -u 01_pretrain.py "$@" 2>&1 | tee -a "$LOG"
