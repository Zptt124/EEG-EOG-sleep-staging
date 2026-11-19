#!/bin/bash
# 用户要求（11-19）：self-cond 提特征（第 17 步）做完后先暂停队列，等用户说继续再 kill -CONT。
# 做法：第 17 步的 python 一启动就对队列 bash 发 SIGSTOP；正在跑的 python 不受影响，跑完后 bash 停住、不启动下一步。
QPID=${1:?usage: pause_queue_after_17.sh <queue bash pid>}
log() { echo "[$(date '+%F %T')] $*"; }
log "watching queue pid $QPID"
while kill -0 "$QPID" 2>/dev/null; do
  if pgrep -P "$QPID" -f "04_extract_features.py --run-name W4_D_selfcond" >/dev/null; then
    kill -STOP "$QPID"; log "step 17 started; queue pid $QPID STOPPED (resume: kill -CONT $QPID)"; exit 0
  fi
  M=$(pgrep -P "$QPID" -f "04_extract_features.py --run-name W4_D_main")
  if [ -n "$M" ]; then   # 第 17 步太快没抓到时的兜底：停住队列并中止主实验提特征（未写 DONE，之后会重提）
    kill -STOP "$QPID"; kill $M; log "missed step 17; queue STOPPED and main extraction pid $M killed"; exit 0
  fi
  sleep 5
done
log "ERROR: queue pid $QPID no longer exists"; exit 1
