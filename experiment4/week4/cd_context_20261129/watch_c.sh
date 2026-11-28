#!/bin/bash
# Poll GitHub every 10 min; exit when C publishes the preflight release, a new release from C appears,
# or C's context branch head changes. Prints what changed.
REPO=Zptt124/EEG-EOG-sleep-staging
BR=feat/cd-context-readout-20261129
cd ~/repos/w3_main || exit 1
git fetch -q origin "$BR" 2>/dev/null
HEAD0=$(git rev-parse "origin/$BR")
REL0=$(gh release list --repo $REPO --limit 50 | awk -F'\t' '{print $3}' | sort | tr '\n' ' ')
echo "watch start $(date '+%F %T') head=$HEAD0"
while true; do
  sleep 600
  git fetch -q origin "$BR" 2>/dev/null
  HEAD1=$(git rev-parse "origin/$BR")
  REL1=$(gh release list --repo $REPO --limit 50 | awk -F'\t' '{print $3}' | sort | tr '\n' ' ')
  if [ "$HEAD1" != "$HEAD0" ] || [ "$REL1" != "$REL0" ]; then
    echo "CHANGE $(date '+%F %T')"
    echo "branch: $HEAD0 -> $HEAD1"
    comm -13 <(echo "$REL0" | tr ' ' '\n' | sort) <(echo "$REL1" | tr ' ' '\n' | sort) | sed 's/^/new release: /'
    git log --oneline "$HEAD0..$HEAD1" 2>/dev/null
    exit 0
  fi
  echo "no change $(date '+%T')"
done
