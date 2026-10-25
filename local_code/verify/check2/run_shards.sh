#!/bin/zsh
cd /Users/perng_mini/AlphaCrops_local/verify/check2
PY=/Users/perng_mini/AlphaCrops_local/.venv/bin/python
for s in "LV ae" "LV s2" "SE ae" "SE s2"; do
  set -- ${=s}
  $PY taskB_share.py --shard $1 $2 > log_$1_$2.txt 2>&1 &
  sleep 25
done
wait
echo ALL_DONE
