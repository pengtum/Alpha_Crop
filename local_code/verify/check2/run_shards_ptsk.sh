#!/bin/zsh
# 补充检查（事后探索）：PT、SK 上较低/较高份额是否会损害已有正增益
cd /Users/perng_mini/AlphaCrops_local/verify/check2
PY=/Users/perng_mini/AlphaCrops_local/.venv/bin/python
while pgrep -f "taskB_share.py --shard (LV|SE)" >/dev/null; do sleep 20; done
for s in "PT ae" "PT s2" "SK ae" "SK s2"; do
  set -- ${=s}
  $PY taskB_share.py --shard $1 $2 --shares 0.01,0.05,1.0 > log_$1_$2.txt 2>&1 &
  sleep 25
done
wait
echo PTSK_DONE
