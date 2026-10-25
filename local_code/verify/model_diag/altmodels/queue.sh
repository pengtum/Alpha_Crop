#!/bin/bash
cd /Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels
PY=/Users/perng_mini/AlphaCrops_local/.venv/bin/python
until grep -q "PART1 DONE\|Traceback" part1b.log; do sleep 5; done
echo "part1 finished $(date)"
$PY run_part2.py lr > part2_lr.log 2>&1; echo "part2 lr exit $? $(date)"
$PY frontier.py > frontier.log 2>&1; echo "frontier exit $? $(date)"
$PY lgb_fitdiag.py SE,PT > fitdiag.log 2>&1; echo "fitdiag exit $? $(date)"
$PY run_part2.py mlp2 > part2_mlp2.log 2>&1; echo "part2 mlp2 exit $? $(date)"
echo QUEUE DONE
