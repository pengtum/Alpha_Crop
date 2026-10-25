#!/bin/bash
cd /Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_altmodels
PY=/Users/perng_mini/AlphaCrops_local/.venv/bin/python
( until grep -q "MLP DONE\|Traceback" mlp_fair_s0r0.log; do sleep 5; done; VTHREADS=2 $PY lgb_seeds.py AT,BE,DE,EE,LT,LV a > lgb_seeds_a.log 2>&1; echo "seeds a exit $? $(date)" ) &
( until grep -q "TREES DONE\|Traceback" lgb_trees.log; do sleep 5; done; VTHREADS=2 $PY lgb_seeds.py SK,SI,SE,RO,PT b > lgb_seeds_b.log 2>&1; echo "seeds b exit $? $(date)" ) &
wait; echo QUEUE2 DONE
