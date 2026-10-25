#!/bin/bash
cd /Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_altmodels
PY=/Users/perng_mini/AlphaCrops_local/.venv/bin/python
VTHREADS=2 $PY mlp_fair.py 1 0 > mlp_fair_s1r0.log 2>&1; echo "s1r0 exit $? $(date)"
VTHREADS=2 $PY mlp_fair.py 0 0 > mlp_fair_s0r0.log 2>&1; echo "s0r0 exit $? $(date)"
