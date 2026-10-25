#!/bin/zsh
# NATIVE s0 的 11 个单元完成后，停掉第一个驱动（避免进入 RAWZ 全部四方法），改跑 RAWZ s0 的 kcenter、kherding
cd /Users/perng_mini/AlphaCrops_local/verify/G_audit/tier3_verify
until [ $(grep -c "task done: NATIVE .* s0" run_t3.log) -ge 11 ]; do sleep 10; done
for p in $(lsof -t run_t3.log); do kill $p; done
sleep 3
for p in $(lsof -t run_t3.log); do kill -9 $p; done
PYTHONDONTWRITEBYTECODE=1 nice -n 10 /Users/perng_mini/AlphaCrops_local/.venv/bin/python run_t3b.py 3 > run_t3b.log 2>&1
echo handoff done
