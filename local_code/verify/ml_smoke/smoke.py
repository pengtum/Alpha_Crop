"""冒烟测试：真实数据，SI s0 r0，只跑 3 组选样（k=0、AE lpm2_nogeo k=50、SRS k=200），三个学习器。不输出任何精度。"""
import os, sys, time
os.environ['ML_OUT'] = os.path.expanduser('~/AlphaCrops_local/verify/ml_smoke/out')
sys.path.insert(0, sys.argv[1])
import pandas as pd
import guidance_G as G
import multilearner_G as ML
ML.unit_jobs = lambda split, rep, _: [('none', 'foreign_only', 0), ('AE', 'lpm2_nogeo', 50), ('none', 'srs', 200)]
D = G.GData()
M = ML.build_manifest.__wrapped__() if hasattr(ML.build_manifest, '__wrapped__') else None
for L, thr in (('lr', 4), ('mlp', 4), ('rf', 4)):
    t = time.time()
    R = pd.DataFrame(ML.run_unit_ml(D, L, 'SI', 0, 0, thr, log=lambda s: print('  ', s, flush=True)))
    print(f'{L}: {len(R)} 行，{time.time() - t:.0f}s；列 {sorted(c for c in R.columns if c.startswith("info_") or c == "C")}；'
          f'C = {R.C.dropna().unique().tolist()}', flush=True)
