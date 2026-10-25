"""审计 11：用当前 guidance_G.oracle_compute 在 RO s0（最小单元）重算 ORACLE，与存档逐项比较。只读，不写结果目录。"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import guidance_G as G
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D = G.GData()
tgt, split = 'RO', 0
S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
O = dict(np.load(os.path.join(OUT, 'oracle', f'{tgt}_s{split}.npz')))
pp = G.pool_positions(S, split, 0)
t0 = time.time()
o = G.oracle_compute(D, S['allT'][pp], threads=2, trees=300)
print(f'{time.time()-t0:.0f}s temp 存档 {float(O["temp"]):.6f} 重算 {o["temp"]:.6f}')
for k in ('P_oof', 'P_T', 'G'):
    print(k, 'max|Δ| =', float(np.abs(np.asarray(o[k], float) - np.asarray(O[k], float)).max()))
