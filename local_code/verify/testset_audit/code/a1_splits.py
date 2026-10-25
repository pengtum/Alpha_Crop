"""审计 1：阶段 A 与 G 的 stage0 划分是否逐位一致；重算 make_split 与存档是否一致；s2/s3 是否与 s0/s1 重叠；缓冲与去重。只读。"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import guidance_G as G
from stageA import stable_hash
from scipy.spatial import cKDTree
R = '/Users/perng_mini/AlphaCrops_local/results'
D = G.GData()
C = D.countries
print('countries', C)
# 1) stageA stage0 vs G stage0
for t in ['PT', 'BE', 'SI']:
    for s in [0, 1, 2]:
        a = np.load(f'{R}/stageA/stage0/{t}_s{s}.npz')
        gp = f'{R}/guidance_G/stage0/{t}_s{s}.npz'
        if os.path.exists(gp):
            g = np.load(gp)
            eq = {k: np.array_equal(a[k], g[k]) for k in ('test', 'cand', 'foreign', 'allT')}
            print(f'stageA vs G {t} s{s}:', eq, 'lat0', float(a['lat0']) == float(g['lat0']))
        else:
            sp = G.make_split(D, t, s)
            eq = {k: np.array_equal(a[k], sp[k]) for k in ('test', 'cand', 'foreign', 'allT')}
            print(f'stageA s{s} vs G.make_split recomputed {t} s{s} (no G file):', eq)
# 2) G stage0 archived vs recomputed make_split for all 11 x 2
bad = 0
for t in C:
    for s in [0, 1]:
        g = np.load(f'{R}/guidance_G/stage0/{t}_s{s}.npz')
        sp = G.make_split(D, t, s)
        ok = all(np.array_equal(g[k], sp[k]) for k in ('test', 'cand', 'foreign', 'allT', 'efar'))
        bad += not ok
        if not ok: print('MISMATCH', t, s)
print('G stage0 == make_split recomputed for all 22:', bad == 0)
