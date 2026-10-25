# -*- coding: utf-8 -*-
"""裁决：对两实现 |z|>2 的 rand/none 项，用大 R（每划分 400 次，新种子）估计"真值"，看两实现各自偏离几个 SE。"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LE = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/'
      'Alpha_Crops/label_expansion')
sys.path.insert(0, LE)
import guidance_G as G  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ST0 = '/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0'
CTR = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
C9 = [c for c in CTR if c not in ('RO', 'PT')]
RR = 400
D = G.GData()
y = np.asarray(D.y)


def lap(v):
    n = np.bincount(v, minlength=10).astype(float)
    return (n + 1) / (n.sum() + 10)


out = []
for t in CTR:
    for s in (0, 1):
        S = np.load(os.path.join(ST0, f'{t}_s{s}.npz'))
        cand, test, fr = S['cand'], S['test'], S['foreign']
        n = len(cand); yc, yT = y[cand], y[test]
        yF = y[fr]; nF = np.bincount(yF, minlength=10)
        cw = np.where(nF > 0, len(yF) / (np.count_nonzero(nF) * np.maximum(nF, 1)), 1.0)
        pit = np.bincount(yF, weights=cw[yF], minlength=10) / cw[yF].sum()
        nT = np.bincount(yT, minlength=10); big = np.flatnonzero(nT >= 30); piT = nT / len(yT)
        P = {}
        for f in ('ae', 's2'):
            p = S[f'P_T_{f}'][n:].astype(np.float64); P[f] = p / p.sum(1, keepdims=True)

        def oa(f, pin):
            w = np.where(pit > 0, pin / np.where(pit > 0, pit, 1), 0)
            return 100 * ((P[f] * w).argmax(1) == yT).mean()
        rng = np.random.default_rng(777000 + 10 * CTR.index(t) + s)
        for r in range(RR):
            s1 = rng.choice(n, 200, replace=False)                 # none：SRS 200
            p1 = lap(yc[s1])
            row = dict(target=t, split=s, r=r, none_L1=np.abs(p1 - piT).sum())
            for f in ('ae', 's2'):
                row[f'none_oa_{f}'] = oa(f, p1)
            perm = rng.permutation(n)                              # rand n0=500 + SRS 200（不重叠）
            S0, S1 = perm[:500], perm[500:700]
            e0, e1, ep = lap(yc[S0]), lap(yc[S1]), lap(yc[np.r_[S0, S1]])
            for f in ('ae', 's2'):
                o0, o1, op = oa(f, e0), oa(f, e1), oa(f, ep)
                row[f'r500_S1mpool_{f}'] = o1 - op
                row[f'r500_S1mS0_{f}'] = o1 - o0
            out.append(row)
        print(t, s, flush=True)
df = pd.DataFrame(out)
df.to_csv(os.path.join(HERE, 'bigR.csv'), index=False)
