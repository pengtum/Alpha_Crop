# -*- coding: utf-8 -*-
"""调查区划分的定义敏感性：同一规格，KMeans 输入格顺序 = 主实现（factorize 首现顺序）或字典序（= 独立实现），
各用大 R（每划分 200 次、新种子）估 clus 型 E_S0 的 OA、先验 L1 与 SRS k=200 的 E_S1−E_S0，看"估计目标"差多少。"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recompute as RCP  # noqa: E402

RR = 200
C11 = RCP.CTR
C9 = [c for c in C11 if c not in ('RO', 'PT')]
D = RCP.G.GData()
out = []
for t in C11:
    for s in (0, 1):
        U = RCP.build(D, t, s)
        for part, creg in (('main', U['creg']), ('lex', U['creg2'])):
            reg = creg[U['cell']]
            rs = np.bincount(reg, minlength=8)
            for n0 in (100, 2000):
                M = RCP.N0M[n0]; ncl = n0 // M
                el = U['cnt'] >= M
                rng = np.random.default_rng(424242 + 1000 * C11.index(t) + 10 * s + n0)
                for r in range(RR):
                    ro = RCP.es(rng, rs)
                    m = 2
                    while m < 8 and (el & np.isin(creg, ro[:m])).sum() < ncl + 20:
                        m += 1
                    pool = np.flatnonzero(el & np.isin(creg, ro[:m]))
                    cells = pool[RCP.es(rng, U['cnt'][pool])[:ncl]]
                    S0 = np.concatenate([rng.choice(U['members'][c], M, replace=False) for c in cells])
                    s0m = np.zeros(U['n'], bool); s0m[S0] = True
                    S1 = rng.choice(np.flatnonzero(~s0m), 200, replace=False)
                    e0, e1 = RCP.lap(U['yc'][S0]), RCP.lap(U['yc'][S1])
                    row = dict(target=t, split=s, part=part, n0=n0, r=r, nreg=m, L1=np.abs(e0 - U['pi_test']).sum())
                    for f in ('ae', 's2'):
                        o0 = RCP.score(U, f, e0)[0]; o1 = RCP.score(U, f, e1)[0]
                        row[f'oa_{f}'] = o0; row[f's1ms0_{f}'] = o1 - o0
                    out.append(row)
        print(t, s, flush=True)
df = pd.DataFrame(out)
df.to_csv(os.path.join(HERE, 'partition_sens.csv'), index=False)
for n0 in (100, 2000):
    for col in ('oa_ae', 'oa_s2', 'L1', 's1ms0_ae', 's1ms0_s2', 'nreg'):
        res = {}
        for part in ('main', 'lex'):
            q = df[(df.part == part) & (df.n0 == n0)]
            g = q.groupby(['target', 'split'])[col]
            m = g.mean().groupby('target').mean().reindex(C11)
            se = (np.sqrt((g.var(ddof=1) / g.count()).groupby('target').sum()) / 2).reindex(C11)
            res[part] = (m, se)
        d = res['main'][0] - res['lex'][0]; sed = np.sqrt(res['main'][1] ** 2 + res['lex'][1] ** 2)
        z = d / sed
        print(f'n0={n0} {col:9s} 11国 main {res["main"][0].mean():+.3f} lex {res["lex"][0].mean():+.3f} 差 {d.mean():+.3f} '
              f'(MC SE {np.sqrt((sed ** 2).sum()) / 11:.3f}); 9国 差 {d[C9].mean():+.3f}；单国 |z|>3: {list(d.index[z.abs() > 3])}；'
              f'单国差最大 {d.abs().idxmax()} {d[d.abs().idxmax()]:+.3f}')
