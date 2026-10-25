# -*- coding: utf-8 -*-
"""对账核对：自写 P0test_i300 底座上的 E_none、E_oracle、候选池先验（G 的 P2），与 G rows_t1 foreign_only 行及 recon_A.csv 比较。"""
import os
import sys

sys.dont_write_bytecode = True
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LE = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/'
      'Alpha_Crops/label_expansion')
sys.path.insert(0, LE)
import guidance_G as G  # noqa: E402

R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
A_DIR = '/Users/perng_mini/AlphaCrops_local/verify/step0/partA'
D = G.GData(); y = np.asarray(D.y)
RC = pd.read_csv(os.path.join(A_DIR, 'recon_A.csv')).set_index(['target', 'split', 'rep', 'feat'])
out = []
for t in 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(','):
    for s in (0, 1):
        S = np.load(os.path.join(R_G, 'stage0', f'{t}_s{s}.npz'))
        cand, test, fr = S['cand'], S['test'], S['foreign']; n = len(cand)
        yT = y[test]; nT = np.bincount(yT, minlength=10); big = np.flatnonzero(nT >= 30); piT = nT / len(yT)
        yF = y[fr]; nF = np.bincount(yF, minlength=10)
        cw = np.where(nF > 0, len(yF) / (np.count_nonzero(nF) * np.maximum(nF, 1)), 1.0)
        pit = np.bincount(yF, weights=cw[yF], minlength=10) / cw[yF].sum()

        def sc(P, pin):
            w = np.where(pit > 0, pin / np.where(pit > 0, pit, 1), 0)
            ph = (P * w).argmax(1); ok = ph == yT
            return 100 * ok.mean(), 100 * (np.bincount(yT[ok], minlength=10)[big] / nT[big]).mean(), ph
        for rep in (0, 1):
            g = pd.read_csv(os.path.join(R_G, f'rows_t1_{t}_s{s}_r{rep}.csv'))
            rng = np.random.default_rng(10_000 * s + 100 * rep + 7)
            pp = np.sort(rng.choice(n, min(30_000, n), replace=False))
            pipool = np.bincount(y[S['allT'][pp]], minlength=10) / len(pp)
            for f in ('ae', 's2'):
                gr = g[(g.method == 'foreign_only') & (g.clf == f)]
                assert len(gr) == 1
                gr = gr.iloc[0]
                P0 = S[f'P0test_{f}_i300'].astype(np.float32).astype(np.float64); P0 /= P0.sum(1, keepdims=True)
                PT = S[f'P_T_{f}'][n:].astype(np.float64); PT /= PT.sum(1, keepdims=True)
                on, bn, ph0 = sc(P0, pit); oo, bo, _ = sc(P0, piT); op, bp, _ = sc(P0, pipool)
                oT, bT, phT = sc(PT, pit)
                r = RC.loc[(t, s, rep, f)]
                out.append(dict(t=t, s=s, rep=rep, f=f, d_none_G=abs(on - gr.oa), d_ba_G=abs(bn - gr.ba_ge30),
                                d_P2=abs(op - gr.oa_P2), d_baP2=abs(bp - gr.ba_P2), orc_m_P2=oo - gr.oa_P2,
                                L1=np.abs(pipool - piT).sum(), nd=int((ph0 != phT).sum()), dPT=abs(oT - on),
                                d_recon=max(abs(on - r.oa_Enone_P0), abs(oo - r.oa_Eoracle_P0), abs(op - r.oa_Epool_P0),
                                            abs(bn - r.ba_Enone_P0), abs(bo - r.ba_Eoracle_P0), abs(oT - r.oa_Enone_PT)),
                                oracle_gain_P0=oo - on))
X = pd.DataFrame(out)
print(len(X), '行')
print(X[['d_none_G', 'd_ba_G', 'd_P2', 'd_baP2', 'd_recon', 'dPT']].max().to_string())
print('E_oracle(P0) − G oa_P2：均值 %.3f 范围 [%.3f, %.3f]；L1 均值 %.3f 范围 %.3f–%.3f；argmax 不同最大 %d' % (
    X.orc_m_P2.mean(), X.orc_m_P2.min(), X.orc_m_P2.max(), X.L1.mean(), X.L1.min(), X.L1.max(), X.nd.max()))
c = X[X.rep == 0].groupby(['t', 'f']).oracle_gain_P0.mean().unstack()
C9 = [k for k in c.index if k not in ('RO', 'PT')]
for f in ('ae', 's2'):
    print(f, 'P0 底座 oracle−none >0 国数', int((c[f] > 0).sum()), '/11；9 国', int((c.loc[C9, f] > 0).sum()), '/9；9 国均值 %.2f' % c.loc[C9, f].mean())
