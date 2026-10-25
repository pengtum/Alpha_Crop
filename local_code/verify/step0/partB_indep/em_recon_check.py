"""核对：本实现的 EM 用法（π_train、em_prior、reweight）在 P0test_i300 底座上复现 G foreign_only 行的 oa_P1/ba_P1。"""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import partB_indep as M
D = M.G.GData(); y = np.asarray(D.y)
mx = 0.0
for tgt in M.COUNTRIES:
    for split in (0, 1):
        S = np.load(os.path.join(M.R_G, 'stage0', f'{tgt}_s{split}.npz'))
        yT = y[S['test']]; big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
        yF = y[S['foreign']]; cw = M.G.class_weights_from(yF)
        pit = np.bincount(yF, weights=cw[yF], minlength=10) / cw[yF].sum()
        g = pd.read_csv(os.path.join(M.R_G, f'rows_t1_{tgt}_s{split}_r0.csv'))
        for f in M.FEATS:
            Q = S[f'P0test_{f}_i300'].astype(np.float64); Q /= Q.sum(1, keepdims=True)
            pe = M.em_prior(Q, pit)[0]
            o, b = M.oa_ba(yT, M.reweight(Q, pe, pit)[0].argmax(1), big)
            r = g[(g.method == 'foreign_only') & (g.clf == f)].iloc[0]
            mx = max(mx, abs(o - r.oa_P1), abs(b - r.ba_P1))
print(f'EM 在 P0test_i300 上复现 G oa_P1/ba_P1：max|差| = {mx:.2e}（22 单元 × 2 特征）')
