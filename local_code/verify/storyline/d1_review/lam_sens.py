"""复核用敏感性（非 rev4 规格；只拟合决策头，不训练分类器）：
同一 D1 子样本（种子规则不变），λ ∈ {1e-4, 1e-2, 0.1}，看 k=400/1000 的曲线平台是否来自 λ。"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '2'
import numpy as np
import pandas as pd
from scipy import stats
import importlib.util

spec = importlib.util.spec_from_file_location('rc', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recheck_d1.py'))
# 只取函数，不重跑主循环
src = open(spec.origin).read().split('rows, ret, tight = [], [], []')[0]
ns = {'__file__': spec.origin}
exec(src, ns)
fit, score, pool_pos, seed, Y, R = ns['fit'], ns['score'], ns['pool_pos'], ns['seed'], ns['Y'], ns['R']
C = 10
rows = []
for t in ns['CTRY']:
    for sp in (0, 1):
        S = np.load(os.path.join(R, 'stage0', f'{t}_s{sp}.npz'))
        nc = len(S['cand'])
        allT, test = S['allT'], S['test']
        yT = Y[test]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= 30)
        for rp in (0, 1):
            pp = pool_pos(nc, sp, rp)
            yP = Y[allT[pp]]
            N = len(pp)
            for f in ('ae', 's2'):
                ZP = np.log(np.clip(S[f'P_T_{f}'][pp].astype(np.float64), 1e-12, None))
                ZT = np.log(np.clip(S[f'P_T_{f}'][nc:].astype(np.float64), 1e-12, None))
                o0, b0 = score(yT, ZT.argmax(1), big)
                for lam in (1e-4, 1e-2):
                    for k in (50, 200, 400, 1000):
                        for r in range(5):
                            pick = np.sort(np.random.default_rng(seed(t, sp, rp, k, r)).choice(N, k, replace=False))
                            s_, b_, _ = fit(ZP[pick], yP[pick], lam)
                            o, b = score(yT, (s_ * ZT + b_).argmax(1), big)
                            rows.append(dict(target=t, clf=f, lam=lam, k=k, dba=b - b0, doa=o - o0))
                    s_, b_, _ = fit(ZP, yP, lam)
                    o, b = score(yT, (s_ * ZT + b_).argmax(1), big)
                    rows.append(dict(target=t, clf=f, lam=lam, k=-1, dba=b - b0, doa=o - o0))
    print(t, flush=True)
D = pd.DataFrame(rows)
D.to_csv(os.path.join(os.path.dirname(spec.origin), 'lam_sens_rows.csv'), index=False)
cv = D.groupby(['lam', 'k', 'target', 'clf'])[['dba', 'doa']].mean().groupby(['lam', 'k', 'target']).mean().reset_index()
for lam in (1e-4, 1e-2):
    for drop in (False, True):
        c = cv[cv.lam == lam]
        if drop:
            c = c[~c.target.isin(['RO', 'PT'])]
        g = c.groupby('k')[['dba', 'doa']].mean()
        print(f'λ={lam} {"去RO/PT" if drop else "11国"}：' + '  '.join(
            f'{"全池" if k < 0 else "k" + str(k)}: BA {g.loc[k, "dba"]:+.2f} OA {g.loc[k, "doa"]:+.2f}' for k in g.index))
