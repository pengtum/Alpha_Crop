import glob, sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, '.')
import guidance_G as G
OUT = G.OUT
R = pd.concat([pd.read_csv(f) for f in glob.glob(f'{OUT}/rows_t*_*.csv')], ignore_index=True)
K = ['target', 'split', 'rep', 'method', 'clf', 'k']
def dd(a, b, ms, met='ba_ge30'):
    A = R[(R.cond == a) & R.method.isin(ms) & (R.rep == 0)]; B = R[(R.cond == b) & R.method.isin(ms) & (R.rep == 0)]
    m = A.merge(B, on=K); m['d'] = m[met + '_x'] - m[met + '_y']; return m
for m in ('lpm2_nogeo', 'typiclust', 'margin'):
    v = dd('ORACLE', 'S2', [m]).groupby('target').d.mean(); se = v.std(ddof=1) / np.sqrt(11)
    print('ORACLE−S2', m, round(v.mean(), 3), '95%CI', round(v.mean() - 2.228 * se, 3), round(v.mean() + 2.228 * se, 3))
    v = dd('ORACLE', 'none', [m])  # placeholder
# ORACLE vs SRS
S = R[R.method == 'srs'][['target', 'split', 'rep', 'clf', 'k', 'ba_ge30']]
for cond in ('ORACLE', 'S2', 'AE'):
    for m in ('lpm2_nogeo', 'typiclust', 'margin'):
        A = R[(R.cond == cond) & (R.method == m) & (R.rep == 0)].merge(S, on=['target', 'split', 'rep', 'clf', 'k'])
        print(cond, m, '− SRS 均值', round((A.ba_ge30_x - A.ba_ge30_y).mean(), 3))
# WCB 独立实现
E1 = G.E1
A = R[(R.cond == 'AE') & R.method.isin(E1)].merge(R[(R.cond == 'S2') & R.method.isin(E1)], on=K)
v = (A.ba_ge30_x - A.ba_ge30_y).groupby(A.target).mean().to_numpy()
t0 = v.mean() / (v.std(ddof=1) / np.sqrt(11))
Wv = np.r_[-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)]
W = np.random.default_rng(0).choice(Wv, size=(9999, 11)); ys = W * v
print('WCB p', ((np.abs(ys.mean(1) / (ys.std(1, ddof=1) / np.sqrt(11)))) >= abs(t0)).mean())
# EE / LV 候选池：两个重复是否相同
D = G.GData()
for t in ('EE', 'LV', 'SE', 'AT'):
    for s in (0, 1):
        Sx = dict(np.load(f'{OUT}/stage0/{t}_s{s}.npz'))
        a, b = G.pool_positions(Sx, s, 0), G.pool_positions(Sx, s, 1)
        print(t, s, 'n_cand', len(Sx['cand']), 'pool', len(a), '两重复池重合', round(len(np.intersect1d(a, b)) / len(a), 3))
