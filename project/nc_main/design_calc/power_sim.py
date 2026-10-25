"""方案 v6 第 6 节功效表的模拟（设计依据，不进入任何检验或判定）。
每次从已有域（nc_v2_domain_vals.csv）有放回抽 n = 29 + 18 个 (g200, oa_ae, e200)，前 29 个记为 LCZ；
M = 2·Binomial(50, oa_ae/100)；H1 = 精确二项符号检验（单侧）；H2 = g200 ~ 1 + [LUCAS] + M 的 OLS，HC3，单侧 t（df n−3）；
Holm，α = 0.025。三种情形：无崩溃（去掉 RO、PT 的 15 域）；含崩溃（全部 17 域）；无崩溃且 d > 0 改为独立 Bernoulli(0.65)。"""
import os

os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
V = pd.read_csv(os.path.join(HERE, 'nc_v2_domain_vals.csv'))
N_LCZ, N_LUCAS, NSIM, SEED, ALPHA = 29, 18, 10000, 20261006, 0.025


def hc3_p(y, X):
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, XtX, X)
    Vc = XtX @ (X.T * (e ** 2 / (1 - h) ** 2)) @ X @ XtX
    return stats.t.cdf(b[2] / np.sqrt(Vc[2, 2]), X.shape[0] - 3)   # 单侧 β < 0


def holm(p1, p2, a=ALPHA):
    lo, hi = min(p1, p2), max(p1, p2)
    r_lo = lo <= a / 2
    r_hi = r_lo and hi <= a
    return ((p1 <= p2 and r_lo) or (p1 > p2 and r_hi)), ((p2 < p1 and r_lo) or (p2 >= p1 and r_hi))


def sim(pool, po=None):
    rng = np.random.default_rng(SEED)
    n = N_LCZ + N_LUCAS
    fam = np.r_[np.zeros(N_LCZ), np.ones(N_LUCAS)]
    oa, g, d = V.oa_ae.values[pool], V.g200.values[pool], V.e200.values[pool]
    R = np.zeros((NSIM, 2))
    for s in range(NSIM):
        ii = rng.integers(0, len(pool), n)
        M = rng.binomial(50, oa[ii] / 100) * 2.0
        p2 = hc3_p(g[ii], np.c_[np.ones(n), fam, M])
        pos = d[ii] > 0 if po is None else rng.uniform(size=n) < po
        p1 = stats.binom.sf(pos.sum() - 1, n, 0.5)
        R[s] = holm(p1, p2)
    return R.mean(0)


if __name__ == '__main__':
    nocol = np.flatnonzero(~V.domain.isin(['RO', 'PT']).values)
    for name, pool, po in (('无崩溃', nocol, None), ('含崩溃（RO、PT）', np.arange(len(V)), None), ('无崩溃，胜出比例 0.65', nocol, 0.65)):
        p = sim(pool, po)
        print(f'{name:18s} n={N_LCZ + N_LUCAS}  H1 {p[0]:.3f}  H2 {p[1]:.3f}')
