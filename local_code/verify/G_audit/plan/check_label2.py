import glob, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy import stats
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in glob.glob(f'{OUT}/rows_t*_*.csv')], ignore_index=True)
def inv(ks, f, y):
    # 最小的 k 使 f(k) >= y（分段线性）
    if y > f.max(): return np.inf
    if y < f.min(): return -np.inf
    for j in range(len(ks)):
        if f[j] >= y:
            if j == 0: return ks[0]
            if f[j] == f[j-1]: return ks[j]
            return ks[j-1] + (y - f[j-1]) * (ks[j] - ks[j-1]) / (f[j] - f[j-1])
vals = {'θ': -0.0539, 'L95': -0.244, 'U95': 0.136, '+0.3': 0.3, '−0.3': -0.3}
out = {}
for t in sorted(R.target.unique()):
    for c in ('s2', 'ae'):
        g = R[(R.target == t) & (R.clf == c) & R.method.isin(['srs', 'foreign_only'])].groupby('k').ba_ge30.mean()
        ks = g.index.to_numpy(float); f = IsotonicRegression().fit_transform(ks, g.to_numpy())
        for k in (50, 200):
            fk = np.interp(k, ks, f)
            for n, dv in vals.items():
                out.setdefault((n, k), []).append(inv(ks, f, fk + dv) - k)
for key, v in out.items():
    v = np.array(v); print(key, 'median', np.median(v), 'n_inf+', int(np.isposinf(v).sum()), 'n_inf-', int(np.isneginf(v).sum()), '/', len(v))
m, se = -0.0539, None
