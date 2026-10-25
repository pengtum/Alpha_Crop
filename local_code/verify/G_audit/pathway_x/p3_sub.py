import glob, itertools
import numpy as np, pandas as pd
from scipy import stats
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
R = pd.concat([pd.read_csv(f) for f in glob.glob(OUT + "/rows_t[12]_*.csv")], ignore_index=True)
K = ['target', 'split', 'rep', 'method', 'clf', 'k']
a = R[R.cond == 'AE'].set_index(K).ba_ge30; b = R[R.cond == 'S2'].set_index(K).ba_ge30
d = (a - b).dropna().rename('d').reset_index(); d = d[d.rep == 0]
G = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']; P = ['margin', 'predstrat_srs']
def cv(x, ms): return x[x.method.isin(ms)].groupby(['target', 'method', 'clf', 'k']).d.mean().groupby('target').mean()
def rep(v, lab):
    n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(.975, n - 1)
    sf = np.array(list(itertools.product([-1, 1], repeat=n))); pm = (np.abs((sf * v.values).mean(1)) >= abs(m) - 1e-12).mean()
    print(f"{lab:32s} {m:+.3f} [{m-q*se:+.3f},{m+q*se:+.3f}] t-p={2*stats.t.sf(abs(m/se),n-1):.3f} perm-p={pm:.3f} >0 {int((v>0).sum())}/{n}")
for k in (50, 200): x = d[d.k == k]; rep(cv(x, P) - cv(x, G), f'pred-geom_r0 | k={k}')
for c in ('s2', 'ae'): x = d[d.clf == c]; rep(cv(x, P) - cv(x, G), f'pred-geom_r0 | clf={c}')
# 单元层面 pred − geom（重复 0，22 单元，忽略国家聚类，仅参考）
u = (d[d.method.isin(P)].groupby(['target', 'split']).d.mean() - d[d.method.isin(G)].groupby(['target', 'split']).d.mean())
n = len(u); print('unit-level pred-geom_r0 (22, 参考)', round(u.mean(), 3), 'p', round(2 * stats.t.sf(abs(u.mean() / (u.std(ddof=1) / np.sqrt(n))), n - 1), 3))
