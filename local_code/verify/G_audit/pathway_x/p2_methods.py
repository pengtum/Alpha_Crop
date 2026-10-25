# 方法层面：pred 这一对方法在 8 个方法的 28 种"2 对 6"划分中排第几；Friedman 方法异质性；重复 0 配对。
import glob, itertools
import numpy as np, pandas as pd
from scipy import stats
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
R = pd.concat([pd.read_csv(f) for f in glob.glob(OUT + "/rows_t[12]_*.csv")], ignore_index=True)
K = ['target', 'split', 'rep', 'method', 'clf', 'k']
a = R[R.cond == 'AE'].set_index(K).ba_ge30; b = R[R.cond == 'S2'].set_index(K).ba_ge30
d = (a - b).dropna().rename('d').reset_index(); d = d[d.rep == 0]
M = d.groupby(['target', 'method', 'clf', 'k']).d.mean().groupby(['target', 'method']).mean().unstack('method')
print(M.round(3).to_string()); print(M.mean().round(3).sort_values().to_string())
print('Friedman across 8 methods (rep 0 country values):', stats.friedmanchisquare(*[M[c] for c in M]))
# 两方法 vs 其余六方法的国家级配对对比，枚举 28 种
res = []
for pair in itertools.combinations(M.columns, 2):
    rest = [c for c in M.columns if c not in pair]
    v = M[list(pair)].mean(1) - M[rest].mean(1)
    n = len(v); t = v.mean() / (v.std(ddof=1) / np.sqrt(n))
    res.append((pair, v.mean(), t, 2 * stats.t.sf(abs(t), n - 1)))
res = sorted(res, key=lambda r: -r[2])
for i, r in enumerate(res[:6]): print(i + 1, r[0], f"diff {r[1]:+.3f} t {r[2]:+.2f} p {r[3]:.3f}")
rk = [i for i, r in enumerate(res) if set(r[0]) == {'margin', 'predstrat_srs'}][0]
print(f"pred pair rank by t: {rk+1}/28  -> 方法标签置换单侧 p = {(rk+1)/28:.3f}")
# 只含"单个方法"时的 t 分布（8 个方法各自国家级 t）
for c in M: 
    v = M[c]; t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))); print(f"{c:14s} mean {v.mean():+.3f} t {t:+.2f} p {2*stats.t.sf(abs(t),10):.3f}")
