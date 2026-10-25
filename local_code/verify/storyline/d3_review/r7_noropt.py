"""补算 NUMBERS 中缺失"去 RO/PT"值、但可计算的几行（θ_E1 分 k、M9b、几何 F 去 RO/PT）。只读已有结果。"""
import glob, os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np, pandas as pd
from scipy import stats
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/rows_t*_*.csv'))])
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']; UNIT = ['target', 'split', 'rep']; DROP = ['RO', 'PT']
def ts(v):
    n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    return f'{m:+.3f} [{m - stats.t.ppf(.975, n - 1) * se:+.3f}, {m + stats.t.ppf(.975, n - 1) * se:+.3f}] 90%[{m - stats.t.ppf(.95, n - 1) * se:+.3f}, {m + stats.t.ppf(.95, n - 1) * se:+.3f}] n={n}'
sub = R[R.method.isin(E1) & R.clf.isin(['ae', 's2'])]
key = UNIT + ['method', 'clf', 'k']
d = (sub[sub.cond == 'AE'].set_index(key)['ba_ge30'] - sub[sub.cond == 'S2'].set_index(key)['ba_ge30']).dropna().rename('d').reset_index()
for k in (50, 200):
    v = d[d.k == k].groupby(['target', 'method', 'clf', 'k']).d.mean().groupby('target').mean()
    print(f'θ_E1 k={k} 去RO/PT:', ts(v.drop(DROP)))
p = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels/part1_rows.csv')
x = p[p.k == 0].pivot_table(index=['target', 'model'], columns='feat', values='ba_ge30')
dd = (x.ae - x.s2).unstack('model')
for mdl in ('lr', 'mlp2', 'mlp3'):
    print(f'M9b {mdl} k0 AE−S2: 11国', ts(dd[mdl].dropna()), '| 去RO/PT', ts(dd[mdl].drop(DROP).dropna()))
