"""复核 H4 几何法 F(3,30)：国家 × 方法两因素无重复 ANOVA（与 skeptic/s1_checks.method_F 同定义），rep0 与全部重复。"""
import glob, os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np, pandas as pd
from scipy import stats
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/rows_t*_*.csv'))])
UNIT = ['target', 'split', 'rep']; E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
sub = R[R.k.isin([50, 200]) & R.clf.isin(['ae', 's2'])]
a = sub[sub.cond.isin(['AE', 'S2']) & sub.method.isin(E1)].copy()
s = sub[sub.method == 'srs'].set_index(UNIT + ['clf', 'k'])['ba_ge30']
a['d'] = a['ba_ge30'].to_numpy() - s.reindex(pd.MultiIndex.from_frame(a[UNIT + ['clf', 'k']])).to_numpy()
def F(CM):
    Y = CM.values; n, m = Y.shape; gm = Y.mean(); rm = Y.mean(1, keepdims=True); cm = Y.mean(0, keepdims=True)
    ssm = n * ((cm - gm) ** 2).sum(); sse = ((Y - rm - cm + gm) ** 2).sum()
    f = (ssm / (m - 1)) / (sse / ((m - 1) * (n - 1))); return f, stats.f.sf(f, m - 1, (m - 1) * (n - 1)), m - 1, (m - 1) * (n - 1)
for lab, reps in (('rep0', [0]), ('all', [0, 1])):
    d = a[a.rep.isin(reps)]
    CM = d.groupby(['target', 'method', 'cond', 'clf', 'k']).d.mean().groupby(['target', 'method']).mean().unstack('method')
    print(lab, 'F(%d,%d)=%.3f p=%.3f' % (F(CM)[2], F(CM)[3], F(CM)[0], F(CM)[1]), '| 去RO F=%.3f p=%.3f' % F(CM.drop('RO'))[:2])
