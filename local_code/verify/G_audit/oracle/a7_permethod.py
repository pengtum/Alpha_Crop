"""审计 7：逐方法、逐国的 ORACLE−S2 与 ORACLE−AE，指标 ba_ge30 / oa / ba_P2 / ba_P1；只读。"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t*_*.csv'))], ignore_index=True)
R = R[R.rep == 0]
KEY = ['target', 'split', 'rep', 'method', 'clf', 'k']
def d(a, b, ms, met):
    A = R[(R.cond == a) & R.method.isin(ms)].set_index(KEY)[met]; B = R[(R.cond == b) & R.method.isin(ms)].set_index(KEY)[met]
    return (A - B).dropna().rename('d').reset_index()
def cv(x): return x.groupby(['target', 'method', 'clf', 'k']).d.mean().groupby('target').mean()
def ts(v):
    v = np.asarray(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v)); q = stats.t.ppf(.975, len(v) - 1)
    return f'{m:+.3f} [{m-q*se:+.3f},{m+q*se:+.3f}] {int((v>0).sum())}/{len(v)}'
for met in ('ba_ge30', 'oa', 'ba_P2', 'ba_P1'):
    print(met)
    for m in ('lpm2_nogeo', 'typiclust', 'margin'):
        print(f'   ORACLE−S2 {m:11s}', ts(cv(d('ORACLE', 'S2', [m], met))), '   ORACLE−AE', ts(cv(d('ORACLE', 'AE', [m], met))))
tab = pd.concat({m: cv(d('ORACLE', 'S2', [m], 'ba_ge30')) for m in ('lpm2_nogeo', 'typiclust', 'margin')}, axis=1)
tab['H'] = tab.mean(1)
tab['AE-S2(3m)'] = cv(d('AE', 'S2', ['lpm2_nogeo', 'typiclust', 'margin'], 'ba_ge30'))
print(tab.round(3).to_string())
# 去掉 margin 的"几何 ORACLE"提升空间，以及去 RO / SE
Hg = cv(d('ORACLE', 'S2', ['lpm2_nogeo', 'typiclust'], 'ba_ge30'))
print('H_geom(lpm2+typiclust):', ts(Hg))
H = cv(d('ORACLE', 'S2', ['lpm2_nogeo', 'typiclust', 'margin'], 'ba_ge30'))
print('H 去 RO:', ts(H.drop('RO')), ' 去 SE:', ts(H.drop('SE')), ' 去 RO,SE:', ts(H.drop(['RO', 'SE'])))
# 单元级噪声底：同方法 AE−S2 在 lpm2_nogeo（两种随机设计）上单元值的 SD
u = d('AE', 'S2', ['lpm2_nogeo'], 'ba_ge30')
print('lpm2_nogeo AE−S2 单行 SD:', round(u.d.std(), 3), ' 国家值 SD:', round(cv(u).std(ddof=1), 3))
