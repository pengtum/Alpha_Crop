"""审计 9：把受伪影影响国家的 ORACLE lpm2_nogeo/typiclust 行替换为 V2 几何的复训结果，复算 H。只读 rows，写 verify。"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
AUD = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle'
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t*_*.csv'))], ignore_index=True)
R = R[R.rep == 0]
V = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(AUD, 'a6_retrain_V2_*.csv'))], ignore_index=True)
V = V.rename(columns={'tgt': 'target'})
KEY = ['target', 'split', 'method', 'clf', 'k']
orc = R[R.cond == 'ORACLE'].set_index(KEY)
s2 = R[R.cond == 'S2'].set_index(KEY)
v = V.set_index(KEY)
cmp = pd.DataFrame({'orig': orc.ba_ge30.reindex(v.index), 'V2': v.ba_ge30, 'S2': s2.ba_ge30.reindex(v.index),
                    'oa_orig': orc.oa.reindex(v.index), 'oa_V2': v.oa})
print('V2 行数', len(v), '国家', sorted(V.target.unique()))
print('V2 − orig（ba_ge30）按国家×方法：')
print((cmp.V2 - cmp.orig).groupby(['target', 'method']).mean().round(3).to_string())
def ts(x):
    x = np.asarray(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); q = stats.t.ppf(.975, len(x) - 1)
    return f'{m:+.3f} [{m-q*se:+.3f},{m+q*se:+.3f}] {int((x>0).sum())}/{len(x)}'
def H(o, ms):
    d = (o.ba_ge30 - s2.ba_ge30.reindex(o.index)).dropna().rename('d').reset_index()
    d = d[d.method.isin(ms)]
    return d.groupby(['target', 'method', 'clf', 'k']).d.mean().groupby('target').mean()
o2 = orc.copy()
o2.loc[v.index, 'ba_ge30'] = v.ba_ge30
for lab, o in (('原始', orc), ('V2 替换', o2)):
    print(lab, 'H(3 方法):', ts(H(o, ['lpm2_nogeo', 'typiclust', 'margin'])), ' H_geom:', ts(H(o, ['lpm2_nogeo', 'typiclust'])))
    print('   各国 H:', H(o, ['lpm2_nogeo', 'typiclust', 'margin']).round(3).to_dict())
