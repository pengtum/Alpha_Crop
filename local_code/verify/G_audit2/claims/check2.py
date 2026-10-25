import numpy as np, pandas as pd, os
from scipy import stats
import analyze_G as A
from guidance_G import E1, OUT
R = A.load_rows()
# SRS curves
g = R[R.method.isin(['srs','foreign_only'])]
print(g.groupby(['method','k']).size())
print(g[g.k==0].cond.unique(), g[g.k==0].method.unique())
rows=[]
for (t,c),gg in g.groupby(['target','clf']):
    piv = gg.pivot_table(index=['split','rep'], columns='k', values='ba_ge30')
    d50_400 = (piv[400]-piv[50]).dropna() if 400 in piv and 50 in piv else pd.Series(dtype=float)
    d0_400 = (piv[400]-piv[0]).dropna() if 0 in piv else pd.Series(dtype=float)
    def tt(x):
        if len(x)<2: return (x.mean(), np.nan, len(x))
        return (x.mean(), stats.ttest_1samp(x,0).pvalue, len(x))
    rows.append(dict(target=t, clf=c, g50_400=tt(d50_400)[0], p50_400=tt(d50_400)[1], n=tt(d50_400)[2], g0_400=tt(d0_400)[0], p0_400=tt(d0_400)[1], n0=tt(d0_400)[2]))
S = pd.DataFrame(rows); print(S.round(3).to_string())
# pred methods vs SRS
for cond in ['AE','S2','ORACLE']:
    for m in ['margin','predstrat_srs']:
        d = A.vs_srs(R, cond, [m]); d = d[d.rep==0]
        if len(d)==0: continue
        t = A.t_summary(A.country_values(d)); print('vs SRS', cond, m, round(t['mean'],3), [round(t['lo95'],3), round(t['hi95'],3)], t['n_pos'])
# SE per method E1
d = A.deltas(R,'AE','S2',E1)
print(d[d.target=='SE'].groupby('method').d.agg(['mean','count', lambda x:(x>0).sum()]))
print('per-method country means'); print(d.groupby(['target','method','clf','k']).d.mean().groupby(['method']).mean())
# Jaccard
M = pd.read_csv(os.path.join(OUT,'mediators_G.csv'))
J = M[M.cond=='AE']
print('jaccard by method,k mean', J.groupby(['method','k']).jaccard.mean().round(4).to_dict())
print('overall mean', J.jaccard.mean(), 'max', J.jaccard.max(), J.loc[J.jaccard.idxmax(), ['target','method','k']].to_dict())
print('E1 only mean', J[J.method.isin(E1)].jaccard.mean())
# variance decomposition within cell (unit,clf,k): method vs guidance for E1+srs? E1 only, AE and S2
sub = R[R.method.isin(E1) & R.cond.isin(['AE','S2']) & R.k.isin([50,200]) & R.clf.isin(['s2','ae'])].copy()
sub['cell'] = sub.target+'_'+sub.split.astype(str)+'_'+sub.rep.astype(str)+'_'+sub.clf+'_'+sub.k.astype(str)
sub['y'] = sub.ba_ge30 - sub.groupby('cell').ba_ge30.transform('mean')
tot = (sub.y**2).sum()
mm = sub.groupby(['cell','method']).y.transform('mean'); cc = sub.groupby(['cell','cond']).y.transform('mean')
print('within-cell SS share: method', (mm**2).sum()/tot, 'guidance', (cc**2).sum()/tot)
