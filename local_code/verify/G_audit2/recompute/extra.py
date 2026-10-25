import glob, os, numpy as np, pandas as pd
from scipy import stats
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT,'rows_t*_*.csv')))],ignore_index=True)
M=pd.read_csv(os.path.join(OUT,'mediators_G.csv')); print('mediators conds', sorted(M.cond.unique()))
E1=['lpm2_nogeo','kcenter','typiclust','kherding']; PRED=['margin','predstrat_srs']
UNIT=['target','split','rep']; CELL=['method','clf','k']
def pair(a,b,ms,met='ba_ge30',ks=(50,200),reps=None):
    S=R[R.k.isin(ks)&R.clf.isin(['s2','ae'])&R.method.isin(ms)]
    if reps is not None: S=S[S.rep.isin(reps)]
    A=S[S.cond==a].set_index(UNIT+CELL)[met]; B=S[S.cond==b].set_index(UNIT+CELL)[met]
    return (A-B).dropna().rename('d').reset_index()
cv=lambda d: d.groupby(['target']+CELL).d.mean().groupby('target').mean()
def ts(v):
    v=np.asarray(v,float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
    return m, m-stats.t.ppf(.975,n-1)*se, m+stats.t.ppf(.975,n-1)*se, m-stats.t.ppf(.95,n-1)*se, m+stats.t.ppf(.95,n-1)*se, 2*stats.t.sf(abs(m/se),n-1)
v=cv(pair('AE','S2',PRED,'ba_far',reps=[0])); print('pred E_far', ['%.6f'%x for x in ts(v)])
d=pair('AE','S2',E1)
for m in E1: print('E1 method',m,['%.3f'%x for x in ts(cv(d[d.method==m]))])
v=cv(d[(d.k==200)&(d.target!='PT')]); print('k200 noPT',['%.3f'%x for x in ts(v)])
# SRS 增益 0->400 per country: unit-level (avg over clf), t-test over 4 units
N=R[R.cond=='none'].groupby(UNIT+['clf','k']).ba_ge30.mean().unstack('k')
for a,b in ((0,400),(50,400),(0,200)):
    g=(N[b]-N[a]).groupby(UNIT).mean().groupby('target')
    out={t:(round(x.mean(),2), round(2*stats.t.sf(abs(x.mean()/(x.std(ddof=1)/2)),3),3)) for t,x in g}
    print(f'gain {a}->{b} (mean, p over 4 units):',out)
# per-curve gain 50->400
G=R[R.cond=='none'].groupby(['target','clf','k']).ba_ge30.mean().unstack('k')
print((G[400]-G[50]).round(2).unstack().to_string())
print((G[400]-G[0]).round(2).unstack().to_string())
