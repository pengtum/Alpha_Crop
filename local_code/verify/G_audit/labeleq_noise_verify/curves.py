import numpy as np, pandas as pd
from load import load
pd.set_option('display.width',250)
R=load()
S=R[R.method.isin(['srs','foreign_only'])][['target','split','rep','clf','k','ba_ge30']]
W=S.pivot_table(index=['target','clf','split','rep'],columns='k',values='ba_ge30')
ks=[0,50,100,200,400]
# curve = unit mean
C=W.groupby(['target','clf']).mean()
sd=W.groupby(['target','clf']).std(ddof=1)
print('level SD per (country,clf,k): median over all', np.nanmedian(sd[[50,100,200,400]].values), ' SE median', np.nanmedian(sd[[50,100,200,400]].values)/2)
print('level SD k=0 median (only 2 distinct values)', np.nanmedian(sd[0].values))
# paired differences within unit relative to k=50
for a,b in [(50,100),(100,200),(200,400),(50,200),(50,400),(0,50)]:
    d=(W[b]-W[a]).groupby(['target','clf'])
    m=d.mean(); s=d.std(ddof=1); n=d.count()
    t=m/(s/np.sqrt(n))
    print(f'{a}->{b}: mean gain median {m.median():+.3f}; within-unit SD median {s.median():.3f}; SE median {(s/np.sqrt(n)).median():.3f}; n curves with |t|<2: {(t.abs()<2).sum()}/22; gain<0: {(m<0).sum()}')
# monotonicity
C2=C[ks]
nonmono=(C2.diff(axis=1).iloc[:,1:]<0).any(axis=1)
print('non-monotone curves:', nonmono.sum(),'/',len(C2))
print('curves with drop 50->100:', (C2[100]<C2[50]).sum(), ' drop 0->50:', (C2[50]<C2[0]).sum(), ' drop 100->200', (C2[200]<C2[100]).sum(), ' drop 200->400', (C2[400]<C2[200]).sum())
out=C2.copy(); out['gain50_400']=C2[400]-C2[50]
d=(W[400]-W[50]).groupby(['target','clf']); out['t50_400']=d.mean()/(d.std(ddof=1)/np.sqrt(d.count()))
out['nonmono']=nonmono
print(out.round(2))
out.to_csv('curves_mine.csv')
W.to_csv('srs_units_mine.csv')
