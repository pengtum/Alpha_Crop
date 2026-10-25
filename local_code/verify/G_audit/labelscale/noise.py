import glob, os
import numpy as np, pandas as pd
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT,'rows_t*_*.csv'))],ignore_index=True)
s=R[R.method=='srs']
p=s.pivot_table(index=['target','split','clf','k'],columns='rep',values='ba_ge30')
d=(p[1]-p[0])
print('SRS rep1-rep0 SD by k (pp):'); print(d.groupby('k').std().round(3))
print('implied single-point SD = SD/sqrt2:'); print((d.groupby('k').std()/np.sqrt(2)).round(3))
# k=0 rep0 vs rep1 identical?
f=R[R.method=='foreign_only'].pivot_table(index=['target','split','clf'],columns='rep',values='ba_ge30')
print('k0 rep1-rep0 max abs:', (f[1]-f[0]).abs().max())
# gains relative to k=50 anchor (same seed scheme)
c=R[R.method.isin(['srs','foreign_only'])].groupby(['target','k']).ba_ge30.mean().unstack('k')
g50=c.sub(c[50],axis=0)
print('gain vs k=50, avg over clf & units'); print(g50.round(3))
print('median across countries'); print(g50.median().round(3))
# per-country E1 delta vs SRS gains
print('R3-like: method-SRS for E1 methods by cond')
E1=['lpm2_nogeo','kcenter','typiclust','kherding']
m=R[R.method.isin(E1)&R.cond.isin(['AE','S2'])].set_index(['target','split','rep','clf','k'])
ss=s.set_index(['target','split','rep','clf','k']).ba_ge30
m=m.join(ss.rename('srs'))
m['d']=m.ba_ge30-m.srs
print(m.groupby('cond').d.mean().round(3))
cv=m.groupby(['cond','target']).d.mean().unstack('cond')
print(cv.round(3)); print(cv.mean().round(3))
