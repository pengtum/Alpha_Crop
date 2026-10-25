import pandas as pd, numpy as np
from scipy import stats
import builtins
_p=builtins.print; builtins.print=lambda *a,**k:None
exec(open('recompute.py').read().split("print('=== H1')")[0])
builtins.print=_p
key=['target','unit','clf','k']
srs=df[(df.method=='srs')][key+['ba_ge30']].rename(columns={'ba_ge30':'srs'})
fo=df[df.method=='foreign_only'][['target','unit','clf','ba_ge30']].rename(columns={'ba_ge30':'fo'})
# SRS gains over foreign-only at country level
s=srs.merge(fo,on=['target','unit','clf']); s['gain']=s.srs-s.fo
g=s.groupby(['target','k']).gain.mean().unstack(); print('SRS gain over k=0 (country, avg clf & units)'); print(g.round(2))
print('median across countries:',g.median().round(3).to_dict(), ' mean:',g.mean().round(3).to_dict())
print('countries with SRS400 gain <0.3:',(g[400]<0.3).sum(),' SRS50 gain<0.3:',(g[50]<0.3).sum())
# ORACLE vs SRS and S2 per method, rep 0
OM=['lpm2_nogeo','typiclust','margin']
for c in ['ORACLE','AE','S2']:
    a=df[(df.cond==c)&df.method.isin(OM)&(df.rep==0)].merge(srs,on=key)
    a['d']=a.ba_ge30-a.srs
    cv=a.groupby(['target','method','clf','k']).d.mean().groupby(['target','method']).mean().unstack()
    print(c,'- SRS by method (country-mean then mean):',cv.mean().round(3).to_dict(),' overall', round(cv.mean(1).mean(),3))
    t=cv.mean(1); print('   t-test all-method', round(t.mean(),3), '95% half', round(stats.t.ppf(.975,10)*t.std(ddof=1)/np.sqrt(11),3))
# ORACLE - S2 by method
for mth in OM:
    d=delta('ORACLE','S2',[mth],reps=[0]); v=country_vals(d); print('ORACLE-S2',mth, round(v.mean(),3), '+-',round(stats.t.ppf(.975,10)*v.std(ddof=1)/np.sqrt(11),3))
# balanced one-way REML mixed model on unit-level values
m=delta('AE','S2',E1); u=m.groupby(['target','unit']).d.mean().unstack()
n,J=u.shape; gm=u.values.mean(); cm=u.mean(1)
MSB=J*((cm-gm)**2).sum()/(n-1); MSW=((u.values-cm.values[:,None])**2).sum()/(n*(J-1))
su2=(MSB-MSW)/J; print('mixed: mu',round(gm,4),'sigma_u2',round(su2,5),'SE',round(np.sqrt(MSB/(n*J)),4), 'MSB',MSB,'MSW',MSW)
