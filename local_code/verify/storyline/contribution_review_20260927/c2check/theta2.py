import glob, numpy as np, pandas as pd
from scipy import stats
exec(open('/private/tmp/c2check/theta.py').read().split("Rs={'LGB'")[0])
Rs={'LGB':load_lgb()}
for l in ['lr','mlp','rf']: Rs[l.upper()]=load_ml(l)
for L,R in Rs.items():
    print(L, sorted(R.method.unique()), sorted(R.k.unique()), R.cond.unique().tolist()[:6])
PRED=['margin','predstrat_srs']
noROPT=['AT','BE','DE','EE','LT','LV','SE','SI','SK']
out=[]
for L,R in Rs.items():
    d=deltas(R,'AE','S2',PRED)
    if len(d):
        v=cv(d); print(f'{L} pred pathway AE-S2', ts(v.values), 'noROPT', ts(v[noROPT].values), 'units', d.groupby('rep').size().to_dict())
    # label value G(L,k) = SRS(k) - k0, per clf, avg over clf, per country
    base=R[(R.method=='foreign_only')&(R.k==0)].groupby(['target','split','clf'])['ba_ge30'].mean()
    for k in (50,200):
        s=R[(R.method=='srs')&(R.k==k)&(R.cond.isin(['none','S2','AE']))]
        s=s.drop_duplicates(subset=['target','split','rep','clf','k']).set_index(['target','split','clf'])['ba_ge30']
        g=(s-base.reindex(s.index)).groupby('target').mean()
        out.append(dict(L=L,k=k,**{kk:vv for kk,vv in ts(g.values).items() if kk in('mean','lo95','hi95')},noROPT=round(g[noROPT].mean(),2)))
print(pd.DataFrame(out))
# theta relative to geometric-methods-vs-SRS spread and to label value
