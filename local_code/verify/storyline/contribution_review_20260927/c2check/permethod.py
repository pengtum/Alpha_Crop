import numpy as np, pandas as pd
from scipy import stats
exec(open('/private/tmp/c2check/theta.py').read().split("Rs={'LGB'")[0])
Rs={'LGB':load_lgb()}
for l in ['lr','mlp','rf']: Rs[l.upper()]=load_ml(l)
rows=[]
for L,R in Rs.items():
    d=deltas(R,'AE','S2',E1)
    for m in E1:
        vm=d[d.method==m].groupby(['target','clf','k'])['d'].mean().groupby('target').mean()
        t=stats.ttest_1samp(vm.values,0)
        rows.append(dict(L=L,m=m,mean=vm.mean(),p=t.pvalue,npos=(vm>0).sum(),SE=vm['SE'],noSE=vm.drop('SE').mean()))
    # k-split
    for k in (50,200):
        dk=d[d.k==k]; v=cv(dk); print(L,'k',k,ts(v.values))
    # clf split (same-source interaction)
    for c in ('s2','ae'):
        dc=d[d.clf==c]; v=cv(dc); print(L,'clf',c,ts(v.values))
T=pd.DataFrame(rows)
p=T.p.values; o=np.argsort(p); n=len(p); adj=np.empty(n); run=0
for i,j in enumerate(o):
    run=max(run,(n-i)*p[j]); adj[j]=min(1,run)
T['holm16']=adj
print(T.round(3).to_string())
# typiclust pooled across learners (not independent)
