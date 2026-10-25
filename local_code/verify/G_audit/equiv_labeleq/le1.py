import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
R = A.load_rows()
print(len(R))
# SRS learning curves
g = R[R.method.isin(['srs','foreign_only'])]
tab = g.groupby(['target','clf','k']).ba_ge30.agg(['mean','count','std']).reset_index()
pv = tab.pivot_table(index=['target','clf'], columns='k', values='mean')
cnt = tab.pivot_table(index=['target','clf'], columns='k', values='count')
print(cnt)
print(pv.round(2))
# gains
pv['g0_50']=pv[50]-pv[0]; pv['g50_100']=pv[100]-pv[50]; pv['g100_200']=pv[200]-pv[100]; pv['g200_400']=pv[400]-pv[200]
print(pv[['g0_50','g50_100','g100_200','g200_400']].round(3))
print(pv[['g0_50','g50_100','g100_200','g200_400']].describe().round(3))
# label equiv detail
rows=[]
dE1 = A.deltas(R,'AE','S2',A.E1); vE1=A.country_values(dE1); sE1=A.t_summary(vE1)
for t in sorted(R.target.unique()):
    for clf in ('s2','ae'):
        ks,f = A.srs_curve(R,t,clf)
        for k in (50,200):
            for name,dv in {'th':sE1['mean'],'L95':sE1['lo95'],'U95':sE1['hi95'],'+0.3':0.3,'-0.3':-0.3}.items():
                rows.append(dict(target=t,clf=clf,k=k,what=name,labels=A.label_equiv(ks,f,k,dv), f_iso=np.round(f,2).tolist()))
L=pd.DataFrame(rows)
L.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/equiv_labeleq/label_equiv_detail.csv',index=False)
for (w,k),gg in L.groupby(['what','k']):
    lab=gg.labels
    print(w,k,'median',np.median(lab),'n=',len(lab),'+inf',int(np.isposinf(lab).sum()),'-inf',int(np.isneginf(lab).sum()), 'finite q:',np.round(np.sort(lab[np.isfinite(lab)].values),1))
