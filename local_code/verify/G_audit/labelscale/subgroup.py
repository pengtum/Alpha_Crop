import sys; sys.argv=['x']
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
R=A.load_rows()
vE1=A.country_values(A.deltas(R,'AE','S2',A.E1))
c=R[R.method.isin(['srs','foreign_only'])].groupby(['target','k']).ba_ge30.mean().unstack('k')
gain400=(c[400]-c[0]); gain50_400=c[400]-c[50]
df=pd.DataFrame(dict(dE1=vE1,gain0_400=gain400,gain50_400=gain50_400)).round(3)
print(df.sort_values('gain0_400'))
print('spearman dE1 vs gain0_400', stats.spearmanr(df.dE1,df.gain0_400))
hi=df[df.gain0_400>=1]; lo=df[df.gain0_400<1]
for n,g in (('label-sensitive (gain400>=1pp)',hi),('flat (<1pp)',lo)):
    t=stats.ttest_1samp(g.dE1,0); print(n,len(g),'mean',round(g.dE1.mean(),3),'n<0',(g.dE1<0).sum(),'t p',round(t.pvalue,3))
# own-country delta in labels, avg over clf of per-clf curves (k=50,200 median)
rows=[]
for t in df.index:
    for clf in ('s2','ae'):
        ks,f=A.srs_curve(R,t,clf)
        for k in (50,200):
            rows.append(dict(target=t,clf=clf,k=k,own=A.label_equiv(ks,f,k,vE1[t]),sesoi=A.label_equiv(ks,f,k,0.3),sesoim=A.label_equiv(ks,f,k,-0.3)))
L=pd.DataFrame(rows)
print(L.groupby(['target','k'])[['own','sesoi','sesoim']].median().unstack('k').round(1))
