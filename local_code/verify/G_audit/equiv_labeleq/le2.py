import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
R = A.load_rows()
dE1 = A.deltas(R,'AE','S2',A.E1)
vE1 = A.country_values(dE1); s=A.t_summary(vE1)
print('pooled', {k:round(v,3) for k,v in s.items()})
vk={}
for k in (50,200):
    v = A.country_values(dE1[dE1.k==k]); vk[k]=v
    sk=A.t_summary(v)
    print('k',k,{kk:round(vv,3) for kk,vv in sk.items()}, 'verdict:',A.verdict(sk))
    # TOST p-values
    se=sk['se']; m=sk['mean']
    p_lo = stats.t.sf((m+0.3)/se,10); p_hi=stats.t.cdf((m-0.3)/se,10)
    print('   TOST p_lower=%.3f p_upper=%.3f max=%.3f'%(p_lo,p_hi,max(p_lo,p_hi)))
# pooled TOST p
m,se=s['mean'],s['se']
print('pooled TOST p', max(stats.t.sf((m+0.3)/se,10), stats.t.cdf((m-0.3)/se,10)))
# k difference
dd=(vk[50]-vk[200])
print('k50-k200 diff', {kk:round(vv,3) for kk,vv in A.t_summary(dd).items()})
print(pd.DataFrame({'k50':vk[50],'k200':vk[200]}).round(3))
# by clf x k
for clf in ('s2','ae'):
    for k in (50,200):
        v=A.country_values(dE1[(dE1.k==k)&(dE1.clf==clf)]); sk=A.t_summary(v)
        print(clf,k,'%.3f 90%%[%.3f,%.3f] 95%%[%.3f,%.3f]'%(sk['mean'],sk['lo90'],sk['hi90'],sk['lo95'],sk['hi95']), A.verdict(sk))
# SRS curve noise: unit-level gain 200->400 SD
g=R[R.method.isin(['srs','foreign_only'])].pivot_table(index=['target','clf','split','rep'],columns='k',values='ba_ge30')
gain=(g[400]-g[200])
se_gain=gain.groupby(['target','clf']).std()/2
print('SE of mean 200->400 gain per combo: median %.3f, range %.3f-%.3f'%(se_gain.median(),se_gain.min(),se_gain.max()))
gain50=(g[50]-g[0]); se50=gain50.groupby(['target','clf']).std()/2
print('SE of mean 0->50 gain per combo: median %.3f'%se50.median())
mg=gain.groupby(['target','clf']).mean()
print('mean 200->400 gain per combo  median %.3f mean %.3f; share <0.3: %d/22'%(mg.median(),mg.mean(),(mg<0.3).sum()))
# pooled curves
pv=g.groupby(['target','clf']).mean()
for lab,sub in (('all',pv),('noRO_PT',pv.drop(['RO','PT'],level=0))):
    mc=sub.mean(); mdc=sub.median()
    for nm,c in (('mean',mc),('median',mdc)):
        ks=np.array([0,50,100,200,400.]); from sklearn.isotonic import IsotonicRegression
        f=IsotonicRegression().fit_transform(ks,c[[0,50,100,200,400]].values)
        out={}
        for k in (50,200):
            out[k]={w:round(A.label_equiv(ks,f,k,dv),1) for w,dv in (('+0.3',0.3),('-0.3',-0.3),('th',s['mean']))}
        print(lab,nm,'curve',np.round(f,2),out)
