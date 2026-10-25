import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
import analyze_G as A
R = A.load_rows()
for met in ('ba_ge30','ba_i250','ba_i200'):
    d = A.deltas(R,'AE','S2',A.E1,met)
    for k in (200,):
        v=A.country_values(d[d.k==k]); s=A.t_summary(v)
        print(met,k,'%.3f 90%%[%.3f,%.3f]'%(s['mean'],s['lo90'],s['hi90']), A.verdict(s))
        v2=v.drop('PT'); s2=A.t_summary(v2)
        print('  noPT %.3f 90%%[%.3f,%.3f]'%(s2['mean'],s2['lo90'],s2['hi90']), A.verdict(s2))
# bootstrap over units of pooled mean curve -> +0.3 at k=200 labels, and over countries
g=R[R.method.isin(['srs','foreign_only'])].pivot_table(index=['target','clf','split','rep'],columns='k',values='ba_ge30')
ks=np.array([0,50,100,200,400.])
rng=np.random.default_rng(0)
targets=sorted(g.index.get_level_values(0).unique())
res={50:[],200:[]}; resm={50:[],200:[]}
for b in range(2000):
    tb=rng.choice(targets,len(targets),replace=True)
    parts=[]
    for t in tb:
        sub=g.loc[t]
        # resample units within country
        idx=rng.integers(0,len(sub),len(sub))
        parts.append(sub.iloc[idx].groupby(level='clf').mean())
    P=pd.concat(parts)
    c=P.mean()[[0,50,100,200,400]].values
    f=IsotonicRegression().fit_transform(ks,c)
    for k in (50,200):
        res[k].append(A.label_equiv(ks,f,k,0.3)); resm[k].append(A.label_equiv(ks,f,k,-0.3))
for k in (50,200):
    a=np.array(res[k]); b=np.array(resm[k])
    print('k',k,'+0.3 labels (mean curve, 2-level bootstrap): median %.1f, 2.5%% %.1f, 97.5%% %.1f, P(inf)=%.3f'%(np.median(a),np.percentile(a,2.5),np.percentile(a,97.5),np.isinf(a).mean()))
    print('     -0.3 labels: median %.1f, 2.5%% %.1f, 97.5%% %.1f, P(-inf)=%.3f'%(np.median(b),np.percentile(b,2.5),np.percentile(b,97.5),np.isinf(b).mean()))
    print('     share of budget for +0.3: median %.2f, P(>0.45)=%.3f'%(np.median(a)/k,(a/k>0.45).mean()))
