import numpy as np, pandas as pd
from scipy import stats
exec(open('/Users/perng_mini/AlphaCrops_local/verify/G_audit/refute/a2_main.py').read().split("d=deltas('AE','S2',E1); v=cv(d)")[0])
import sys; sys.path.insert(0,'.')
import analyze_G as A
A.NOISE=False
L=[]
for t in sorted(R.target.unique()):
    for clf in ('s2','ae'):
        ks,f=A.srs_curve(R,t,clf)
        for k in (50,200):
            for name,dv in {'+0.3':.3,'-0.3':-.3,'theta':-0.054,'L95':-0.244,'U95':0.136,'+0.1':.1}.items():
                L.append(dict(target=t,clf=clf,k=k,what=name,lab=A.label_equiv(ks,f,k,dv)))
L=pd.DataFrame(L)
T=L.pivot_table(index=['target','clf'],columns=['k','what'],values='lab')
print(T.round(0).to_string())
print(L.groupby(['what','k']).lab.median().unstack())
print('fraction inf for +0.3 at k=200:', (L[(L.what=='+0.3')&(L.k==200)].lab==np.inf).mean())
print('fraction -inf for -0.3 at k=50:', (L[(L.what=='-0.3')&(L.k==50)].lab==-np.inf).mean())
# country-level: label-sensitive countries (SRS gain k0->k400 averaged over clf > 1pp)
S=R[R.method.isin(['srs','foreign_only'])].groupby(['target','clf','k']).ba_ge30.mean().unstack('k')
gain=(S[400]-S[0]).groupby('target').mean(); gain200=(S[200]-S[50]).groupby('target').mean()
d=deltas('AE','S2',E1); v=cv(d)
df=pd.DataFrame({'theta':v,'gain0_400':gain,'gain50_200':gain200})
print(df.round(2).sort_values('gain0_400').to_string())
print('spearman |theta| vs gain0_400', stats.spearmanr(df.theta.abs(),df.gain0_400))
for thr in [1.0,1.5,2.0]:
    sub=df[df.gain0_400>thr].theta; 
    if len(sub)>2: show(f'gain0_400>{thr} ({len(sub)}国) [探索]', ts(sub))
# Delta relative to gain: theta / gain50_200 per country
print((df.theta/df.gain50_200).round(2).to_dict())
