import glob, os, numpy as np, pandas as pd
from scipy import stats
G="/Users/perng_mini/AlphaCrops_local/results/guidance_G"
M="/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml"
E1=['lpm2_nogeo','kcenter','typiclust','kherding']
KEY=['target','split','rep','method','clf','k']
def load_lgb():
    R=pd.concat([pd.read_csv(f) for f in glob.glob(G+'/rows_t*_*.csv')],ignore_index=True)
    return R
def load_ml(l):
    R=pd.concat([pd.read_csv(f) for f in glob.glob(M+f'/rows_{l}_*.csv')],ignore_index=True)
    return R
def deltas(R,a,b,methods,metric='ba_ge30',ks=(50,200),clfs=('s2','ae')):
    sub=R[R.k.isin(ks)&R.clf.isin(clfs)&R.method.isin(methods)]
    A=sub[sub.cond==a].set_index(KEY)[metric]; B=sub[sub.cond==b].set_index(KEY)[metric]
    return (A-B).dropna().rename('d').reset_index()
def cv(d): return d.groupby(['target','method','clf','k'])['d'].mean().groupby('target').mean()
def ts(v):
    v=np.asarray(v,float);n=len(v);m=v.mean();se=v.std(ddof=1)/np.sqrt(n)
    q95=stats.t.ppf(.975,n-1);q90=stats.t.ppf(.95,n-1)
    return dict(mean=round(m,3),lo95=round(m-q95*se,3),hi95=round(m+q95*se,3),lo90=round(m-q90*se,3),hi90=round(m+q90*se,3),npos=int((v>0).sum()),n=n)
Rs={'LGB':load_lgb()}
for l in ['lr','mlp','rf']: Rs[l.upper()]=load_ml(l)
print(Rs['LGB'].cond.unique()); print(Rs['LR'].cond.unique())
gq=pd.read_csv(G+'/guidance_quality_G.csv')
knn=gq.pivot_table(index='target',columns='feat',values='knn10_ba',aggfunc='mean')
knnd=(knn.ae-knn.s2)
print('knn diff per country (mean over splits):');print(knnd.round(2).to_dict())
au=gq.pivot_table(index='target',columns='feat',values='auroc_margin',aggfunc='mean')
aud=au.ae-au.s2
print('AUROC AE-S2 per country:',aud.round(4).to_dict()); print('AUROC diff ts:',ts(aud.values))
print('AUROC means', au.mean().round(4).to_dict())
noROPT=[c for c in knnd.index if c not in('RO','PT')]
res={}
for L,R in Rs.items():
    d=deltas(R,'AE','S2',E1)
    v=cv(d); res[L]=v
    print(f'\n== {L} theta_E1',ts(v.values),' noROPT',ts(v[noROPT].values))
    # per method
    for m in E1:
        vm=d[d.method==m].groupby(['target','clf','k'])['d'].mean().groupby('target').mean()
        print(f'   {m:11s}',ts(vm.values), 'SE:',round(vm['SE'],2),'PT:',round(vm['PT'],2))
    # correlation with knn
    x=knnd.loc[v.index]; 
    r,p=stats.pearsonr(x,v); rs,ps=stats.spearmanr(x,v)
    xs=x.drop('SE'); vs=v.drop('SE'); r2,p2=stats.pearsonr(xs,vs)
    xo=x.drop(['RO','PT']); vo=v.drop(['RO','PT']); r3,p3=stats.pearsonr(xo,vo)
    print(f'   corr(knn adv, theta): r={r:.2f} p={p:.3f}; spearman={rs:.2f} p={ps:.3f}; noSE r={r2:.2f} p={p2:.3f}; noROPT r={r3:.2f} p={p3:.3f}')
    # OA
    do=deltas(R,'AE','S2',E1,metric='oa'); vo_=cv(do)
    print('   OA theta',ts(vo_.values))
pd.DataFrame(res).round(3).to_csv('/private/tmp/c2check/theta_by_country.csv')
print(pd.DataFrame(res).round(3))
