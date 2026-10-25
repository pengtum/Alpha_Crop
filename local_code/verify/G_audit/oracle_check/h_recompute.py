import glob, numpy as np, pandas as pd
from scipy import stats
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(OUT+'/rows_t*_*.csv')],ignore_index=True)
print(R.shape, R.cond.value_counts().to_dict())
U=['target','split','rep']; K=U+['method','clf','k']
O=R[R.cond=='ORACLE']
print('ORACLE rows',len(O), O.groupby('method').size().to_dict(), 'units',O.groupby(U).ngroups, 'reps',O.rep.unique())
print('dup keys', R.duplicated(K+['cond']).sum())
def tsum(v):
    v=np.asarray(v,float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n); q=stats.t.ppf(.975,n-1)
    return f'{m:+.3f} [{m-q*se:+.3f},{m+q*se:+.3f}] half={q*se:.3f} p={2*stats.t.sf(abs(m/se),n-1):.3f} pos={(v>0).sum()}/{n}'
def piv(cond,methods,metric='ba_ge30',reps=[0]):
    x=R[(R.cond==cond)&R.method.isin(methods)&R.rep.isin(reps)]
    return x.set_index(K)[metric]
def srs(metric='ba_ge30',reps=[0]):
    x=R[(R.method=='srs')&R.rep.isin(reps)].set_index(U+['clf','k'])[metric]
    return x
def cv(d):  # d: series indexed by K
    d=d.dropna().rename('d').reset_index()
    return d.groupby(['target','method','clf','k']).d.mean().groupby('target').mean()
ms=['lpm2_nogeo','typiclust','margin']
H=piv('ORACLE',ms)-piv('S2',ms)
print('H (my recompute):',tsum(cv(H)), 'n cells',H.dropna().shape)
A=piv('AE',ms)-piv('S2',ms); print('AE-S2 same methods:',tsum(cv(A)))
for m in ms:
    print(' method',m,'ORACLE-S2',tsum(cv(piv('ORACLE',[m])-piv('S2',[m]))),'| AE-S2',tsum(cv(piv('AE',[m])-piv('S2',[m]))))
geo=['lpm2_nogeo','typiclust']
print('H geometry-only (lpm2,typiclust):',tsum(cv(piv('ORACLE',geo)-piv('S2',geo))))
print('ORACLE-AE (3 methods):',tsum(cv(piv('ORACLE',ms)-piv('AE',ms))))
# vs SRS
S=srs()
def minus_srs(cond,methods,metric='ba_ge30',reps=[0]):
    a=piv(cond,methods,metric,reps).reset_index()
    s=srs(metric,reps).rename('s').reset_index()
    m=a.merge(s,on=U+['clf','k'])
    m['d']=m[metric]-m['s']
    return m.set_index(K)['d']
for c in ['ORACLE','AE','S2']:
    print(c,'-SRS (3 methods):',tsum(cv(minus_srs(c,ms))))
    for m in ms: print('   ',m,tsum(cv(minus_srs(c,[m]))))
# per k
for k in (50,200):
    x=(piv('ORACLE',ms)-piv('S2',ms)).reset_index(); x=x[x.k==k].set_index(K)['ba_ge30']
    print('H k=',k,tsum(cv(x)))
for clf in R.clf.unique():
    x=(piv('ORACLE',ms)-piv('S2',ms)).reset_index(); x=x[x.clf==clf].set_index(K)['ba_ge30']
    print('H clf=',clf,tsum(cv(x)))
# per country H
print(cv(H).round(3).to_dict())
# other metrics
for met in ['oa','ba_P2','ba_far']:
    if met in R: print('H',met,tsum(cv(piv('ORACLE',ms,met)-piv('S2',ms,met))))
