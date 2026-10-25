import glob, os
import numpy as np, pandas as pd
from scipy import stats
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT,'rows_t*_*.csv'))],ignore_index=True)
print('rows',len(R)); print(R.groupby(['cond','method']).size().to_string())
UNIT=['target','split','rep']; KEY=UNIT+['method','clf','k']
def deltas(a,b,ms,metric='ba_ge30',reps=[0]):
    A=R[(R.cond==a)&R.method.isin(ms)].set_index(KEY)[metric]
    B=R[(R.cond==b)&R.method.isin(ms)].set_index(KEY)[metric]
    d=(A-B).dropna().rename('d').reset_index()
    return d[d.rep.isin(reps)]
def cv(d): return d.groupby(['target','method','clf','k']).d.mean().groupby('target').mean()
def ts(v):
    v=np.asarray(v,float);n=len(v);m=v.mean();se=v.std(ddof=1)/np.sqrt(n)
    q=stats.t.ppf(.975,n-1);q9=stats.t.ppf(.95,n-1)
    return f"{m:+.3f} 95%[{m-q*se:+.3f},{m+q*se:+.3f}] 90%[{m-q9*se:+.3f},{m+q9*se:+.3f}] p={2*stats.t.sf(abs(m/se),n-1):.3f} pos={int((v>0).sum())}/{n} halfw95={q*se:.3f} sd_ctry={v.std(ddof=1):.3f}"
ms=['lpm2_nogeo','typiclust','margin']
for metric in ['ba_ge30','oa','ba_P2','ba_P1','f1_ge30','ba_far']:
    print('\n==',metric)
    d=deltas('ORACLE','S2',ms,metric); print(' H (3 methods)  ', ts(cv(d)))
    for m in ms:
        dm=deltas('ORACLE','S2',[m],metric); print(f'  ORACLE-S2 {m:12s}', ts(cv(dm)))
    for m in ms:
        dm=deltas('AE','S2',[m],metric); print(f'  AE-S2     {m:12s}', ts(cv(dm)))
    dm=deltas('ORACLE','AE',ms,metric); print('  ORACLE-AE all  ', ts(cv(dm)))
# per-country H
d=deltas('ORACLE','S2',ms); c=cv(d); print('\nH per country\n',c.round(3).to_string())
for drop in [['RO'],['SE'],['RO','SE']]:
    print('drop',drop, ts(c.drop(drop)))
# row-level noise
for m in ['lpm2_nogeo','typiclust','margin','kcenter','kherding']:
    dd=deltas('AE','S2',[m],reps=[0,1]); print(f'row SD AE-S2 {m:12s} n={len(dd)} sd={dd.d.std():.3f}')
    do=deltas('ORACLE','S2',[m]); 
    if len(do): print(f'row SD ORACLE-S2 {m:12s} n={len(do)} sd={do.d.std():.3f} mean={do.d.mean():+.3f}')
# vs SRS
print()
srs=R[(R.method=='srs')&(R.rep==0)].set_index(['target','split','rep','clf','k'])
for cond in ['ORACLE','S2','AE']:
    for m in ms:
        for metric in ['ba_ge30','oa','ba_P2']:
            A=R[(R.cond==cond)&(R.method==m)&(R.rep==0)].set_index(['target','split','rep','clf','k'])[metric]
            dd=(A-srs[metric]).dropna().rename('d').reset_index()
            v=dd.groupby(['target','clf','k']).d.mean().groupby('target').mean()
            print(f'{cond:6s} {m:12s} {metric:8s} - SRS:', ts(v))
