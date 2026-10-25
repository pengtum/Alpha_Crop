import glob, os
import numpy as np, pandas as pd
from scipy import stats
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT,'rows_t*_*.csv'))],ignore_index=True)
print(len(R), R.cond.value_counts().to_dict())
UNIT=['target','split','rep']; KEY=UNIT+['method','clf','k']
E1=['lpm2_nogeo','kcenter','typiclust','kherding']; PRED=['margin','predstrat_srs']; HYB=['gist_margin','badge_c']
def deltas(a_c,b_c,ms,met='ba_ge30',reps=None):
    a=R[(R.cond==a_c)&R.method.isin(ms)].set_index(KEY)[met]
    b=R[(R.cond==b_c)&R.method.isin(ms)].set_index(KEY)[met]
    d=(a-b).dropna().rename('d').reset_index()
    if reps is not None: d=d[d.rep.isin(reps)]
    return d
def cv(d): return d.groupby(['target','method','clf','k']).d.mean().groupby('target').mean()
def ts(v):
    v=np.asarray(v,float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
    q95=stats.t.ppf(.975,n-1); q90=stats.t.ppf(.95,n-1)
    return f"{m:+.3f} 95%[{m-q95*se:+.3f},{m+q95*se:+.3f}] 90%[{m-q90*se:+.3f},{m+q90*se:+.3f}] p={2*stats.t.sf(abs(m/se),n-1):.4f} pos={(v>0).sum()}/{n}"
# check units in pred
dp=deltas('AE','S2',PRED)
print('pred rows', len(dp), 'reps', dp.rep.unique(), 'units', dp.groupby(UNIT).ngroups)
dp0=deltas('AE','S2',PRED,reps=[0])
print('pred rep0 rows', len(dp0))
print('\n== pathway pred (rep0), ba_ge30 ==', ts(cv(dp0)))
print('per-country:', cv(dp0).round(3).to_dict())
for m in PRED:
    v=cv(dp0[dp0.method==m]); print(' method',m, ts(v)); print('   ', v.round(3).to_dict())
for met in ['ba_P2','ba_P1','ba_far','ba_i200','ba_i250','oa','f1_ge30']:
    print(f'pred {met}:', ts(cv(deltas('AE','S2',PRED,met,reps=[0]))))
    for m in PRED:
        print(f'   {m} {met}:', ts(cv(deltas('AE','S2',[m],met,reps=[0]))))
for clf in ['s2','ae']:
    print('pred clf',clf, ts(cv(dp0[dp0.clf==clf])))
for k in [50,200]:
    print('pred k',k, ts(cv(dp0[dp0.k==k])))
    for m in PRED:
        x=dp0[(dp0.k==k)&(dp0.method==m)]; print('   ',m,'k',k, ts(cv(x)))
v=cv(dp0)
print('\nleave-one-out:')
for c in v.index: print(' drop',c, ts(v.drop(c)))
print(' drop check2', ts(v.drop(['PT','SE','SK','LV'])))
print(' drop dev', ts(v.drop(['PT','BE','SI'])))
# wilcoxon
print('wilcoxon pred', stats.wilcoxon(v).pvalue)
# WCB
def webb(v,B=9999,seed=0):
    v=np.asarray(v,float); n=len(v); t0=v.mean()/(v.std(ddof=1)/np.sqrt(n))
    W=np.random.default_rng(seed).choice(np.r_[-np.sqrt(1.5),-1,-np.sqrt(.5),np.sqrt(.5),1,np.sqrt(1.5)],size=(B,n))
    ys=W*v[None,:]; t=ys.mean(1)/(ys.std(1,ddof=1)/np.sqrt(n)); return (np.abs(t)>=abs(t0)).mean()
print('WCB pred', webb(v.to_numpy()))
vps=cv(dp0[dp0.method=='predstrat_srs'])
print('WCB predstrat', webb(vps.to_numpy()), 'wilcoxon predstrat', stats.wilcoxon(vps).pvalue)
# geom on rep0 only for comparability
print('\ngeom rep0 only:', ts(cv(deltas('AE','S2',E1,reps=[0]))))
print('geom all:', ts(cv(deltas('AE','S2',E1))))
print('hyb rep0:', ts(cv(deltas('AE','S2',HYB,reps=[0]))))
pv={n:cv(deltas('AE','S2',ms,reps=[0] if n!='geom' else None)) for n,ms in [('geom',E1),('pred',PRED),('hyb',HYB)]}
print('E2:', ts(pd.concat(pv,axis=1).mean(1)))
# E2 without geom
print('pred - geom (country diff):', ts(pv['pred']-pv['geom']))
# vs SRS for pred methods each cond
S=R[R.method=='srs'].set_index(UNIT+['clf','k'])['ba_ge30']
print('\nvs SRS (rep0, ba_ge30):')
for cond in ['AE','S2']:
    for m in PRED+E1+HYB:
        a=R[(R.cond==cond)&(R.method==m)&(R.rep==0)].set_index(KEY)['ba_ge30'].reset_index()
        a['d']=a.ba_ge30.to_numpy()-S.reindex(pd.MultiIndex.from_frame(a[UNIT+['clf','k']])).to_numpy()
        print(f'  {cond} {m}-SRS:', ts(cv(a)))
