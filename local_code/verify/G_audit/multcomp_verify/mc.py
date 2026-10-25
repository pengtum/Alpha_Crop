# 独立复核：pred 通路的多重比较问题（只读结果文件）
import sys, os, glob
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import numpy as np, pandas as pd
from scipy import stats
from guidance_G import E1, FAMILIES, OUT
print('OUT', OUT, 'FAMILIES', FAMILIES)
fs = glob.glob(os.path.join(OUT, 'rows_t*_*.csv'))
R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
print('rows', len(R))
UNIT = ['target','split','rep']; KEY = UNIT+['method','clf','k']
def deltas(a_, b_, methods, metric='ba_ge30', reps=None):
    a = R[(R.cond==a_)&R.method.isin(methods)].set_index(KEY)[metric]
    b = R[(R.cond==b_)&R.method.isin(methods)].set_index(KEY)[metric]
    d = (a-b).dropna().rename('d').reset_index()
    if reps is not None: d = d[d.rep.isin(reps)]
    return d
def cv(d, col='d'):
    return d.groupby(['target','method','clf','k'])[col].mean().groupby('target').mean()
def tp(v):
    v = np.asarray(v, float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
    return m, 2*stats.t.sf(abs(m/se), n-1), int((v>0).sum())
P = {}
dE1 = deltas('AE','S2',E1)
g = dE1.pivot_table(index=UNIT+['method','k'], columns='clf', values='d').dropna()
g['gamma']=(g['ae']-g['s2'])/2
P['gamma'] = tp(g.reset_index().groupby(['target','method','k']).gamma.mean().groupby('target').mean())
for clf in ('s2','ae'): P[f'E1|clf {clf}'] = tp(cv(dE1[dE1.clf==clf]))
for k in (50,200): P[f'E1|k={k}'] = tp(cv(dE1[dE1.k==k]))
pv = {}
for name in ('geom','pred','hyb'):
    pv[name] = cv(deltas('AE','S2',FAMILIES[name], reps=None if name=='geom' else [0]))
    P[f'path {name}'] = tp(pv[name])
for lab, met in (('dP2','ba_P2'),('dP1','ba_P1'),('E_far','ba_far')): P[lab]=tp(cv(deltas('AE','S2',E1,met)))
dd = deltas('AE','S2',E1,'ba_ge30').set_index(KEY).d - deltas('AE','S2',E1,'ba_P2').set_index(KEY).d
P['dP0-dP2'] = tp(cv(dd.rename('d').reset_index()))
ms=['lpm2_nogeo','kcenter']
for a,b in (('AE','S2'),('AE','S2XY'),('AE','XY'),('S2XY','S2')): P[f'XY {a}-{b}'] = tp(cv(deltas(a,b,ms,reps=[0])))
ms=['lpm2_nogeo','typiclust','margin']
P['oracle H'] = tp(cv(deltas('ORACLE','S2',ms,reps=[0])))
P['oracle AE-S2 same'] = tp(cv(deltas('AE','S2',ms,reps=[0])))
T = pd.DataFrame(P, index=['mean','p','npos']).T.sort_values('p')
pd.set_option('display.width',200)
print(T.round(4))
print('pred exact p =', repr(P['path pred'][1]))

def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,j in enumerate(o):
        run=max(run, min(1,(m-i)*p[j])); adj[j]=run
    return adj
def bh(p):
    p=np.asarray(p); o=np.argsort(p)[::-1]; m=len(p); adj=np.empty(m); run=1
    for i,j in enumerate(o):
        rank=m-i; run=min(run, p[j]*m/rank); adj[j]=run
    return adj
def hochberg(p):
    p=np.asarray(p); o=np.argsort(p)[::-1]; m=len(p); adj=np.empty(m); run=1
    for i,j in enumerate(o):
        run=min(run,(i+1)*p[j]); adj[j]=min(1,run)
    return adj
plan12 = ['gamma','path pred','path hyb','E1|k=50','E1|k=200','XY AE-S2XY','XY AE-XY','XY S2XY-S2','oracle H','dP2','dP0-dP2','E_far']
fams = {'plan12':plan12, 'all18':list(P), 'plan12+geom':plan12+['path geom'], 'paths3':['path geom','path pred','path hyb'], 'paths2':['path pred','path hyb']}
for fn, keys in fams.items():
    p = np.array([P[k][1] for k in keys]); ip = keys.index('path pred')
    print(f'{fn:12s} m={len(keys):2d}  pred: holm={holm(p)[ip]:.4f} bh={bh(p)[ip]:.4f} hochberg={hochberg(p)[ip]:.4f} sidak={1-(1-p[ip])**len(keys):.4f}  min other p={np.min(np.delete(p,ip)):.4f}  #p<.05={int((p<.05).sum())}')

# 其他检验方式的 pred 通路 p
v = pv['pred'].to_numpy()
print('pred country values', pv['pred'].round(3).to_dict())
print('Wilcoxon p', stats.wilcoxon(v).pvalue)
from itertools import product
flips = np.array(list(product([-1,1], repeat=len(v))))
T0 = abs(v.mean()); Tp = np.abs((flips*v).mean(1))
print('exact sign-flip permutation p', (Tp >= T0-1e-12).mean())
print('sign test p (8/11)', stats.binomtest(int((v>0).sum()), len(v)).pvalue)
# pred − geom 差（国家层面配对）
dg = (pv['pred'] - pv['geom']).dropna(); print('pred-geom', tp(dg))
# geom 仅 rep0 做对照
g0 = cv(deltas('AE','S2',E1,reps=[0])); print('geom rep0', tp(g0)); print('pred - geom(rep0)', tp((pv['pred']-g0).dropna()))
# pred 通路 OA
for met in ('oa','f1_ge30','ba_P2','ba_far','ba_i200','ba_i250'):
    print('pred', met, tp(cv(deltas('AE','S2',FAMILIES['pred'],met,reps=[0]))))
# pred 通路各方法/分类器/k
dp = deltas('AE','S2',FAMILIES['pred'],reps=[0])
for m in FAMILIES['pred']: print('method', m, tp(cv(dp[dp.method==m])))
for c in ('s2','ae'): print('clf', c, tp(cv(dp[dp.clf==c])))
for k in (50,200): print('k', k, tp(cv(dp[dp.k==k])))
print('n cells pred', dp.groupby(UNIT).ngroups, dp.rep.unique())
