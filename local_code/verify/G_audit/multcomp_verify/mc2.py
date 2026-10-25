# 考虑检验间相关的族错误率校正（国家整体符号翻转，Westfall–Young maxT）+ pred 逐国剔除
import sys, os, glob
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import numpy as np, pandas as pd
from itertools import product
from scipy import stats
from guidance_G import E1, FAMILIES, OUT
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT,'rows_t*_*.csv'))], ignore_index=True)
UNIT=['target','split','rep']; KEY=UNIT+['method','clf','k']
def deltas(a_,b_,methods,metric='ba_ge30',reps=None):
    a=R[(R.cond==a_)&R.method.isin(methods)].set_index(KEY)[metric]; b=R[(R.cond==b_)&R.method.isin(methods)].set_index(KEY)[metric]
    d=(a-b).dropna().rename('d').reset_index()
    return d if reps is None else d[d.rep.isin(reps)]
def cv(d,col='d'): return d.groupby(['target','method','clf','k'])[col].mean().groupby('target').mean()
V={}
dE1=deltas('AE','S2',E1)
g=dE1.pivot_table(index=UNIT+['method','k'],columns='clf',values='d').dropna(); g['gamma']=(g['ae']-g['s2'])/2
V['gamma']=g.reset_index().groupby(['target','method','k']).gamma.mean().groupby('target').mean()
V['path pred']=cv(deltas('AE','S2',FAMILIES['pred'],reps=[0])); V['path hyb']=cv(deltas('AE','S2',FAMILIES['hyb'],reps=[0]))
for k in (50,200): V[f'k={k}']=cv(dE1[dE1.k==k])
ms=['lpm2_nogeo','kcenter']
for a,b in (('AE','S2XY'),('AE','XY'),('S2XY','S2')): V[f'{a}-{b}']=cv(deltas(a,b,ms,reps=[0]))
V['oracle H']=cv(deltas('ORACLE','S2',['lpm2_nogeo','typiclust','margin'],reps=[0]))
V['dP2']=cv(deltas('AE','S2',E1,'ba_P2'))
dd=deltas('AE','S2',E1,'ba_ge30').set_index(KEY).d-deltas('AE','S2',E1,'ba_P2').set_index(KEY).d
V['dP0-dP2']=cv(dd.rename('d').reset_index()); V['E_far']=cv(deltas('AE','S2',E1,'ba_far'))
M=pd.DataFrame(V).dropna(); print('countries', list(M.index), 'm', M.shape[1])
X=M.to_numpy(); n=X.shape[0]
def tstat(X): return X.mean(0)/(X.std(0,ddof=1)/np.sqrt(X.shape[0]))
t0=np.abs(tstat(X))
F=np.array(list(product([-1,1],repeat=n)))
T=np.abs(np.stack([tstat(f[:,None]*X) for f in F]))
praw=(T>=t0-1e-12).mean(0)
maxT=T.max(1); padj=np.array([(maxT>=t-1e-12).mean() for t in t0])
out=pd.DataFrame({'t':t0,'p_perm':praw,'p_maxT_FWER':padj},index=M.columns).sort_values('p_perm')
print(out.round(4))
# 检验间相关（国家值）
C=np.corrcoef(X.T); print('pred 与其他项的国家值相关：'); print(pd.Series(C[list(M.columns).index('path pred')],index=M.columns).round(2).to_string())
# pred 逐国剔除
v=V['path pred']
for c in v.index:
    w=v.drop(c).to_numpy(); m=w.mean(); se=w.std(ddof=1)/np.sqrt(len(w)); print(f'drop {c}: {m:+.3f} p={2*stats.t.sf(abs(m/se),len(w)-1):.3f}')
