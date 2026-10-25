import sys, os
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
os.chdir(sys.path[0])
import numpy as np, pandas as pd
import analyze_G as A
from guidance_G import FAMILIES, E1, PRED, HYB
R = A.load_rows("/Users/perng_mini/AlphaCrops_local/results/guidance_G")
P = {}
dE1 = A.deltas(R,'AE','S2',E1)
g = dE1.pivot_table(index=A.UNIT+['method','k'], columns='clf', values='d').dropna(); g['gamma']=(g['ae']-g['s2'])/2
P['gamma'] = A.t_summary(g.reset_index().groupby(['target','method','k']).gamma.mean().groupby('target').mean())['p']
for clf in ('s2','ae'): P[f'E1|clf {clf}'] = A.t_summary(A.country_values(dE1[dE1.clf==clf]))['p']
for k in (50,200): P[f'E1|k={k}'] = A.t_summary(A.country_values(dE1[dE1.k==k]))['p']
for name, ms in FAMILIES.items():
    P[f'path {name}'] = A.t_summary(A.country_values(A.deltas(R,'AE','S2',ms,reps=None if name=='geom' else [0])))['p']
for lab, met in (('dP2','ba_P2'),('dP1','ba_P1'),('E_far','ba_far')):
    P[lab] = A.t_summary(A.country_values(A.deltas(R,'AE','S2',E1,met)))['p']
dd = A.deltas(R,'AE','S2',E1,'ba_ge30').set_index(A.KEY).d - A.deltas(R,'AE','S2',E1,'ba_P2').set_index(A.KEY).d
P['dP0-dP2'] = A.t_summary(A.country_values(dd.rename('d').reset_index()))['p']
ms=['lpm2_nogeo','kcenter']
for a,b in (('AE','S2'),('AE','S2XY'),('AE','XY'),('S2XY','S2')):
    P[f'XYblk {a}-{b}'] = A.t_summary(A.country_values(A.deltas(R,a,b,ms,reps=[0])))['p']
ms=['lpm2_nogeo','typiclust','margin']
P['oracle H'] = A.t_summary(A.country_values(A.deltas(R,'ORACLE','S2',ms,reps=[0])))['p']
P['oracle AE-S2 same'] = A.t_summary(A.country_values(A.deltas(R,'AE','S2',ms,reps=[0])))['p']
S = pd.Series(P).sort_values()
print(S.round(4).to_string())
def holm(p):
    p = np.asarray(p); m=len(p); o=np.argsort(p); adj=np.empty(m); run=0
    for i,j in enumerate(o):
        run=max(run,(m-i)*p[j]); adj[j]=min(1,run)
    return adj
def bh(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p)[::-1]; adj=np.empty(m); run=1
    for i,j in enumerate(o):
        r=m-i; run=min(run,p[j]*m/r); adj[j]=run
    return adj
fams = {
 'plan-listed (12)': ['gamma','path pred','path hyb','E1|k=50','E1|k=200','XYblk AE-S2XY','XYblk AE-XY','XYblk S2XY-S2','oracle H','dP2','dP0-dP2','E_far'],
 'all printed secondary t-tests (18)': list(S.index),
 'paths only non-primary (2)': ['path pred','path hyb'],
 'paths incl geom (3)': ['path geom','path pred','path hyb'],
}
for fn, keys in fams.items():
    p = S[keys].to_numpy()
    print(f'\n{fn}: m={len(keys)}')
    print(pd.DataFrame({'p':p,'holm':holm(p),'bh':bh(p)}, index=keys).sort_values('p').round(4).to_string())
