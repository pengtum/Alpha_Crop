import sys, os, glob
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
os.chdir(sys.path[0])
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
from guidance_G import FAMILIES, E1, PRED, HYB
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
R = A.load_rows(OUT)
print(len(R), R.cond.value_counts().to_dict())
def ts(v): 
    s = A.t_summary(v); return s
def show(lab, v):
    s = ts(v.dropna()); print(f'{lab:45s} {A.fmt(s)}')
    return s
# reproduce pred path
d = A.deltas(R,'AE','S2',PRED,reps=[0])
print('n deltas', len(d), d.groupby(['target']).size().unique())
v = A.country_values(d)
s = show('pred path', v)
print(v.round(3).to_dict())
# rep check: pred has only rep0 anyway
print('reps present for pred:', R[R.method.isin(PRED)].rep.unique())
for m in PRED:
    show(f'method {m}', A.country_values(d[d.method==m]))
for clf in ('s2','ae'):
    show(f'clf {clf}', A.country_values(d[d.clf==clf]))
for k in (50,200):
    show(f'k {k}', A.country_values(d[d.k==k]))
for m in PRED:
  for clf in ('s2','ae'):
    for k in (50,200):
        show(f'{m} clf={clf} k={k}', A.country_values(d[(d.method==m)&(d.clf==clf)&(d.k==k)]))
for sp in (0,1):
    show(f'split {sp}', A.country_values(d[d.split==sp]))
# gamma within pred
g = d.pivot_table(index=A.UNIT+['method','k'], columns='clf', values='d').dropna()
g['gamma']=(g['ae']-g['s2'])/2
show('gamma pred (ae-s2 clf)/2', g.reset_index().groupby(['target','method','k']).gamma.mean().groupby('target').mean())
# LOO
print('LOO:')
for c in v.index:
    s = ts(v.drop(c)); print(f'  drop {c}: {s["mean"]:+.3f} [{s["lo95"]:+.3f},{s["hi95"]:+.3f}] p={s["p"]:.3f}')
# wilcoxon, sign test
print('wilcoxon p', stats.wilcoxon(v).pvalue, 'sign test p', stats.binomtest(int((v>0).sum()), len(v)).pvalue)
print('WCB webb p', A.webb_wcb(v.to_numpy()))
# metrics
for met in ['oa','f1_ge30','ba_P1','oa_P1','ba_P2','oa_P2','ba_far','oa_far','ba_i200','ba_i250','oa_i200','oa_i250']:
    dm = A.deltas(R,'AE','S2',PRED,met,reps=[0])
    show(f'metric {met}', A.country_values(dm))
    for m in PRED:
        show(f'   {m} {met}', A.country_values(dm[dm.method==m]))
# mechanism split
dd = (A.deltas(R,'AE','S2',PRED,'ba_ge30',reps=[0]).set_index(A.KEY).d - A.deltas(R,'AE','S2',PRED,'ba_P2',reps=[0]).set_index(A.KEY).d).rename('d').reset_index()
show('pred Δ(P0)-Δ(P2)', A.country_values(dd))
for m in PRED:
    show(f'  {m} Δ(P0)-Δ(P2)', A.country_values(dd[dd.method==m]))
d.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/pred/pred_deltas.csv', index=False)
