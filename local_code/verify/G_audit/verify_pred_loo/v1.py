import os, sys, itertools
import numpy as np, pandas as pd
from scipy import stats
sys.argv = [sys.argv[0]]
import analyze_G as A
from guidance_G import FAMILIES
R = A.load_rows()
print('rows', len(R))
d = A.deltas(R, 'AE', 'S2', FAMILIES['pred'], reps=[0])
print('pred delta rows', len(d), 'units per country', d.groupby('target')[['split','rep']].apply(lambda x: len(x.drop_duplicates())).to_dict())
print('cells per country', d.groupby('target').size().to_dict())
v = A.country_values(d)
print('country values:\n', v.round(3).to_dict())
s = A.t_summary(v)
print('full:', {k: round(x,4) if isinstance(x,float) else x for k,x in s.items()})
print('median', round(v.median(),3), 'trim20', round(stats.trim_mean(v,0.2),3), 'trim10', round(stats.trim_mean(v,0.1),3))
# sign test
npos = (v>0).sum(); n=len(v)
print('sign test 8/11 two-sided', stats.binomtest(int(npos), n, 0.5).pvalue)
# Wilcoxon
print('wilcoxon', stats.wilcoxon(v).pvalue)
# exact sign flip
vals = v.to_numpy(); obs = vals.mean()
cnt=0; tot=0
for signs in itertools.product([1,-1], repeat=n):
    m = (vals*np.array(signs)).mean(); tot+=1
    if abs(m) >= abs(obs)-1e-12: cnt+=1
print('exact sign-flip p (mean stat)', cnt/tot)
# sign flip using t stat
def tstat(x): return x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
to = tstat(vals); c2=0
for signs in itertools.product([1,-1], repeat=n):
    if abs(tstat(vals*np.array(signs))) >= abs(to)-1e-12: c2+=1
print('exact sign-flip p (t stat)', c2/tot)
print('WCB webb', A.webb_wcb(vals))
print('\nLOO:')
for c in v.index:
    ss = A.t_summary(v.drop(c))
    print(f'  drop {c}: mean {ss["mean"]:+.3f} CI95 [{ss["lo95"]:+.3f},{ss["hi95"]:+.3f}] p={ss["p"]:.4f} pos {ss["n_pos"]}/10')
print('\ndrop pairs with p>0.05 count:')
res=[]
for a,b in itertools.combinations(v.index,2):
    ss = A.t_summary(v.drop([a,b])); res.append((a,b,ss['mean'],ss['p']))
res = pd.DataFrame(res, columns=['a','b','mean','p'])
print('n pairs', len(res), 'p>0.05', (res.p>0.05).sum())
print(res[(res.a.isin(['RO','SE']))|(res.b.isin(['RO','SE']))].query("a=='RO' and b=='SE'"))
print('plan sensitivity subsets:')
for lab, drop in (('no check2', A.CHECK2), ('no RO', ['RO']), ('no dev', A.DEV)):
    ss = A.t_summary(v.drop(drop)); print(f'  {lab}: {ss["mean"]:+.3f} [{ss["lo95"]:+.3f},{ss["hi95"]:+.3f}] p={ss["p"]:.3f} pos {ss["n_pos"]}/{ss["n"]}')
d.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/verify_pred_loo/pred_cells.csv', index=False)
