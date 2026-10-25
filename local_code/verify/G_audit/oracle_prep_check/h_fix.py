import numpy as np, pandas as pd, glob, sys, os
sys.path.insert(0, '.')
import analyze_G as A
AUD = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_prep_check'
R = pd.concat([pd.read_csv(f) for f in glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/rows_t*.csv')], ignore_index=True)
F = pd.concat([pd.read_csv(f) for f in glob.glob(AUD + '/retrain_rows/*.csv')], ignore_index=True)
clfs = sorted(F.clf.unique()); print('retrained rows', len(F), 'clfs', clfs)
key = ['target', 'split', 'rep', 'method', 'clf', 'k']
R2 = R.copy()
m = R2.merge(F[key + ['ba_fix']], on=key, how='left')
sel = (m.cond == 'ORACLE') & m.ba_fix.notna()
print('replaced', sel.sum())
m.loc[sel, 'ba_ge30'] = m.loc[sel, 'ba_fix']
def show(RR, ms, lab, clfs_=None):
    d = A.deltas(RR, 'ORACLE', 'S2', ms, reps=[0])
    if clfs_ is not None: d = d[d.clf.isin(clfs_)]
    s = A.t_summary(A.country_values(d))
    print(f'{lab:28s} {str(ms):45s} {s["mean"]:+.3f} [{s["lo95"]:+.3f}, {s["hi95"]:+.3f}] p={s["p"]:.3f} npos={s["n_pos"]} ncell={len(d)}')
    return A.country_values(d)
for c_ in ([clfs] if len(clfs) == 1 else [None, ['s2'], ['ae']]):
    print('--- clf subset', c_)
    for ms in (['lpm2_nogeo'], ['typiclust'], ['lpm2_nogeo', 'typiclust'], ['lpm2_nogeo', 'typiclust', 'margin']):
        show(R, ms, 'ORIG', c_); show(m, ms, 'FIX(plan z-clip)', c_)
# 逐国
d0 = A.deltas(R, 'ORACLE', 'S2', ['lpm2_nogeo', 'typiclust'], reps=[0]); d1 = A.deltas(m, 'ORACLE', 'S2', ['lpm2_nogeo', 'typiclust'], reps=[0])
if len(clfs) == 1: d0 = d0[d0.clf.isin(clfs)]; d1 = d1[d1.clf.isin(clfs)]
print(pd.concat([d0.groupby(['target', 'method']).d.mean().unstack().add_suffix('_orig'), d1.groupby(['target', 'method']).d.mean().unstack().add_suffix('_fix')], axis=1).round(3))
# 逐格 FIX − ORIG 的离散度
o = R[(R.cond == 'ORACLE') & R.method.isin(['lpm2_nogeo', 'typiclust'])].set_index(key).ba_ge30
f = F.set_index(key).ba_fix
dd = (f - o).dropna()
print('cell FIX-ORIG: mean %.3f sd %.3f n %d' % (dd.mean(), dd.std(), len(dd)))
a = R[(R.cond == 'AE') & (R.rep == 0) & R.method.isin(['lpm2_nogeo', 'typiclust'])].set_index(key).ba_ge30
b = R[(R.cond == 'S2') & (R.rep == 0) & R.method.isin(['lpm2_nogeo', 'typiclust'])].set_index(key).ba_ge30
print('cell AE-S2 (same cells, rep0): sd %.3f' % (a - b).dropna().loc[dd.index.intersection((a-b).dropna().index)].std())
