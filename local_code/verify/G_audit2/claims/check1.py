import numpy as np, pandas as pd, os
from scipy import stats
import analyze_G as A
from guidance_G import E1, OUT
R = A.load_rows()
d = A.deltas(R, 'AE', 'S2', E1)
v = A.country_values(d)
s = A.t_summary(v); print('theta', s)
# C panel correlation
Q = pd.read_csv(os.path.join(OUT, 'guidance_quality_G.csv'))
q = Q.pivot_table(index='target', columns='feat', values='knn10_ba')
gap = (q['ae'] - q['s2']).reindex(v.index)
print('gap per country\n', gap.round(2).sort_values())
print('gap>0', (gap>0).sum(), 'mean gap', gap.mean(), 'ae', q['ae'].mean(), 's2', q['s2'].mean())
print('pearson', stats.pearsonr(gap, v)); print('spearman', stats.spearmanr(gap, v))
keep = ~v.index.isin(['SE','PT'])
print('pearson noSE/PT', stats.pearsonr(gap[keep], v[keep]), stats.spearmanr(gap[keep], v[keep]))
for drop in ['SE','PT']:
    kk = v.index != drop
    print('drop', drop, stats.pearsonr(gap[kk], v[kk]))
lr = stats.linregress(gap, v); print('slope', lr.slope, 'intercept', lr.intercept, 'zero cross', -lr.intercept/lr.slope)
# NATIVE diff in diff
two = ['lpm2_nogeo','typiclust']
def cv(a,b,ms,ks=(50,),reps=[0]):
    return A.country_values(A.deltas(R,a,b,ms,reps=reps,ks=ks))
nat = cv('NATIVE_AE','NATIVE_S2',two); main = cv('AE','S2',two)
print('NATIVE', A.t_summary(nat)); print('main subset', A.t_summary(main))
print('DiD NATIVE - main', A.t_summary((nat-main).dropna()))
print('NATIVE_AE-AE', A.t_summary(cv('NATIVE_AE','AE',two)))
print('NATIVE_S2-S2', A.t_summary(cv('NATIVE_S2','S2',two)))
raw = cv('RAWZ_AE','RAWZ_S2',E1); mainr = cv('AE','S2',E1)
print('RAWZ', A.t_summary(raw)); print('DiD RAWZ-main', A.t_summary(raw-mainr))
# pred path ba_far exact
P = ['margin','predstrat_srs']
for m in ['ba_ge30','ba_far','ba_P1','ba_P2','oa']:
    t = A.t_summary(A.country_values(A.deltas(R,'AE','S2',P,metric=m,reps=[0])))
    print('pred', m, {k: round(t[k],4) for k in ['mean','lo95','hi95','p','n_pos']})
