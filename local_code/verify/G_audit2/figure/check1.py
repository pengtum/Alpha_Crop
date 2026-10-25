import numpy as np, pandas as pd, sys
import analyze_G as A
from guidance_G import E1, FAMILIES, OUT
from scipy import stats
R = A.load_rows()
print('rows', len(R))
# availability of conds by k, rep, method, clf
g = R.groupby(['cond']).agg(ks=('k', lambda s: sorted(set(s))), reps=('rep', lambda s: sorted(set(s))),
                            clfs=('clf', lambda s: sorted(set(s))), methods=('method', lambda s: sorted(set(s))), n=('k','size'))
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(g)
# units per cond
for c in R.cond.unique():
    sub = R[R.cond==c]
    print(c, 'units', sub.groupby(A.UNIT).ngroups, 'targets', sub.target.nunique())
# independent recompute: own pairing
def own(ca, cb, ms, reps=None, ks=(50,200), clfs=('s2','ae')):
    s = R[R.k.isin(ks) & R.clf.isin(clfs) & R.method.isin(ms)]
    if reps is not None: s = s[s.rep.isin(reps)]
    a = s[s.cond==ca].set_index(A.KEY).ba_ge30; b = s[s.cond==cb].set_index(A.KEY).ba_ge30
    d = (a-b).dropna()
    assert len(d)==len(a)==len(b), (ca,cb,len(a),len(b),len(d))
    cv = d.groupby(['target','method','clf','k']).mean().groupby('target').mean()
    n=len(cv); m=cv.mean(); se=cv.std(ddof=1)/np.sqrt(n); q=stats.t.ppf(.975,n-1)
    return len(d), n, m, m-q*se, m+q*se
items = [('geom all reps', 'AE','S2',E1,None),('pred r0','AE','S2',FAMILIES['pred'],[0]),('hyb r0','AE','S2',FAMILIES['hyb'],[0]),
 ('AE-S2XY','AE','S2XY',['lpm2_nogeo','kcenter'],[0]),('AE-XY','AE','XY',['lpm2_nogeo','kcenter'],[0]),('S2XY-S2','S2XY','S2',['lpm2_nogeo','kcenter'],[0]),
 ('ORACLE-S2','ORACLE','S2',['lpm2_nogeo','typiclust','margin'],[0]),
 ('RAWZ k50','RAWZ_AE','RAWZ_S2',E1,[0],(50,)),('NATIVE k50','NATIVE_AE','NATIVE_S2',['lpm2_nogeo','typiclust'],[0],(50,)),
 ('ORACLE2-S2 two','ORACLE2','S2',['lpm2_nogeo','typiclust'],[0]),
 ('ORACLE2-S2 two k50 only','ORACLE2','S2',['lpm2_nogeo','typiclust'],[0],(50,)),
]
for it in items:
    lab, ca, cb, ms, reps = it[:5]; ks = it[5] if len(it)>5 else (50,200)
    print(lab, [round(x,3) if isinstance(x,float) else x for x in own(ca,cb,ms,reps,ks)])
for clf in ('s2','ae'):
    print('clf',clf, own('AE','S2',E1,None,clfs=(clf,)))
for k in (50,200):
    print('k',k, own('AE','S2',E1,None,ks=(k,)))
# H ORACLE2 composite (as plot)
dH = pd.concat([A.deltas(R,'ORACLE2','S2',['lpm2_nogeo','typiclust'],reps=[0]), A.deltas(R,'ORACLE','S2',['margin'],reps=[0])])
print('H_ORACLE2', {k:round(v,3) for k,v in A.t_summary(A.country_values(dH)).items() if k in ('mean','lo95','hi95')})
# which k rows used by ORACLE2 composite
print('dH k counts', dH.groupby(['method','k']).size().to_dict())
# unit counts for pred/hyb rep0 & geom all
for nm, reps in (('geom',None),('pred',[0]),('hyb',[0])):
    d = A.deltas(R,'AE','S2',FAMILIES[nm],reps=reps); print(nm, 'units', d.groupby(A.UNIT).ngroups)
# A: country values + unit spread
d = A.deltas(R,'AE','S2',E1)
v = A.country_values(d).sort_values(); print(v.round(3).to_dict())
u = A.unit_level(d); print('unit range', u.min().round(3), u.max().round(3), 'n units', len(u))
print('units per target', u.groupby('target').size().to_dict())
pi = A.prediction_interval(v.to_numpy(), np.full(len(v), A.within_var(d))); print('PI', {k:round(x,3) for k,x in pi.items()})
s = A.t_summary(v); print({k:round(x,3) for k,x in s.items()})
# C
Q = pd.read_csv(OUT + '/guidance_quality_G.csv')
q = Q.pivot_table(index='target', columns='feat', values='knn10_ba')
gap = (q['ae']-q['s2']).reindex(v.index)
print('gap', gap.round(2).to_dict()); print('mean ae,s2', q.mean().round(3).to_dict())
from scipy.stats import spearmanr, pearsonr
print('pearson', pearsonr(gap, v)); print('spearman', spearmanr(gap, v))
keep = ~v.index.isin(['SE','PT']); print('r drop SE PT', pearsonr(gap[keep], v[keep]), spearmanr(gap[keep], v[keep]).correlation)
# leave-one-out r
for c in v.index:
    k2 = v.index != c; print('drop', c, round(np.corrcoef(gap[k2], v[k2])[0,1],3))
