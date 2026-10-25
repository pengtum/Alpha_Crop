import numpy as np, pandas as pd, glob, sys
sys.path.insert(0,'.')
import analyze_G as A
R = pd.concat([pd.read_csv(f) for f in glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/rows_t*.csv')])
print(len(R), A.KEY)
for ms in (['lpm2_nogeo','typiclust','margin'], ['lpm2_nogeo'], ['typiclust'], ['margin'], ['lpm2_nogeo','typiclust']):
    d = A.deltas(R,'ORACLE','S2',ms,reps=[0])
    cv = A.country_values(d)
    s = A.t_summary(cv)
    print(ms, 'H=%.3f [%.3f, %.3f] npos %d n=%d'%(s['mean'],s['lo95'],s['hi95'],s['n_pos'], len(d)))
d = A.deltas(R,'ORACLE','S2',['lpm2_nogeo','typiclust','margin'],reps=[0])
print(d.groupby(['target','method'])['d'].mean().unstack().round(3))
# ORACLE vs SRS for geometry methods
srs = R[(R.method=='srs')&(R.rep==0)]
print(srs.groupby(['k']).ba_ge30.mean())
o = R[(R.cond=='ORACLE')].groupby(['method','k']).ba_ge30.mean()
s2 = R[(R.cond=='S2')&(R.rep==0)&R.method.isin(['lpm2_nogeo','typiclust','margin'])].groupby(['method','k']).ba_ge30.mean()
print(pd.concat([o.rename('ORACLE'),s2.rename('S2')],axis=1))
