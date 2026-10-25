import numpy as np, pandas as pd
from scipy import stats
exec(open('/Users/perng_mini/AlphaCrops_local/verify/G_audit/refute/a2_main.py').read().split("d=deltas('AE','S2',E1); v=cv(d)")[0])
G2=['lpm2_nogeo','typiclust']
show('ORACLE-S2 geom(lpm2,typi) [探索]', ts(cv(deltas('ORACLE','S2',G2,reps=[0]))))
show('ORACLE-AE geom(lpm2,typi) [探索]', ts(cv(deltas('ORACLE','AE',G2,reps=[0]))))
show('AE-S2 geom(lpm2,typi) rep0', ts(cv(deltas('AE','S2',G2,reps=[0]))))
show('H 报告版 (3 方法)', ts(cv(deltas('ORACLE','S2',['lpm2_nogeo','typiclust','margin'],reps=[0]))))
for k in [50,200]:
    dd=deltas('ORACLE','S2',G2,reps=[0]); show(f'ORACLE-S2 geom k={k}', ts(cv(dd[dd.k==k])))
# per country oracle-S2 geom
o=cv(deltas('ORACLE','S2',G2,reps=[0])); a=cv(deltas('AE','S2',G2,reps=[0]))
print(pd.DataFrame({'oracle_minus_S2':o,'AE_minus_S2':a}).round(2).to_string())
# SESOI in terms of within-cell selection SD
d=deltas('AE','S2',E1)
# ratio: how much between-method difference exists within guidance (e.g. best vs worst method, S2 guidance) at country level
for cond in ['AE','S2']:
    srs=R[R.method=='srs'].set_index(UNIT+['clf','k']).ba_ge30
    a2=R[(R.cond==cond)&R.method.isin(E1)].copy(); a2['d']=a2.ba_ge30.values-srs.reindex(pd.MultiIndex.from_frame(a2[UNIT+['clf','k']])).values
    mc=a2.groupby(['target','method']).d.mean().unstack()
    print(cond,'method-SRS by country'); print(mc.round(2).to_string())
