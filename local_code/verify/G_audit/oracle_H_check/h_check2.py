import glob, os
import numpy as np, pandas as pd
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT,'rows_t*_*.csv'))],ignore_index=True)
KEY=['target','split','rep','method','clf','k']
def deltas(a,b,ms,metric='ba_ge30',reps=[0]):
    A=R[(R.cond==a)&R.method.isin(ms)].set_index(KEY)[metric]
    B=R[(R.cond==b)&R.method.isin(ms)].set_index(KEY)[metric]
    d=(A-B).dropna().rename('d').reset_index(); return d[d.rep.isin(reps)]
d=deltas('AE','S2',['lpm2_nogeo'],reps=[0]); print('row SD AE-S2 lpm2 rep0', round(d.d.std(),3), len(d))
E1=['lpm2_nogeo','kcenter','typiclust','kherding']
c=deltas('AE','S2',E1,reps=[0,1]).groupby(['target','method','clf','k']).d.mean().groupby('target').mean()
print('theta_E1 country SD', round(c.std(ddof=1),3))
# per-country per-method ORACLE-S2
d=deltas('ORACLE','S2',['lpm2_nogeo','typiclust','margin'])
t=d.groupby(['target','method']).d.mean().unstack().round(2); print(t.to_string())
# per-country per-method ORACLE-S2 OA
d=deltas('ORACLE','S2',['lpm2_nogeo','typiclust','margin'],metric='oa')
print(d.groupby(['target','method']).d.mean().unstack().round(2).to_string())
# class composition of ORACLE margin vs S2 margin: comp_L1, n_classes_sel
for cond in ['ORACLE','S2','AE']:
    x=R[(R.cond==cond)&(R.method=='margin')&(R.rep==0)]
    print(cond,'margin comp_L1 mean',round(x.comp_L1.mean(),3),'n_classes_sel',round(x.n_classes_sel.mean(),2))
x=R[(R.method=='srs')&(R.rep==0)]; print('srs comp_L1',round(x.comp_L1.mean(),3),'n_classes',round(x.n_classes_sel.mean(),2))
