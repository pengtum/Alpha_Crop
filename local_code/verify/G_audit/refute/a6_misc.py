import numpy as np, pandas as pd
from scipy import stats
exec(open('/Users/perng_mini/AlphaCrops_local/verify/G_audit/refute/a2_main.py').read().split("d=deltas('AE','S2',E1); v=cv(d)")[0])
import sys; sys.path.insert(0,'.')
import analyze_G as A
d=deltas('AE','S2',E1)
# per-country theta by k and clf, and label equivalents of own theta
print('=== 各国 θ 按 k × 分类器 ===')
x=d.groupby(['target','clf','k']).d.mean().unstack(['clf','k'])
print(x.round(2).to_string())
rows=[]
for t in sorted(R.target.unique()):
    for clf in ('s2','ae'):
        ks,f=A.srs_curve(R,t,clf)
        for k in (50,200):
            dv=d[(d.target==t)&(d.clf==clf)&(d.k==k)].groupby('method').d.mean().mean()
            rows.append(dict(target=t,clf=clf,k=k,delta=dv,labels=A.label_equiv(ks,f,k,dv)))
LL=pd.DataFrame(rows); print(LL[LL.target.isin(['PT','SE','RO','SK','AT'])].round(2).to_string())
# pathways
print('\n=== 通路（rep0）与逐方法 TOST ===')
for m in ['margin','predstrat_srs','gist_margin','badge_c']:
    show(f'{m} AE-S2', ts(cv(deltas('AE','S2',[m],reps=[0]))))
    show(f'{m} AE-S2 ba_P2', ts(cv(deltas('AE','S2',[m],'ba_P2',reps=[0]))))
show('pred path', ts(cv(deltas('AE','S2',['margin','predstrat_srs'],reps=[0]))))
show('pred path P2', ts(cv(deltas('AE','S2',['margin','predstrat_srs'],'ba_P2',reps=[0]))))
show('E1 rep0 only', ts(cv(deltas('AE','S2',E1,reps=[0]))))
# E2 computation
P={}
P['geom']=cv(deltas('AE','S2',E1)); P['pred']=cv(deltas('AE','S2',['margin','predstrat_srs'],reps=[0])); P['hyb']=cv(deltas('AE','S2',['gist_margin','badge_c'],reps=[0]))
show('E2', ts(pd.concat(P,axis=1).mean(1)))
# all 8 methods equally weighted (not plan) 
show('8 方法等权 rep0 [探索]', ts(cv(deltas('AE','S2',E1+['margin','predstrat_srs','gist_margin','badge_c'],reps=[0]))))
# SE typiclust mechanism
print('\n=== SE typiclust 机制 ===')
q=R[(R.target=='SE')&(R.method.isin(['typiclust','srs']))&(R.cond.isin(['AE','S2','none']))]
print(q.groupby(['cond','method','k','clf'])[['ba_ge30','oa','comp_L1','n_classes_sel','ba_P2']].mean().round(2).to_string())
# rec per class SE typiclust AE vs S2
rc=[f'rec{i}' for i in range(10)]
print(q.groupby(['cond','method'])[rc].mean().round(1).to_string())
# comp_L1 / n_classes AE vs S2 by method, all countries
print('\n=== comp_L1 与 n_classes_sel（AE−S2，E1）===')
for met in ['comp_L1','n_classes_sel']:
    dd=deltas('AE','S2',E1,met); print(met, dd.groupby('method').d.mean().round(3).to_dict())
# correlation of cell-level delta ba with delta comp_L1
dc=deltas('AE','S2',E1,'comp_L1').set_index(KEY).d; db=d.set_index(KEY).d
print('corr(Δba, Δcomp_L1)=', np.corrcoef(db.loc[dc.index],dc)[0,1].round(3))
