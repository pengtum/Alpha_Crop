import glob, os
import numpy as np, pandas as pd
from scipy import stats
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT,'rows_t*_*.csv'))],ignore_index=True)
UNIT=['target','split','rep']; KEY=UNIT+['method','clf','k']
PRED=['margin','predstrat_srs']; E1=['lpm2_nogeo','kcenter','typiclust','kherding']
def deltas(a_c,b_c,ms,met='ba_ge30',reps=None):
    a=R[(R.cond==a_c)&R.method.isin(ms)].set_index(KEY)[met]
    b=R[(R.cond==b_c)&R.method.isin(ms)].set_index(KEY)[met]
    d=(a-b).dropna().rename('d').reset_index()
    if reps is not None: d=d[d.rep.isin(reps)]
    return d
def cv(d): return d.groupby(['target','method','clf','k']).d.mean().groupby('target').mean()
def ts(v):
    v=np.asarray(v,float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
    q95=stats.t.ppf(.975,n-1); q90=stats.t.ppf(.95,n-1)
    return f"{m:+.4f} 95%[{m-q95*se:+.3f},{m+q95*se:+.3f}] 90%[{m-q90*se:+.3f},{m+q90*se:+.3f}] p={2*stats.t.sf(abs(m/se),n-1):.4f} pos={(v>0).sum()}/{n}"
# P0-P2 for pred
a=deltas('AE','S2',PRED,'ba_ge30',[0]).set_index(KEY).d; b=deltas('AE','S2',PRED,'ba_P2',[0]).set_index(KEY).d
print('pred Δ(P0)-Δ(P2):', ts(cv((a-b).rename('d').reset_index())))
for m in PRED:
    a=deltas('AE','S2',[m],'ba_ge30',[0]).set_index(KEY).d; b=deltas('AE','S2',[m],'ba_P2',[0]).set_index(KEY).d
    print(f'  {m} Δ(P0)-Δ(P2):', ts(cv((a-b).rename('d').reset_index())))
# per split
dp=deltas('AE','S2',PRED,reps=[0])
for s in [0,1]:
    print('pred split',s, ts(cv(dp[dp.split==s])))
# unit-level sign
u=dp.groupby(UNIT).d.mean(); print('unit-level pos', (u>0).sum(), '/', len(u))
# cell-level
print('cell-level AE>S2 fraction', (dp.d>0).mean(), 'ties', (dp.d==0).mean(), 'n', len(dp))
# comp_L1 & n_classes sel
for m in PRED:
    for cond in ['AE','S2']:
        x=R[(R.cond==cond)&(R.method==m)]
        print(m,cond,'comp_L1',round(x.comp_L1.mean(),4),'n_cls',round(x.n_classes_sel.mean(),3))
# mediator: country-level foreign-model nll_pool diff vs pred delta
Q=pd.read_csv(os.path.join(OUT,'guidance_quality_G.csv'))
q=Q.pivot_table(index='target',columns='feat',values=['nll_pool','auroc_margin','prior_L1_argmax','ece_oof'])
dq=pd.DataFrame({k:(q[(k,'ae')]-q[(k,'s2')]) for k in ['nll_pool','auroc_margin','prior_L1_argmax','ece_oof']})
v=cv(dp); dq['pred']=v
print(dq.round(3))
for k in ['nll_pool','auroc_margin','prior_L1_argmax','ece_oof']:
    r,p=stats.spearmanr(dq[k],dq.pred); print('spearman pred vs Δ',k, round(r,3), round(p,3))
# shrinkage check predstrat
# Holm over secondary p-values listed in report
ps={'gamma':.865,'clf_s2':.637,'clf_ae':.549,'k50':.872,'k200':.310,'geom':.543,'pred':.031,'hyb':.454,'P2':.447,'P1':.437,'Efar':.699,'P0-P2':.963,'AE-S2xy':.271,'AE-S2XY':.989,'AE-XY':.273,'S2XY-S2':.596,'H':.953,'sameH':.325}
print('n secondary tests', len(ps), 'Bonferroni pred', min(1,.031*len(ps)), 'Bonf over 3 pathways', .031*3)
