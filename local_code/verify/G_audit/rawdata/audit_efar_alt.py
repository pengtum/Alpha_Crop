"""E_far：计划定义（类集合 = 全测试集 ≥30 的类）vs 替代定义（E_far 内 ≥30 的类），E1 通路 AE−S2 国家均值。"""
import glob, os
import numpy as np, pandas as pd
RES='/Users/perng_mini/AlphaCrops_local/results/guidance_G'; y=np.load('/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G/y.npy')
R=pd.concat([pd.read_csv(f) for f in glob.glob(f'{RES}/rows_t1_*.csv')])
E1=['lpm2_nogeo','kcenter','typiclust','kherding']
x=R[R.method.isin(E1)&R.cond.isin(['AE','S2'])&R.k.isin([50,200])].copy()
alt=[]
cache={}
for _,r in x.iterrows():
    key=(r.target,r.split)
    if key not in cache:
        S=np.load(f'{RES}/stage0/{r.target}_s{r.split}.npz'); yT=y[S['test']]; ef=S['efar'].astype(bool)
        cache[key]=(yT,ef,np.flatnonzero(np.bincount(yT[ef],minlength=10)>=30))
    yT,ef,bigf=cache[key]
    P=np.load(f"{RES}/probs/{r.target}_s{r.split}_r{r.rep}_{r.clf}_k{r.k}_{r.sel_hash}.npz")['i300']
    ph=P.argmax(1)[ef]; yy=yT[ef]
    alt.append(100*np.mean([(ph[yy==c]==c).mean() for c in bigf]))
x['ba_far_alt']=alt
for col in ('ba_far','ba_far_alt'):
    w=x.pivot_table(index=['target','split','rep','method','k','clf'],columns='cond',values=col); d=(w.AE-w.S2).groupby('target').mean()
    se=d.std(ddof=1)/np.sqrt(len(d))
    print(col, 'theta %.3f  SE %.3f'%(d.mean(),se), d.round(3).to_dict())
