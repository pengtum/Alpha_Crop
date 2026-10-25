import os
os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
import glob, pandas as pd, numpy as np
D='/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml/'
fs=glob.glob(D+'rows_*_*.csv')
df=pd.concat([pd.read_csv(f,usecols=['learner','cond','method','clf','k','target','split','rep','status','oa','ba_ge30']) for f in fs])
print(df.method.unique()[:30], df.cond.unique())
d=df[(df.status=='ok')&(df.method.isin(['srs','foreign_only']))]
# unit mean over split,rep ; country mean
g=d.groupby(['learner','clf','k','target'])[['oa','ba_ge30']].mean().reset_index()
g=g.groupby(['learner','k','target'])[['oa','ba_ge30']].mean().reset_index() # avg features
for sub,name in [(None,'11'),(['RO','PT'],'9')]:
    h=g if sub is None else g[~g.target.isin(sub)]
    base=h[h.k==0].set_index(['learner','target'])
    for L in ['lr','mlp','rf']:
        out=[]
        for k in [50,200,400]:
            x=h[(h.learner==L)&(h.k==k)].set_index(['learner','target'])
            dd=(x[['oa','ba_ge30']]-base.loc[x.index,['oa','ba_ge30']])
            out.append(f"k{k}: BA {dd.ba_ge30.mean():+.2f} OA {dd.oa.mean():+.2f} (OA>0 {int((dd.oa>0).sum())}/{len(dd)})")
        b0=base.loc[L]
        print(name,L,'k0 BA %.2f OA %.2f'%(b0.ba_ge30.mean(),b0.oa.mean()),' | '.join(out))
