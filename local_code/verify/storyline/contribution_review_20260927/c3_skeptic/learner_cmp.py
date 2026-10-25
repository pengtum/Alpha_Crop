import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
import glob, numpy as np, pandas as pd
from scipy import stats
ML='/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml'
D1='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/decision_curve'
rows=pd.concat([pd.read_csv(f) for f in glob.glob(ML+'/rows_*.csv')],ignore_index=True)
print(rows.learner.value_counts().to_dict(), rows.method.unique()[:20], rows.cond.unique())
rows=rows[(rows.cond=='none')&(rows.method.isin(['srs','foreign_only']))&(rows.status=='ok')]
# LGB rows from G
GR=pd.concat([pd.read_csv(f) for f in glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/rows_t*.csv')],ignore_index=True)
GR=GR[(GR.cond=='none')&(GR.method.isin(['srs','foreign_only']))]
GR['learner']='lgb'
cols=['learner','method','clf','k','target','split','rep','oa','ba_ge30']
A=pd.concat([rows[cols],GR[cols]],ignore_index=True)
A.loc[A.method=='foreign_only','k']=0
lev=A.groupby(['learner','clf','k','target'])[['ba_ge30','oa']].mean().reset_index()
U=pd.read_csv(D1+'/decision_curve_units.csv')
H=U.pivot_table(index=['target','split','rep','clf','k'],columns='metric',values='head').reset_index()
H=H.groupby(['clf','k','target'])[['ba_ge30','oa']].mean().reset_index()
B=U[U.k==50].pivot_table(index=['target','split','rep','clf'],columns='metric',values='base_PT').reset_index().groupby(['clf','target'])[['ba_ge30','oa']].mean().reset_index()
def ci(v):
    v=np.asarray(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); t=stats.t.ppf(.975,len(v)-1)
    return f'{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] {int((v>0).sum())}/{len(v)}'
DROP={'RO','PT'}
for sub,name in ((None,'11'),(DROP,'9')):
  print('=== countries',name)
  for f in ['ae','s2','both']:
    for k in [0,50,200,400]:
      out=[]
      for L in ['lgb','lr','mlp','rf']:
        x=lev[(lev.learner==L)&(lev.k==k)]
        x=x if f=='both' else x[x.clf==f]
        x=x.groupby('target')[['ba_ge30','oa']].mean()
        if sub: x=x.drop(list(sub))
        out.append(f'{L} {x.ba_ge30.mean():.1f}/{x.oa.mean():.1f}')
      if k>0:
        h=H[H.k==k]; h=h if f=='both' else h[h.clf==f]; h=h.groupby('target')[['ba_ge30','oa']].mean()
        if sub: h=h.drop(list(sub))
        out.append(f'LGB+head {h.ba_ge30.mean():.1f}/{h.oa.mean():.1f}')
        # paired diff mlp retrain - lgb head
        m=lev[(lev.learner=='mlp')&(lev.k==k)]; m=m if f=='both' else m[m.clf==f]; m=m.groupby('target')[['ba_ge30','oa']].mean()
        if sub: m=m.drop(list(sub))
        d=(m-h.loc[m.index])
        out.append(f'| MLPretrain-LGBhead BA {ci(d.ba_ge30)} OA {ci(d.oa)}')
        l=lev[(lev.learner=='lr')&(lev.k==k)]; l=l if f=='both' else l[l.clf==f]; l=l.groupby('target')[['ba_ge30','oa']].mean()
        if sub: l=l.drop(list(sub))
        d2=(l-h.loc[l.index])
        out.append(f'| LRretrain-LGBhead BA {ci(d2.ba_ge30)} OA {ci(d2.oa)}')
      print(f,k,' ; '.join(out))
