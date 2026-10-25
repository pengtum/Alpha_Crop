import glob, sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(OUT+'/rows_t*_*.csv')],ignore_index=True)
N=R[R.cond=='none']
print(N.groupby(['method','k']).size())
c=N.groupby(['target','method','k']).ba_ge30.mean().groupby(['method','k']).mean()
print('SRS curve (country-mean of ba_ge30):'); print(c.round(3))
c2=N.groupby(['target','method','k']).oa.mean().groupby(['method','k']).mean(); print(c2.round(3))
# per-country slope 50->200, 200->400
p=N[N.method=='srs'].groupby(['target','k']).ba_ge30.mean().unstack()
f=N[N.method=='foreign_only'].groupby('target').ba_ge30.mean()
p[0]=f; p=p[sorted(p.columns)]
print(p.round(2))
print('mean gain 0->50',(p[50]-p[0]).mean().round(3),' 50->200',(p[200]-p[50]).mean().round(3),' 200->400',(p[400]-p[200]).mean().round(3))
# method - SRS spread for all methods, AE and S2 (rep all), country values
U=['target','split','rep']
s=N[N.method=='srs'].set_index(U+['clf','k']).ba_ge30.rename('s')
for cond in ['AE','S2','ORACLE','XY','S2XY']:
    a=R[R.cond==cond].merge(s.reset_index(),on=U+['clf','k'])
    a['d']=a.ba_ge30-a.s
    v=a.groupby(['target','method','clf','k']).d.mean().groupby(['target','method']).mean().groupby('method').agg(['mean','std'])
    print(cond,'method-SRS (country mean, sd across countries):'); print(v.round(3).T)
