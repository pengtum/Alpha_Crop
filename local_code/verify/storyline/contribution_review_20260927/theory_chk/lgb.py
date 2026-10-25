import os
os.environ['OMP_NUM_THREADS']='2'
import glob,pandas as pd
fs=glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/rows_t*.csv')
cols=pd.read_csv(fs[0],nrows=1).columns.tolist(); print(cols[:25])
df=pd.concat([pd.read_csv(f) for f in fs])
d=df[df.method.isin(['srs','foreign_only'])]
print(d.groupby(['method','k']).size())
g=d.groupby(['clf','k','target'])[['oa','ba_ge30']].mean().reset_index().groupby(['k','target'])[['oa','ba_ge30']].mean().reset_index()
for sub,nm in [(None,'11'),(['RO','PT'],'9')]:
    h=g if sub is None else g[~g.target.isin(sub)]
    print(nm, h.groupby('k')[['oa','ba_ge30']].mean().round(2).to_dict())
