import glob, os
import numpy as np, pandas as pd
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT,'rows_t*_*.csv'))],ignore_index=True)
print(R.shape)
g=R[R.method.isin(['srs','foreign_only'])]
print(g.groupby(['method','k']).size())
print(g.groupby(['method','k']).apply(lambda d: d[['target','split','rep']].drop_duplicates().shape[0]))
# per country x clf x k mean over units
c=g.groupby(['target','clf','k']).ba_ge30.mean().unstack('k')
print(c.round(2))
gain=c.sub(c[0],axis=0)
print('gain per country x clf'); print(gain.round(3))
gc=gain.groupby('target').mean()
print('gain per country (avg over clf)'); print(gc.round(3))
print('median across countries'); print(gc.median().round(3))
print('mean across countries'); print(gc.mean().round(3))
# also by-unit matched: k=0 exists for which units?
print(g[g.k==0].groupby(['target','clf']).size().unstack())
print(g[g.k==400].groupby(['target','clf']).size().unstack())
print(g[g.k==50].groupby(['target','clf']).size().unstack())
