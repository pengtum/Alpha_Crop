import glob, numpy as np, pandas as pd
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(OUT+'/rows_t*_*.csv')],ignore_index=True)
print(R.shape)
g=R[R.method.isin(['srs','foreign_only'])]
print(g.groupby(['method','k']).size())
print(g.groupby(['method','k']).apply(lambda x: x[['target','split','rep']].drop_duplicates().shape[0]))
C=g.groupby(['target','clf','k']).ba_ge30.mean().unstack('k')
pd.set_option('display.width',200)
print(C.round(2))
# gains
G=pd.DataFrame({'0->50':C[50]-C[0],'50->100':C[100]-C[50],'50->200':C[200]-C[50],'200->400':C[400]-C[200],'50->400':C[400]-C[50]})
print(G.round(2))
print('mean'); print(G.mean().round(3)); print('median'); print(G.median().round(3))
Gx=G.drop(index=['PT','RO'],level=0)
print('excl PT RO mean', Gx.mean().round(3).to_dict(), 'median', Gx.median().round(3).to_dict())
C.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq_verify/srs_curve_means.csv')
