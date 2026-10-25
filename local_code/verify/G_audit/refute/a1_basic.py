import glob, os, numpy as np, pandas as pd
from scipy import stats
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(OUT+'/rows_t*_*.csv')],ignore_index=True)
print(R.shape); print(R.columns.tolist())
print(R.groupby(['cond','method']).size().to_string())
print(R.k.value_counts().sort_index())
R.to_pickle('/Users/perng_mini/AlphaCrops_local/verify/G_audit/refute/R.pkl')
