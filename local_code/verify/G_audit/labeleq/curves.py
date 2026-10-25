import glob, os, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT,'rows_t*_*.csv'))],ignore_index=True)
print('rows',len(R))
S=R[R.method.isin(['srs','foreign_only'])]
print(S.groupby(['method','k','cond']).size())
# duplicates check
dup=S.duplicated(['target','split','rep','method','clf','k']).sum(); print('dups',dup)
# foreign_only identical across reps?
fo=S[S.method=='foreign_only'].pivot_table(index=['target','split','clf'],columns='rep',values='ba_ge30')
print('foreign_only rep0-rep1 max abs diff', (fo[0]-fo[1]).abs().max())
S.to_pickle('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/S.pkl')
C=S.groupby(['target','clf','k']).ba_ge30.mean().unstack('k')
pd.set_option('display.width',200)
print(C.round(2))
C.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/srs_curve_raw.csv')
# unit-level curves too
U=S.groupby(['target','clf','split','rep','k']).ba_ge30.mean().unstack('k')
U.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/srs_curve_units.csv')
