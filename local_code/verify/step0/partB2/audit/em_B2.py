import os, sys
os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'; os.environ['VECLIB_MAXIMUM_THREADS']='2'
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
from scipy import stats
M = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/fa_rows_B2.csv', float_precision='round_trip')
E = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/envelope_B2.csv', float_precision='round_trip')
DROP=['RO','PT']
# 共同可达口径：Z0EM vs Z0（EM 先验 = 零标签的 p̂，作为 PK 构造的"安慰剂"对照）
Ew = E.set_index(['target','unit','clf','off','arm'])['ba'].unstack('arm')
for a in ('Z0EM','PK50','PK200','PKo'):
    m = Ew[a].notna() & Ew['Z0'].notna()
    d = (Ew[a]-Ew['Z0']).where(m).groupby(['target','unit','clf']).mean()
    cov = m.groupby(['target','unit','clf']).mean()
    d[cov<0.5] = np.nan
    D = d.reset_index(name='d')
    ae = D[D.clf=='ae'].groupby('target').d.mean(); s2 = D[D.clf=='s2'].groupby('target').d.mean()
    both = pd.concat([ae,s2],axis=1).mean(axis=1)  # 允许单特征
    v = both.drop(DROP).dropna(); n=len(v); mm=v.mean(); sd=v.std(ddof=1); q=stats.t.ppf(.975,n-1)
    print(f'{a:6s} − Z0（共同可达，9 国 both）{mm:+.2f} [{mm-q*sd/np.sqrt(n):+.2f},{mm+q*sd/np.sqrt(n):+.2f}] >0 {(v>0).sum()}/{n}；单元×特征可用 {d.notna().sum()}/88')
