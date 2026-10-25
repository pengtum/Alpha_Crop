import glob, os, pandas as pd, numpy as np
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
fs=sorted(glob.glob(os.path.join(OUT,'rows_t*_*.csv')))
L=[]
for f in fs:
    d=pd.read_csv(f); d['tier']=os.path.basename(f).split('_')[1]; d['file']=os.path.basename(f); L.append(d)
R=pd.concat(L,ignore_index=True)
print(len(R)); print(R.groupby('tier').size())
print(R.groupby(['tier','cond']).size())
print(R.groupby(['tier','cond']).apply(lambda g: g.groupby(['target','split','rep']).ngroups))
print(R[R.tier.isin(['t3','t4'])].groupby(['cond','method','k','clf']).size())
KEY=['target','split','rep','cond','method','clf','k']
print('dups', R.duplicated(KEY).sum(), 'dups t12', R[R.tier.isin(['t1','t2'])].duplicated(KEY).sum())
print('nan ba_ge30', R.ba_ge30.isna().sum())
print(R[R.tier.isin(['t1','t2'])].groupby('cond').size())
print('t1 units', R[R.tier=='t1'].groupby(['target','split','rep']).ngroups, 't2 units', R[R.tier=='t2'].groupby(['target','split','rep']).ngroups)
print(sorted(R.method.unique()))
print(R.groupby(['cond'])['method'].unique())
