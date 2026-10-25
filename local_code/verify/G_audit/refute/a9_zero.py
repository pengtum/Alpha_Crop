import numpy as np, pandas as pd
R=pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/G_audit/refute/R.pkl')
E1=['lpm2_nogeo','kcenter','typiclust','kherding']; KEY=['target','split','rep','method','clf','k']
A=R[(R.cond=='AE')&R.method.isin(E1)].set_index(KEY); S=R[(R.cond=='S2')&R.method.isin(E1)].set_index(KEY)
j=A[['sel_hash','ba_ge30']].join(S[['sel_hash','ba_ge30']],lsuffix='_ae',rsuffix='_s2')
print('same sel_hash AE vs S2:', (j.sel_hash_ae==j.sel_hash_s2).sum(), ' exact zero Δ:', (j.ba_ge30_ae==j.ba_ge30_s2).sum(), ' n', len(j))
# duplicate rows?
print('dup keys AE', A.index.duplicated().sum(), 'S2', S.index.duplicated().sum())
print(R[R.cond.isin(['AE','S2'])].groupby(['cond','method','k']).size().unstack())
