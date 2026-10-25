import pandas as pd, numpy as np
D=pd.read_pickle('G_all.pkl'); key=['target','split','rep','clf','k']
B=D[(D.cond=='none')&(D.k.isin([50,200]))].set_index(key)[['ba_ge30','oa']]
M=D[(D.tier=='t1')&D.cond.isin(['AE','S2'])&D.method.isin(['lpm2_nogeo','kcenter','typiclust','kherding'])]
g=M.groupby(key+['cond','method'])[['ba_ge30','oa']].mean().reset_index().set_index(key).join(B,rsuffix='_s')
g['dba']=g.ba_ge30-g.ba_ge30_s; g['doa']=g.oa-g.oa_s
c=g.reset_index().groupby(['k','target'])[['dba','doa']].mean().groupby('k').mean()
print('几何 4 法 × 2 指导 平均（相对 SRS）', c.round(2).to_dict())
H=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/headroom_summary.csv')
print(H.groupby(['k','metric'])[['U2_gain','oracle_equal_gain','oracle_sqrt_gain','random_sd']].agg(['mean','min','max']).round(2).to_string())
