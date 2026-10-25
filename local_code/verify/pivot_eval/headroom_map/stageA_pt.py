import pandas as pd, numpy as np
pd.set_option('display.width',250)
A=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/stageA/stageA_merged_interim_PT.csv')
print(A.shape, A.target.unique(), sorted(A.k.unique()), A.r.unique())
print(A.groupby(['split','rep']).size())
m=A.groupby(['method','k'])[['ba_ge30','oa','ba_P1','oa_P1','ba_P2','oa_P2','comp_L1','n_classes_sel']].mean()
print(m.loc[['srs','oracle_prop','oracle_sqrt','oracle_equal','oracle_err','bait','gist_maxmin','kdpp','predstrat','predstrat_equal','margin']].round(2).to_string())
# per class recall srs vs oracle_equal k=1000
r=A[A.k==1000].groupby('method')[[f'rec{i}' for i in range(10)]].mean()*100
print(r.loc[['srs','oracle_sqrt','oracle_equal','bait']].round(1).to_string())
