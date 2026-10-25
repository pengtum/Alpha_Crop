import sys; sys.path.insert(0,'.')
from load import ci, fmt
import numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
A=pd.read_pickle('A.pkl')
A=A[A.cond.isin(['AE','S2'])]
med=['d_comp_L1','d_LD','d_LD_oa','d_maj_share','d_nonbig_share','d_nbig_sel','d_sc5','d_any5','d_spat','d_far_own','d_far_common']
out=['d_ba_ge30','d_ba_P2','d_oa','d_oa_P2']
T=A.groupby(['method','cond','k'])[med+out].mean()
print(T.round(3).to_string())
T.to_csv('a2_method_mediators.csv')
# 相关：方法×条件×k 层（32 个点）
print('\n方法层（32 点）相关（Pearson）:')
print(T[med].corrwith(T.d_ba_ge30).round(2).to_string())
print('with d_oa'); print(T[med].corrwith(T.d_oa).round(2).to_string())
