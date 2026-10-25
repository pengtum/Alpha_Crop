import sys; sys.path.insert(0,'.')
from load import ci, fmt, load
import numpy as np, pandas as pd
from scipy import stats
pd.set_option('display.width',250)
A=pd.read_pickle('A.pkl'); A=A[A.cond.isin(['AE','S2'])]
X=pd.read_pickle('X.pkl')
# SRS 标签价值：SRS(k) − foreign_only，逐国（两分类器、两 k 平均）
fo=X[X.method=='foreign_only'].groupby(['target','clf'])[['ba_ge30','oa']].mean()
sr=X[(X.method=='srs')&X.k.isin([50,200])].groupby(['target','clf'])[['ba_ge30','oa','oa_P2','ba_P2']].mean()
val=(sr[['ba_ge30','oa']]-fo).groupby('target').mean().rename(columns={'ba_ge30':'val_ba','oa':'val_oa'})
# P2 对 SRS 的作用（先验修正本身的价值）
fo2=X[X.method=='foreign_only'].groupby(['target','clf'])[['oa_P2','ba_P2','oa']].mean()
val['P2gain_fo_oa']=(fo2.oa_P2-fo2.oa).groupby('target').mean()
M=A.groupby(['method','target'])[['d_ba_ge30','d_oa','d_oa_P2']].mean().unstack('method')
T=pd.concat([val.round(2),M['d_oa'][['kcenter','predstrat_srs','margin','gist_margin','typiclust']].round(2).add_prefix('dOA_'),
             M['d_ba_ge30'][['kcenter','predstrat_srs','margin','gist_margin','typiclust']].round(2).add_prefix('dBA_')],axis=1)
print(T.to_string())
print('\n跨国相关（Spearman，n=11）：')
for m in ['kcenter','predstrat_srs','margin','gist_margin','badge_c','typiclust','lpm2_nogeo','kherding']:
    for met in ['d_oa','d_ba_ge30']:
        v=M[met][m]
        r1=stats.spearmanr(val.val_oa if met=='d_oa' else val.val_ba, v)
        r2=stats.spearmanr(val.val_oa if met=='d_oa' else val.val_ba, v.abs())
        print(f'  {m:14s} {met:9s} vs SRS 标签价值: ρ={r1.statistic:+.2f} (p={r1.pvalue:.3f}); |Δ| ρ={r2.statistic:+.2f} (p={r2.pvalue:.3f})')
# 各方法去掉 RO、PT 后
print('\n去掉 RO、PT 后（9 国）的方法 − SRS：')
for m in ['kcenter','predstrat_srs','margin','gist_margin','badge_c','typiclust','lpm2_nogeo','kherding']:
    for cond in ('AE','S2'):
        a=A[(A.method==m)&(A.cond==cond)&~A.target.isin(['RO','PT'])]
        v=a.groupby(['clf','k','target']).d_ba_ge30.mean().groupby('target').mean()
        o=a.groupby(['clf','k','target']).d_oa.mean().groupby('target').mean()
        print(f'  {m:14s} {cond} BA {fmt(ci(v))}   OA {fmt(ci(o))}')
