"""质疑者视角 s3：检验 2 中 uncertainty 的优势对 SE、分类器的依赖（pilot_selection.csv，无缓冲划分）。"""
import pandas as pd, numpy as np
from scipy import stats
P=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/pilot_selection.csv')
P['ba_ge30']*=100
r=P[P.strategy=='random'].set_index(['target','budget','seed','clf_feat']).ba_ge30
x=P[~P.strategy.isin(['random','foreign_only','target_full'])].copy()
x['d']=x.ba_ge30.to_numpy()-r.reindex(pd.MultiIndex.from_frame(x[['target','budget','seed','clf_feat']])).to_numpy()
x=x.dropna(subset=['d'])
print(sorted(x.strategy.unique()))
T=x.groupby(['strategy','budget','target','clf_feat']).d.mean().unstack(['clf_feat'])
for s in ['uncertainty']:
    for b in (50,200,1000):
        g=T.loc[(s,b)]
        print(s,b,'各国×分类器：'); print(g.round(2).to_string())
        m=g.mean(1); print(f'  4 国均值 {m.mean():+.2f}；去 SE {m.drop("SE").mean():+.2f}；只 s2 分类器 {g["s2"].mean():+.2f}；只 ae 分类器 {g["ae"].mean():+.2f}；去 SE 只 s2 {g["s2"].drop("SE").mean():+.2f}')
# 所有策略：去 SE 后 4→3 国均值
M=x.groupby(['strategy','budget','target']).d.mean().unstack('target')
M['mean4']=M[['LV','PT','SE','SK']].mean(1); M['exSE']=M[['LV','PT','SK']].mean(1)
print(M.round(2).to_string())
