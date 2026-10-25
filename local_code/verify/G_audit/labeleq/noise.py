import numpy as np, pandas as pd
pd.set_option('display.width',220)
S=pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/S.pkl')
S=S[S.method=='srs']
g=S.groupby(['target','clf','k']).ba_ge30
sd=g.std(ddof=1).unstack('k')
print('单元间 SD（4 单元，k=50..400）'); print(sd.round(2))
print('每点 SD 的中位数', np.nanmedian(sd.values).round(3), ' → 4 单元均值 SE ≈', (np.nanmedian(sd.values)/2).round(3))
# 去掉划分效应：单元内减去划分均值？ 测试集随划分变化，但同一划分内 rep 只改变候选池/种子
S2=S.copy()
S2['res']=S2.ba_ge30-S2.groupby(['target','clf','k','split']).ba_ge30.transform('mean')
w=S2.groupby(['target','clf','k','split']).res.apply(lambda x: (x**2).sum()).groupby(['target','clf','k']).sum()/2  # df=2 (2 splits × (2-1))
print('\n同划分内（只因重复/种子不同）的 SD 中位数：', np.sqrt(w).median().round(3))
print(np.sqrt(w).unstack('k').round(2))
# gain 50->400 单元层面
G=S.pivot_table(index=['target','clf','split','rep'],columns='k',values='ba_ge30')
G['g50_400']=G[400]-G[50]; G['g50_200']=G[200]-G[50]; G['g200_400']=G[400]-G[200]
q=G.groupby(['target','clf'])[['g50_200','g200_400','g50_400']].agg(['mean','std'])
q[('g50_400','t')]=q[('g50_400','mean')]/(q[('g50_400','std')]/2)
print(q.round(2))
