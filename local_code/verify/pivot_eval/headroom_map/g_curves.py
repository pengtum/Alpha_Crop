import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
D = pd.read_pickle('G_all.pkl')
B = D[(D.cond=='none')].copy()
unit = ['target','split','rep','clf']
# SRS curve: mean over 4 units per country x clf
cols=['ba_ge30','oa','ba_P1','oa_P1','ba_P2','oa_P2','ba_i200','ba_i250','ba_i300','oa_i200','oa_i300','ba_far','comp_L1','n_classes_sel']
C = B.groupby(['clf','target','k'])[cols].mean()
for clf in ['s2','ae']:
    print('=== clf',clf,'ba_ge30 by k'); t=C.loc[clf]['ba_ge30'].unstack('k'); t['d200']=t[200]-t[0]; t['d400']=t[400]-t[0]; t.loc['MEAN']=t.mean(); t.loc['MEAN_noRO_PT']=t.drop(['RO','PT','MEAN']).mean(); print(t.round(2))
    print('=== clf',clf,'oa by k'); t=C.loc[clf]['oa'].unstack('k'); t['d200']=t[200]-t[0]; t['d400']=t[400]-t[0]; t.loc['MEAN']=t.mean(); t.loc['MEAN_noRO_PT']=t.drop(['RO','PT','MEAN']).mean(); print(t.round(2))
# unit-level SD of srs gains for per-country significance
g = B.pivot_table(index=unit, columns='k', values='ba_ge30')
g['d200']=g[200]-g[0]; g['d400']=g[400]-g[0]
s = g.groupby(['clf','target'])[['d200','d400']].agg(['mean','std'])
print(s.round(2))
