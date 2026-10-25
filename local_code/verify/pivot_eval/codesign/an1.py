import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 40)
R = pd.read_csv('diag1_srs_rows.csv')
k0 = R[R['head'] == 'k0'][['unit', 'feat', 'ba', 'oa', 'ba_G_k0']].rename(columns={'ba': 'ba0', 'oa': 'oa0'})
print('k0 与 G 行一致性：max|ba0-ba_G_k0| =', (k0.ba0 - k0.ba_G_k0).abs().max())
D = R[R['head'] != 'k0'].merge(k0[['unit', 'feat', 'ba0', 'oa0']], on=['unit', 'feat'])
D['dba'] = D.ba - D.ba0; D['doa'] = D.oa - D.oa0
D['xRP'] = ~D.tgt.isin(['RO', 'PT'])
# 单元内先平均 4 个（划分×重复），再跨国平均
def summ(df):
    c = df.groupby(['head', 'feat', 'k', 'tgt'])[['dba', 'doa']].mean().reset_index()
    a = c.groupby(['head', 'feat', 'k'])[['dba', 'doa']].mean()
    b = c[~c.tgt.isin(['RO', 'PT'])].groupby(['head', 'feat', 'k'])['dba'].mean().rename('dba_xRP')
    npos = c.assign(p=c.dba > 0).groupby(['head', 'feat', 'k'])['p'].sum().rename('n_pos')
    return pd.concat([a, b, npos], axis=1)
S = summ(D).round(2)
print(S.unstack('k').to_string())
