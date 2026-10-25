from common_mv import *
import pandas as pd
pd.set_option('display.width', 250)
df = pd.read_csv('d4_sens_rows.csv')
G = pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/headroom_map/G_all.pkl')
G = G[G.tier == 't1'].rename(columns={'ba_ge30': 'baG', 'oa': 'oaG'})
U = ['target', 'split', 'rep']
g0 = G[(G.method == 'foreign_only') & (G.clf == 'ae')].set_index(U)[['baG', 'oaG']]
gs = G[(G.cond == 'none') & (G.method == 'srs') & (G.clf == 'ae')].set_index(U + ['k'])[['baG', 'oaG']]
print('=== α=0.5，正则敏感性：两视图/AE 校准层 − G 重训(AE)，同一批 SRS 标签；11 国均值（正向国家数）===')
tab = []
for (k, gw, gb, v), d in df[df.k < 1000].groupby(['k', 'gw', 'gb', 'views']):
    j = d.set_index(U).join(gs.xs(k, level='k'))
    dB = (j.ba - j.baG).groupby(level=0).mean(); dO = (j.oa - j.oaG).groupby(level=0).mean()
    tab.append(dict(k=k, gw=gw, gb=gb, views=v, dBA=round(dB.mean(), 2), posBA=int((dB > 0).sum()),
                    dOA=round(dO.mean(), 2), posOA=int((dO > 0).sum())))
print(pd.DataFrame(tab).to_string(index=False))
o = df[df.k == 30000].set_index(U).join(g0)
print('\n神谕 α=0.5 两视图校准层（全池 3 万标签）相对 k0 AE：dBA %.2f dOA %.2f' % ((o.ba - o.baG).mean(), (o.oa - o.oaG).mean()))
# 逐国：两视图 − AE（gw=5, gb=1, k=200）
d = df[(df.gw == 5) & (df.gb == 1) & (df.k == 200)].pivot_table(index=U, columns='views', values=['ba', 'oa'])
print('\n逐国两视图 − AE 单视图（k=200, α=0.5）ΔBA:', (d[('ba', 'both')] - d[('ba', 'ae')]).groupby(level=0).mean().round(2).to_dict())
print('ΔOA:', (d[('oa', 'both')] - d[('oa', 'ae')]).groupby(level=0).mean().round(2).to_dict())
W = pd.read_csv('d4_weights.csv').groupby('target').mean(numeric_only=True)
sh = pd.DataFrame({c: W[f'ws_{c}'] / (W[f'wa_{c}'] + W[f'ws_{c}']) for c in CLASSES})
print('\n学到的 S2 视图份额 ws/(wa+ws)（k=200，4 单元平均；0.5=等权）'); print(sh.round(2).to_string())
