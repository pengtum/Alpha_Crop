from common_mv import *
import pandas as pd
pd.set_option('display.width', 250)
df = pd.read_csv('d6_design_rows.csv')
def ttest(v):
    v = np.asarray(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    return m, m - 2.228 * se, m + 2.228 * se, int((v > 0).sum())
def fmt(v):
    m, lo, hi, n = ttest(v); return f'{m:+.2f} [{lo:+.2f},{hi:+.2f}] {n}/11'
U = ['target', 'split', 'k', 'seed']
base = df[(df.design == 'srs') & (df.views == 'both')].set_index(U)
base_ae = df[(df.design == 'srs') & (df.views == 'ae')].set_index(U)
print('=== 两视图校准层（α=0.5）下，设计 − SRS（每单元 20 种子平均；国家为单位）===')
for k in (50, 100, 200):
    for (d, f, v), g in df[(df.k == k) & (df.design != 'srs')].groupby(['design', 'fit', 'views']):
        j = g.set_index(U).join(base if v == 'both' else base_ae, rsuffix='_s', how='inner')
        dB = (j.ba - j.ba_s).groupby(level=0).mean(); dO = (j.oa - j.oa_s).groupby(level=0).mean()
        print(f'k={k} {d:13s} {f:5s} {v:4s} dBA {fmt(dB)} | dOA {fmt(dO)} | frac_dis {g.frac_dis.mean():.2f} n_cls {g.n_cls.mean():.1f}')
print('\nSRS 下 frac_dis', df[df.design == 'srs'].groupby('k').frac_dis.mean().round(3).to_dict())
# 两视图 − AE 单视图，在 srs 与 xvstrat_dis 上
for d in ('srs', 'predstrat_ae'):
    for k in (50, 200):
        g = df[(df.design == d) & (df.fit == 'plain') & (df.k == k)].pivot_table(index=U, columns='views', values=['ba', 'oa'])
        print(f'{d} k={k} 两视图−AE: dBA {fmt((g[("ba","both")]-g[("ba","ae")]).groupby(level=0).mean())} dOA {fmt((g[("oa","both")]-g[("oa","ae")]).groupby(level=0).mean())}')
# 种子间 SD
print('\n种子间 SD（ba）:', df[(df.views == 'both') & (df.fit == 'plain')].groupby(['design', 'k', 'target', 'split']).ba.std().groupby(['design', 'k']).mean().unstack().round(2).to_string())
# 逐国：xvstrat_dis ht 与 predstrat_ae plain 相对 SRS
for d, f in (('xvstrat_dis', 'ht'), ('xvstrat', 'ht'), ('predstrat_ae', 'plain'), ('oracle_sqrt', 'ht')):
    g = df[(df.design == d) & (df.fit == f) & (df.views == 'both') & (df.k == 200)].set_index(U).join(base, rsuffix='_s')
    print(d, f, 'k200 逐国 dBA', (g.ba - g.ba_s).groupby(level=0).mean().round(2).to_dict())
