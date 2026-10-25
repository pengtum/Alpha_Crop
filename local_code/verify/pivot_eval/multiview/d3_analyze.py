from common_mv import *
import pandas as pd
pd.set_option('display.width', 250)
df = pd.read_csv('d3_paired_rows.csv')
G = pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/headroom_map/G_all.pkl')
G = G[G.tier == 't1'].rename(columns={'ba_ge30': 'baG', 'oa': 'oaG'})
U = ['target', 'split', 'rep']
k0 = G[G.method == 'foreign_only'].set_index(U + ['clf'])[['baG', 'oaG']]
def ttest(v):
    v = np.asarray(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    return m, m - 2.228 * se, m + 2.228 * se, int((v > 0).sum())
def fmt(v):
    m, lo, hi, n = ttest(v); return f'{m:+.2f} [{lo:+.2f},{hi:+.2f}] {n}/11'
srsG = G[(G.cond == 'none') & (G.method == 'srs')].set_index(U + ['k', 'clf'])[['baG', 'oaG']]
C = df[(df.cond == 'none') & (df.method == 'srs')].set_index(U + ['k'])
print('=== 同一批 SRS 标签：校准层（冻结 k0 两视图）− G 重训 LightGBM（r=50），11 国均值 [95%CI] 正向国家数 ===')
for k in (50, 100, 200, 400):
    for a in (0.0, 0.5, 1.0):
        for views in ('both', 'ae'):
            c = C[(C.alpha == a) & (C.views == views)].xs(k, level='k')
            out = []
            for clf in ('ae', 's2'):
                j = c.join(srsG.xs(k, level='k').xs(clf, level='clf'), how='inner')
                out.append(f'vs重训{clf}: dBA {fmt((j.ba - j.baG).groupby(level=0).mean())}; dOA {fmt((j.oa - j.oaG).groupby(level=0).mean())}')
            print(f'k={k} α={a} {views:4s} | ' + ' | '.join(out))
print('\n=== 相对 k0（各自对照：校准层对 AE 外国模型 k0；重训对各自分类器 k0）的 ΔBA / ΔOA，国家均值（SRS 标签）===')
res = []
k0ae = k0.xs('ae', level='clf')
for k in (50, 100, 200, 400):
    for a in (0.0, 0.5, 1.0):
        for views in ('both', 'ae', 's2'):
            c = C[(C.alpha == a) & (C.views == views)].xs(k, level='k').join(k0ae)
            dB = (c.ba - c.baG).groupby(level=0).mean(); dO = (c.oa - c.oaG).groupby(level=0).mean()
            res.append(dict(k=k, arm=f'cal_{views}_a{a}', dBA=dB.mean(), dOA=dO.mean(),
                            dBA_noROPT=dB.drop(['RO', 'PT']).mean(), dOA_noROPT=dO.drop(['RO', 'PT']).mean()))
    for clf in ('ae', 's2'):
        g = srsG.xs(k, level='k').xs(clf, level='clf').join(k0.xs(clf, level='clf'), rsuffix='0')
        dB = (g.baG - g.baG0).groupby(level=0).mean(); dO = (g.oaG - g.oaG0).groupby(level=0).mean()
        res.append(dict(k=k, arm=f'G_retrain_{clf}', dBA=dB.mean(), dOA=dO.mean(), dBA_noROPT=dB.drop(['RO', 'PT']).mean(),
                        dOA_noROPT=dO.drop(['RO', 'PT']).mean()))
R_ = pd.DataFrame(res)
R_.to_csv('d3_frontier.csv', index=False)
print(R_.pivot(index='arm', columns='k', values=['dBA', 'dOA']).round(2).to_string())
print(R_[R_.k == 200].set_index('arm')[['dBA_noROPT', 'dOA_noROPT']].round(2).to_string())
print('\n=== 两视图 − 单视图 AE（同一 α、同一批 SRS 标签）===')
for k in (50, 100, 200, 400):
    for a in (0.0, 0.5, 1.0):
        p = C[C.alpha == a].xs(k, level='k').reset_index().pivot_table(index=U, columns='views', values=['ba', 'oa'])
        d = (p[('ba', 'both')] - p[('ba', 'ae')]).groupby(level=0).mean()
        do = (p[('oa', 'both')] - p[('oa', 'ae')]).groupby(level=0).mean()
        print(f'k={k} α={a}: dBA {fmt(d)}; dOA {fmt(do)}')
