# D7：与更强的重训基线比较（同一批标签）：G 重训 300 轮、200 轮（ba_i200）；以及 predstrat_srs 标签下的比较。
from common_mv import *
import pandas as pd
pd.set_option('display.width', 250)
df = pd.read_csv('d3_paired_rows.csv')
G = pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/headroom_map/G_all.pkl')
G = G[G.tier == 't1']
U = ['target', 'split', 'rep']
def ttest(v):
    v = np.asarray(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    return m, m - 2.228 * se, m + 2.228 * se, int((v > 0).sum())
def fmt(v):
    m, lo, hi, n = ttest(v); return f'{m:+.2f} [{lo:+.2f},{hi:+.2f}] {n}/11'
out = []
for (lab_cond, lab_m) in (('none', 'srs'), ('AE', 'predstrat_srs'), ('S2', 'predstrat_srs')):
    for k in (50, 200):
        c = df[(df.cond == lab_cond) & (df.method == lab_m) & (df.k == k) & (df.alpha == 0.5)]
        for views in ('both', 'ae'):
            cc = c[c.views == views].set_index(U)[['ba', 'oa']]
            for clf in ('ae', 's2'):
                for it in ('300', '200'):
                    g = G[(G.cond == lab_cond) & (G.method == lab_m) & (G.k == k) & (G.clf == clf)].set_index(U)
                    bcol, ocol = ('ba_ge30', 'oa') if it == '300' else ('ba_i200', 'oa_i200')
                    j = cc.join(g[[bcol, ocol]].rename(columns={bcol: 'gB', ocol: 'gO'}), how='inner')
                    if len(j) == 0: continue
                    dB = (j.ba - j.gB).groupby(level=0).mean(); dO = (j.oa - j.gO).groupby(level=0).mean()
                    out.append(dict(labels=f'{lab_cond}_{lab_m}', k=k, cal=views, retrain=f'{clf}@{it}', n=len(j),
                                    dBA=fmt(dB), dOA=fmt(dO)))
o = pd.DataFrame(out)
print(o.to_string(index=False))
# 全流程：predstrat_srs(AE) 标签 + 两视图校准层  vs  G 最好的重训组合（predstrat_srs(AE) 重训）与 SRS 重训
print('\n=== 全流程（只有 rep0 的 22 个单元）：校准层[predstrat_srs(AE) 标签] − 重训[SRS 标签]（AE 分类器 300 轮）===')
for k in (50, 200):
    c = df[(df.cond == 'AE') & (df.method == 'predstrat_srs') & (df.k == k) & (df.alpha == 0.5) & (df.views == 'both')].set_index(U)
    g = G[(G.cond == 'none') & (G.method == 'srs') & (G.k == k) & (G.clf == 'ae')].set_index(U)
    j = c[['ba', 'oa']].join(g[['ba_ge30', 'oa']].rename(columns={'ba_ge30': 'gB', 'oa': 'gO'}), how='inner')
    print(k, 'dBA', fmt((j.ba - j.gB).groupby(level=0).mean()), 'dOA', fmt((j.oa - j.gO).groupby(level=0).mean()))
