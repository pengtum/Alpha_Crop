# D3b：换成"两视图校准层"这种标签用法后，选样方法相对 SRS 的差（同单元、同 k）；并与 G 重训下的同一差比较。
from common_mv import *
import pandas as pd
pd.set_option('display.width', 250)
df = pd.read_csv('d3_paired_rows.csv')
G = pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/headroom_map/G_all.pkl')
G = G[G.tier.isin(['t1', 't2', 't4'])]
U = ['target', 'split', 'rep', 'k']
def ttest(v):
    v = np.asarray(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    return m, m - 2.228 * se, m + 2.228 * se, int((v > 0).sum())
out = []
for a in (0.5, 1.0):
    c = df[(df.alpha == a) & (df.views == 'both')]
    srs = c[(c.cond == 'none') & (c.method == 'srs')].set_index(U)[['ba', 'oa']]
    for (cond, m), g in c[c.cond != 'none'].groupby(['cond', 'method']):
        for k in (50, 200):
            gg = g[g.k == k].set_index(U)[['ba', 'oa', 'n_cls']]
            if len(gg) == 0: continue
            j = gg.join(srs, rsuffix='_s', how='inner')
            d = (j.ba - j.ba_s).groupby(level=0).mean(); do = (j.oa - j.oa_s).groupby(level=0).mean()
            if len(d) < 11: continue
            mb, lo, hi, n = ttest(d); mo, loo, hio, no = ttest(do)
            # G 重训下同一对比（AE 分类器与 S2 分类器平均）
            gr = G[(G.cond == cond) & (G.method == m) & (G.k == k)].groupby(U + ['clf'])[['ba_ge30', 'oa']].mean()
            gs = G[(G.cond == 'none') & (G.method == 'srs') & (G.k == k)].groupby(U + ['clf'])[['ba_ge30', 'oa']].mean()
            jr = gr.join(gs, rsuffix='_s', how='inner')
            dr = (jr.ba_ge30 - jr.ba_ge30_s).groupby(level=0).mean().mean()
            out.append(dict(alpha=a, cond=cond, method=m, k=k, n_units=len(j), dBA_cal=mb, lo=lo, hi=hi, npos=n,
                            dOA_cal=mo, dBA_retrain=dr))
o = pd.DataFrame(out)
o.to_csv('d3b_acq_vs_srs.csv', index=False)
for a in (0.5, 1.0):
    print(f'\n=== α={a}，两视图校准层下，方法 − SRS（pp）===')
    print(o[o.alpha == a].sort_values(['k', 'dBA_cal'], ascending=[True, False]).drop(columns='alpha').round(2).to_string(index=False))
