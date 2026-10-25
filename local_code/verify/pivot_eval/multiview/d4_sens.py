# D4：两视图校准层对正则强度的敏感性（α=0.5，G 的 SRS 标签，k=50/200）；逐国两视图增益；学到的视图权重。
from common_mv import *
import pandas as pd, glob, stacker as st
pd.set_option('display.width', 250)
G = pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/headroom_map/G_all.pkl')
G = G[G.tier == 't1']
rows, W = [], []
for t in COUNTRIES:
    for s in (0, 1):
        z = np.load(f'{R}/results/guidance_G/stage0/{t}_s{s}.npz')
        nc = len(z['cand'])
        Pa, Ps = z['P_T_ae'].astype(np.float64), z['P_T_s2'].astype(np.float64)
        yT = Y[z['test']]; big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
        LaT, LsT = st._ll(Pa[nc:]), st._ll(Ps[nc:])
        for r in (0, 1):
            pp = pool_positions(nc, s, r)
            LaP, LsP, yP = st._ll(Pa[pp]), st._ll(Ps[pp]), Y[z['allT'][pp]]
            for k in (50, 200):
                sp = np.load(f'{R}/results/guidance_G/sel/{t}_s{s}_r{r}_none_srs_k{k}.npz')['sel_pos']
                for gw in (1.0, 5.0, 20.0):
                    for gb in (0.1, 1.0, 10.0):
                        for views in ('both', 'ae'):
                            th = st.fit(LaP[sp], LsP[sp], yP[sp], mode=0.5, gw=gw, gb=gb, views=views)
                            oa, ba = st.scores(yT, st.predict(th, LaT, LsT), big)
                            rows.append(dict(target=t, split=s, rep=r, k=k, gw=gw, gb=gb, views=views, oa=oa, ba=ba))
                            if gw == 5.0 and gb == 1.0 and views == 'both' and k == 200:
                                W.append(dict(target=t, split=s, rep=r, **{f'wa_{CLASSES[c]}': th[c] for c in range(10)},
                                              **{f'ws_{CLASSES[c]}': th[10 + c] for c in range(10)}))
        # 神谕 α=0.5（全池标签）
        pp = pool_positions(nc, s, 0)
        th = st.fit(st._ll(Pa[pp]), st._ll(Ps[pp]), Y[z['allT'][pp]], mode=0.5, gw=0.01, gb=0.01, views='both')
        oa, ba = st.scores(yT, st.predict(th, LaT, LsT), big)
        rows.append(dict(target=t, split=s, rep=0, k=30000, gw=0.01, gb=0.01, views='both', oa=oa, ba=ba))
df = pd.DataFrame(rows); df.to_csv('d4_sens_rows.csv', index=False)
pd.DataFrame(W).to_csv('d4_weights.csv', index=False)
U = ['target', 'split', 'rep']
g0 = G[G.method == 'foreign_only'].set_index(U + ['clf'])[['ba_ge30', 'oa']].xs('ae', level='clf')
gs = G[(G.cond == 'none') & (G.method == 'srs')].set_index(U + ['k', 'clf'])[['ba_ge30', 'oa']].xs('ae', level='clf')
print('=== α=0.5，正则敏感性：两视图校准层 − G 重训(AE)，同一批 SRS 标签；11 国均值（正向国家数）===')
for k in (50, 200):
    tab = []
    for (gw, gb, v), d in df[df.k == k].groupby(['gw', 'gb', 'views']):
        j = d.set_index(U).join(gs.xs(k, level='k'))
        dB = (j.ba - j.ba_ge30).groupby(level=0).mean(); dO = (j.oa - j.oa).groupby(level=0).mean()
        dO = (d.set_index(U).oa - gs.xs(k, level='k').oa).dropna().groupby(level=0).mean()
        tab.append(dict(k=k, gw=gw, gb=gb, views=v, dBA=round(dB.mean(), 2), posBA=int((dB > 0).sum()),
                        dOA=round(dO.mean(), 2), posOA=int((dO > 0).sum())))
    print(pd.DataFrame(tab).to_string(index=False))
o = df[df.k == 30000].set_index(['target', 'split']).join(g0.xs(0, level='rep'))
print('\n神谕 α=0.5 两视图校准层（全池 3 万标签）相对 k0 AE：dBA %.2f dOA %.2f' % ((o.ba - o.ba_ge30).mean(), (o.oa - o.oa_G if 'oa_G' in o else o.oa - g0.xs(0, level='rep').oa.reindex(o.index)).mean()))
