"""只读 G 缓存概率：树数 200→300 的 ba 下降是否只是工作点漂移？
对 SRS k∈{50,200} 的 probs（i200/i300）与阶段 0 的 k=0（P0test i200/i300），4 个单元，比较：原生 ba/OA、τ=0.5 下 ba/OA、
以及"i300 在 i200 原生 OA 处的 Pareto 包络 ba"（用测试标签描包络，诊断用）。"""
from vlib import *
from scipy import stats
D = G.GData()
rows = []
for split in (0, 1):
    for tgt in D.countries:
        S, fr, test, yT, big, cw = unit(D, tgt, split)
        for rep in (0, 1):
            R = pd.read_csv(f'{GOUT}/rows_t1_{tgt}_s{split}_r{rep}.csv')
            for feat in ('ae', 's2'):
                items = []
                if rep == 0:
                    items.append((0, {it: S[f'P0test_{feat}_i{it}'].astype(np.float32) for it in (200, 300)}))
                for k in (50, 200):
                    h = R[(R.method == 'srs') & (R.k == k) & (R.clf == feat)].sel_hash.iloc[0]
                    z = np.load(f'{GOUT}/probs/{tgt}_s{split}_r{rep}_{feat}_k{k}_{h}.npz')
                    items.append((k, {it: z[f'i{it}'] for it in (200, 300)}))
                for k, Pd in items:
                    C2, C3 = curve(Pd[200], yT, big), curve(Pd[300], yT, big)
                    n0 = int(np.flatnonzero(TAUS == 0)[0])
                    o2, b2 = at_tau(C2, 0.5); o3, b3 = at_tau(C3, 0.5)
                    rows.append(dict(target=tgt, split=split, rep=rep, feat=feat, k=k,
                                     ba200=C2[n0, 2], ba300=C3[n0, 2], oa200=C2[n0, 1], oa300=C3[n0, 1],
                                     ba200_t05=b2, ba300_t05=b3, oa200_t05=o2, oa300_t05=o3,
                                     env300_at_oa200=env(C3, C2[n0, 1]), env200_at_oa300=env(C2, C3[n0, 1]),
                                     max200=C2[:, 2].max(), max300=C3[:, 2].max()))
        print(tgt, split, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'trees_wp_rows.csv'), index=False)
def tci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v); m = v.mean(); h = stats.t.ppf(.975, n-1) * v.std(ddof=1) / np.sqrt(n)
    return f'{m:+.2f} [{m-h:+.2f}, {m+h:+.2f}] {int((v>0).sum())}/{n} p={stats.ttest_1samp(v,0).pvalue:.3f}'
for k in (0, 50, 200):
    g = R[R.k == k].groupby('target').mean(numeric_only=True)
    print(f'=== k={k}（国家层面，单元×特征平均）===')
    print('  原生 ba300−ba200', tci(g.ba300 - g.ba200), '| OA', tci(g.oa300 - g.oa200))
    print('  τ=0.5 ba300−ba200', tci(g.ba300_t05 - g.ba200_t05), '| OA', tci(g.oa300_t05 - g.oa200_t05))
    print('  包络：i300@i200 原生 OA − i200 原生 ba', tci(g.env300_at_oa200 - g.ba200))
    print('  包络：i300 原生 ba − i200@i300 原生 OA', tci(g.ba300 - g.env200_at_oa300))
    print('  τ 网格最大 ba：300−200', tci(g.max300 - g.max200))
