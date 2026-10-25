"""G 的 E1 × k=200 × s0r0：直接用 G 缓存的 LightGBM 测试概率（probs/*.npz 的 i300），做无标签先验调整 τ∈{0,0.5,1}
后重算 AE−S2（不重训）。检验核心对比是否对决策工作点敏感。"""
import sys; sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels')
from altlib import *
import pandas as pd
from scipy import stats
D = G.GData()
E = pd.read_csv(os.path.join(OUT, 'lgb_e1_k200_s0r0.csv'))
rows = []
for tgt in D.countries:
    S, fr, test, yT, big, cw = load_unit(D, tgt)
    for _, r in E[E.target == tgt].iterrows():
        P = np.load(f'{GOUT}/probs/{tgt}_s0_r0_{r.clf}_k200_{r.sel_hash}.npz')['i300'].astype(np.float64)
        Lg = np.log(np.clip(P, 1e-12, None)); pi = np.clip(P.mean(0), 1e-6, None)
        for t in (0.0, 0.5, 1.0):
            oa, ba = scores(yT, (Lg - t * np.log(pi)).argmax(1), big)
            if t == 0.0:
                assert abs(ba - r.ba_ge30) < 1e-6, (tgt, r.method, ba, r.ba_ge30)
            rows.append(dict(target=tgt, cond=r.cond, method=r.method, clf=r.clf, tau=t, ba=ba, oa=oa))
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'e1_tau_rows.csv'), index=False)
def tci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); h = stats.t.ppf(.975, n-1)*v.std(ddof=1)/np.sqrt(n)
    return f'{m:+.3f} [{m-h:+.3f}, {m+h:+.3f}] p={stats.ttest_1samp(v,0).pvalue:.2f} {int((v>0).sum())}/{n}'
for t in (0.0, 0.5, 1.0):
    x = R[R.tau == t].pivot_table(index=['target', 'method', 'clf'], columns='cond', values=['ba', 'oa'])
    d = (x.xs('AE', axis=1, level=1) - x.xs('S2', axis=1, level=1)).reset_index().groupby('target')[['ba', 'oa']].mean()
    print(f'LGB τ={t}: AE−S2 ba {tci(d.ba)} | oa {tci(d.oa)} | 平均 ba {R[R.tau==t].ba.mean():.2f} oa {R[R.tau==t].oa.mean():.2f}')
