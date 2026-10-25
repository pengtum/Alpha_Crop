"""固定 τ（无标签规则）下的 LightGBM 先验调整 vs 替代模型原生输出。τ=0 原生；τ=0.5、1（除以测试集平均预测概率）。"""
import numpy as np, pandas as pd
from scipy import stats
Z = np.load('/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels/frontier_curves.npz')
TAUS = np.round(np.arange(-3, 3.001, 0.05), 2)
rows = []
for key in Z.files:
    tgt, feat, k, model = key.split('|')
    C = Z[key]
    for t in (0.0, 0.5, 1.0):
        j = int(np.argmin(np.abs(TAUS - t)))
        rows.append(dict(target=tgt, feat=feat, k=int(k), model=model, tau=t, oa=C[j, 1], ba=C[j, 2]))
R = pd.DataFrame(rows)
R.to_csv('/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels/frontier_fixed_tau.csv', index=False)
def tci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); h = stats.t.ppf(.975, n-1)*v.std(ddof=1)/np.sqrt(n)
    return f'{m:+.2f} [{m-h:+.2f}, {m+h:+.2f}] {int((v>0).sum())}/{n}'
for k in (0, 200):
    g = R[R.k == k].groupby(['model', 'tau', 'target'])[['ba', 'oa']].mean()
    print(f'=== k={k} 均值（11 国，ae/s2 平均）===')
    print(g.groupby(level=[0, 1]).mean().round(2).to_string())
    base = g.loc[('lgb', 0.0)]
    for m, t in (('lgb', 0.5), ('lgb', 1.0), ('lr', 0.0), ('mlp2', 0.0), ('lr', 1.0), ('mlp2', 1.0)):
        x = g.loc[(m, t)]
        print(f'  {m}@τ={t} − lgb@τ=0: Δba {tci(x.ba - base.ba)}  ΔOA {tci(x.oa - base.oa)}')
    for m in ('lr', 'mlp2'):
        x = g.loc[(m, 0.0)]; y = g.loc[('lgb', 1.0)]
        print(f'  {m}@τ=0 − lgb@τ=1: Δba {tci(x.ba - y.ba)}  ΔOA {tci(x.oa - y.oa)}')
        y = g.loc[('lgb', 0.5)]
        print(f'  {m}@τ=0 − lgb@τ=0.5: Δba {tci(x.ba - y.ba)}  ΔOA {tci(x.oa - y.oa)}')
