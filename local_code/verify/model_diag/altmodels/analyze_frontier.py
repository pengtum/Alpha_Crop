import numpy as np, pandas as pd
from scipy import stats
F = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels/frontier_rows.csv')
def tci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); h = stats.t.ppf(.975, n-1)*v.std(ddof=1)/np.sqrt(n)
    return f'{m:+.2f} [{m-h:+.2f}, {m+h:+.2f}] {int((v>0).sum())}/{n}'
for model in ('lr', 'mlp2'):
    for k in (0, 200):
        x = F[(F.model == model) & (F.k == k)].groupby('target').mean(numeric_only=True)
        print(f'--- {model} k={k} (国家层面，ae/s2 平均) ---')
        print('  原生：ba(m)−ba(lgb)', tci(x.ba_m - x.ba_lgb), '| OA(m)−OA(lgb)', tci(x.oa_m - x.oa_lgb))
        print('  LGB 调到 m 的 OA：ba(m)−ba(lgb@mOA)', tci(x.ba_m - x.lgb_ba_at_mOA), ' OA 差残余', tci(x.oa_m - x.lgb_oa_at_mOA),
              ' 平均 τ', round(x.lgb_tau_at_mOA.mean(), 2))
        print('  m 调到 LGB 的 OA：ba(m@lgbOA)−ba(lgb)', tci(x.m_ba_at_lgbOA - x.ba_lgb), ' OA 差残余', tci(x.m_oa_at_lgbOA - x.oa_lgb))
        print('  τ 网格上的最大 ba（用标签选 τ，上界）：m−lgb', tci(x.m_ba_max - x.lgb_ba_max), f' 均值 m {x.m_ba_max.mean():.2f} lgb {x.lgb_ba_max.mean():.2f}')
