"""汇总 mlp_fair_rows_{s1r0,s0r0}.csv。国家层面（ae/s2 平均）t 区间。"""
import numpy as np, pandas as pd, sys, os
from scipy import stats
V = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_altmodels/'
def tci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v)
    if n < 2: return f'n={n}'
    m = v.mean(); h = stats.t.ppf(.975, n-1) * v.std(ddof=1) / np.sqrt(n)
    return f'{m:+.2f} [{m-h:+.2f}, {m+h:+.2f}] {int((v>0).sum())}/{n}'
tags = [t for t in ('s1r0', 's0r0') if os.path.exists(V + f'mlp_fair_rows_{t}.csv')]
for tag in tags + (['both'] if len(tags) == 2 else []):
    if tag == 'both':
        R = pd.concat([pd.read_csv(V + f'mlp_fair_rows_{t}.csv').assign(u=t) for t in tags])
    else:
        R = pd.read_csv(V + f'mlp_fair_rows_{tag}.csv').assign(u=tag)
    full = R.groupby(['target', 'u']).size(); 
    print(f'\n######## {tag}: 国家数 {R.target.nunique()}，行 {len(R)}')
    for k in (200, 0):
        x = R[R.k == k]
        if x.empty: continue
        g = x.groupby(['model', 'target']).mean(numeric_only=True)
        L = g.loc['lgb']
        print(f'=== k={k} === LGB 原生 ba {L.ba0.mean():.2f} OA {L.oa0.mean():.2f} | τ=.5 ba {L.ba05.mean():.2f} OA {L.oa05.mean():.2f}')
        models = [m for m in ['lr', 'mlpES', 'mlpF_ens_e20', 'mlpF_s0_e20', 'mlpF_s1_e20', 'mlpF_s2_e20', 'mlpF_ens_e5', 'mlpF_ens_e10', 'mlpF_ens_e40',
                              'mlpF_s0_e5', 'mlpF_s0_e10', 'mlpF_s0_e40', 'mlpFa3_s0_e10', 'mlpFa3_s0_e20', 'mlpFa3_s0_e40'] if m in g.index.get_level_values(0)]
        for m in models:
            M = g.loc[m]
            print(f'{m:14s} ba {M.ba0.mean():5.2f} OA {M.oa0.mean():5.2f} trainBA {M.train_ba_foreign.mean():5.1f} | 原生−LGB原生 Δba {tci(M.ba0-L.ba0)} ΔOA {tci(M.oa0-L.oa0)}'
                  f' | 原生−LGB(τ.5) Δba {tci(M.ba0-L.ba05)} ΔOA {tci(M.oa0-L.oa05)}')
            print(f'{"":14s} 同τ.5 Δba {tci(M.ba05-L.ba05)} ΔOA {tci(M.oa05-L.oa05)} | 包络: m原生−LGB包络@m原生OA {tci(M.ba0-M.lgbenv_at_mOA0)}'
                  f' | m包络@LGB(τ.5)OA−LGB(τ.5) {tci(M.env_at_lgbOA05-L.ba05)} | m包络@LGB原生OA−LGB原生 {tci(M.env_at_lgbOA0-L.ba0)}'
                  f' | τ网格最大 m−LGB {tci(M.ba_max-L.ba_max)}')
    # 种子噪声
    x = R[(R.k == 200) & R.model.isin(['mlpF_s0_e20', 'mlpF_s1_e20', 'mlpF_s2_e20'])]
    sd = x.groupby(['u', 'target', 'feat']).ba0.std()
    print('mlpF e20 三个种子间 ba 标准差（单元×国家×特征）: 中位', round(sd.median(), 2), '最大', round(sd.max(), 2))
    tg = R[(R.k == 200) & R.model.str.startswith('mlpF_s0')].groupby('model')[['train_ba_foreign', 'tgt_train_acc']].mean().round(1)
    print(tg.to_string())
    # k 增益
    p = R[R.model.isin(['lgb', 'lr', 'mlpES', 'mlpF_s0_e20'])].pivot_table(index=['u', 'target', 'feat', 'model'], columns='k', values=['ba0', 'oa0'])
    gain = (p[('ba0', 200)] - p[('ba0', 0)]).unstack('model').groupby(['u', 'target']).mean()
    goa = (p[('oa0', 200)] - p[('oa0', 0)]).unstack('model').groupby(['u', 'target']).mean()
    for m in ('lr', 'mlpES', 'mlpF_s0_e20'):
        print(f'k 0→200 增益 {m}: {tci(gain[m])} OA {tci(goa[m])} | 减 LGB 增益 {tci(gain[m]-gain.lgb)}   (LGB 增益 {tci(gain.lgb)} OA {tci(goa.lgb)})')
