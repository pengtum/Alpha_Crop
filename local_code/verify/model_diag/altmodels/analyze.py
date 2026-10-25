"""汇总：各模型 vs LightGBM（G 配置）的配对差（国家层面均值与 95% t 区间，df=n-1）。"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels'
P = pd.read_csv(os.path.join(OUT, 'part1_rows.csv'))
L = pd.read_csv(os.path.join(OUT, 'lgb_ref_s0r0.csv')).rename(columns={'clf': 'feat'})
L = L.assign(model='lgb', t_fit=L.t_train)[['target', 'feat', 'k', 'model', 'ba_ge30', 'oa', 't_fit']]
A = pd.concat([P[['target', 'feat', 'k', 'model', 'ba_ge30', 'oa', 't_fit']], L])


def tci(d):
    d = np.asarray(d, float); n = len(d)
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h, n, int((d > 0).sum())


def fmt(x):
    m, lo, hi, n, pos = x
    return f'{m:+6.2f} [{lo:+6.2f},{hi:+6.2f}] {pos}/{n}'


lines = []
order = ['lgb', 'hgb', 'rf', 'et', 'lr', 'knn10', 'knn30', 'mlp2', 'mlp3', 'iw_lgb', 'iw_lr', 'tonly_lgb', 'tonly_lr',
         'tonly_rf', 'tonly_knn10']
res = []
for k in (0, 200):
    for feat in ('ae', 's2', 'both'):
        base = A[(A.model == 'lgb') & (A.k == k)]
        if feat != 'both':
            base = base[base.feat == feat]
        base = base.groupby('target')[['ba_ge30', 'oa']].mean()
        for m in order:
            x = A[(A.model == m) & (A.k == k)]
            if feat != 'both':
                x = x[x.feat == feat]
            if x.empty:
                continue
            nfeat = x.groupby('target').feat.nunique()
            x = x[x.target.isin(nfeat[nfeat == (2 if feat == 'both' else 1)].index)]
            g = x.groupby('target')[['ba_ge30', 'oa', 't_fit']].mean()
            j = g.join(base, rsuffix='_lgb', how='inner')
            if len(j) < 3:
                continue
            db, do = tci(j.ba_ge30 - j.ba_ge30_lgb), tci(j.oa - j.oa_lgb)
            res.append(dict(k=k, feat=feat, model=m, n=len(j), ba=j.ba_ge30.mean(), oa=j.oa.mean(), t_fit=g.t_fit.mean(),
                            dba=db[0], dba_lo=db[1], dba_hi=db[2], dba_pos=db[4], doa=do[0], doa_lo=do[1], doa_hi=do[2],
                            doa_pos=do[4]))
R = pd.DataFrame(res)
R.to_csv(os.path.join(OUT, 'part1_summary.csv'), index=False)
pd.set_option('display.width', 250)
for (k, feat), g in R.groupby(['k', 'feat'], sort=False):
    print(f'\n=== k={k} feat={feat} ===')
    for _, r in g.iterrows():
        print(f"{r.model:12s} n={r.n:2d} ba {r.ba:6.2f} oa {r.oa:6.2f} t {r.t_fit:6.1f}s | Δba {r.dba:+6.2f} [{r.dba_lo:+6.2f},{r.dba_hi:+6.2f}] {r.dba_pos}/{r.n}"
              f" | ΔOA {r.doa:+6.2f} [{r.doa_lo:+6.2f},{r.doa_hi:+6.2f}] {r.doa_pos}/{r.n}")
