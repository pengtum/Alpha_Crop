"""第 2 部分汇总：E1（4 个几何方法）× k=200 × s0r0，AE 指导 − S2 指导，按国家先平均（方法 × 分类器特征），再做 t 区间。"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels'
P2 = pd.read_csv(os.path.join(OUT, 'part2_rows.csv'))
L = pd.read_csv(os.path.join(OUT, 'lgb_e1_k200_s0r0.csv')).assign(model='lgb')
A = pd.concat([P2[['target', 'cond', 'method', 'clf', 'model', 'ba_ge30', 'oa']], L[['target', 'cond', 'method', 'clf', 'model', 'ba_ge30', 'oa']]])


def tci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean()
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h, int((v > 0).sum()), n, stats.ttest_1samp(v, 0).pvalue


percty = {}
for model in A.model.unique():
    x = A[A.model == model]
    p = x.pivot_table(index=['target', 'method', 'clf'], columns='cond', values=['ba_ge30', 'oa'])
    d = (p.xs('AE', axis=1, level=1) - p.xs('S2', axis=1, level=1)).dropna().reset_index()
    cnt = d.groupby('target').size()
    full = cnt[cnt == 8].index
    d = d[d.target.isin(full)]
    for clf in ('both', 'ae', 's2'):
        dd = d if clf == 'both' else d[d.clf == clf]
        c = dd.groupby('target')[['ba_ge30', 'oa']].mean()
        if clf == 'both':
            percty[model] = c
        for met in ('ba_ge30', 'oa'):
            m, lo, hi, pos, n, pv = tci(c[met])
            print(f'{model:5s} clf={clf:4s} {met:7s} AE−S2 = {m:+.3f} [{lo:+.3f}, {hi:+.3f}] p={pv:.2f} {pos}/{n}')
    # 各方法
    mm = d.groupby(['method', 'target']).ba_ge30.mean().unstack()
    print(model, 'per-method ba AE−S2 (country mean):', mm.mean(1).round(3).to_dict())
    # 该模型下的绝对水平
    print(model, 'mean ba (E1,k200):', x.groupby('cond').ba_ge30.mean().round(2).to_dict())
if 'lr' in percty:
    j = percty['lr'].join(percty['lgb'], lsuffix='_lr', rsuffix='_lgb')
    dd = j.ba_ge30_lr - j.ba_ge30_lgb
    m, lo, hi, pos, n, pv = tci(dd)
    print(f'\n(LR 的 AE−S2) − (LGB 的 AE−S2), ba: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] p={pv:.2f} n={n}')
    r = np.corrcoef(j.ba_ge30_lr, j.ba_ge30_lgb)[0, 1]
    print(f'国家层面 AE−S2 的相关（LR vs LGB，ba）: r={r:+.2f}')
    print(j.round(2).to_string())
for mdl in [m for m in percty if m not in ('lgb', 'lr')]:
    j = percty[mdl].join(percty['lgb'], lsuffix='_m', rsuffix='_lgb')
    m, lo, hi, pos, n, pv = tci(j.ba_ge30_m - j.ba_ge30_lgb)
    print(f'\n({mdl} 的 AE−S2) − (LGB 的 AE−S2), ba: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] p={pv:.2f} n={n}; r={np.corrcoef(j.ba_ge30_m, j.ba_ge30_lgb)[0,1]:+.2f}')
