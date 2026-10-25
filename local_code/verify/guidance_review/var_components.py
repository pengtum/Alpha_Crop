"""G 实验功效规划：用已有数据估计方差分量（只读已有结果，不训练）。"""
import glob
import os

import numpy as np
import pandas as pd

R = '/Users/perng_mini/AlphaCrops_local/results'
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

# ------------------------------------------------------------ 检验 2
d = pd.read_csv(f'{R}/pilot_selection.csv')
d = d[d.budget.isin([50, 200, 1000]) & d.strategy.isin(['ae_rep', 's2_rep', 'ae_gap', 's2_gap', 'random', 'geo_strat'])]
d['m'] = 100 * d.ba_ge30
w = d.pivot_table(index=['target', 'budget', 'clf_feat', 'seed'], columns='strategy', values='m').reset_index()
w['d_rep'] = w.ae_rep - w.s2_rep
w['d_gap'] = w.ae_gap - w.s2_gap
w['d_rr'] = w.ae_rep - w.random  # 参照：几何方法 − 随机
w['d_geo'] = w.s2_rep - w.random
b = pd.read_csv(f'{R}/pilot_betti_parcel.csv')
b['m'] = 100 * b.ba_ge30
bw = b.pivot_table(index=['target', 'budget', 'clf_feat', 'seed'], columns='strategy', values='m').reset_index()
bw['d_betti'] = bw.img_rep_ae - bw.img_rep_s2
w = w.merge(bw[['target', 'budget', 'clf_feat', 'seed', 'd_betti', 'img_rep_ae', 'img_rep_s2', 'admin_rep']],
            on=['target', 'budget', 'clf_feat', 'seed'])
w['d_fam'] = w[['d_rep', 'd_gap', 'd_betti']].mean(1)

print('=== 检验 2：逐单元差值（pp），单元 = 国×k×分类器×种子 ===')
for kk in [[50, 200], [50, 200, 1000]]:
    s = w[w.budget.isin(kk)]
    print(f'\n预算 {kk}: n={len(s)}')
    print(s[['d_rep', 'd_gap', 'd_betti', 'd_fam']].agg(['mean', 'std']).round(3))


def varcomp(s, col):
    """嵌套分解：国家（随机）/ 国内 k×clf 组合（固定，移除）/ 种子残差。
    返回：总 SD、种子残差 SD（国×k×clf 内跨种子）、国家间 SD（矩估计，扣除噪声）。"""
    g = s.groupby(['target', 'budget', 'clf_feat'])[col]
    resid_sd = np.sqrt(g.var(ddof=1).mean())          # 种子重复间（同划分）
    cm = s.groupby('target')[col].mean()                # 国家均值
    n_per = s.groupby('target').size().mean()
    # 国家均值方差 = tau^2 + (国内格×种子)方差 / n_per；国内方差用国内全部单元的方差近似
    within_c = s.groupby('target')[col].var(ddof=1).mean()
    tau2 = cm.var(ddof=1) - within_c / n_per
    return dict(total_sd=s[col].std(), seed_sd=resid_sd, within_country_sd=np.sqrt(within_c),
                country_means=cm.round(2).to_dict(), tau_mom=np.sqrt(max(tau2, 0)), tau2_raw=tau2)


print('\n=== 方差分量（k∈{50,200}，两种分类器）===')
s = w[w.budget.isin([50, 200])]
for col in ['d_rep', 'd_gap', 'd_betti', 'd_fam']:
    print(col, {k: (np.round(v, 3) if not isinstance(v, dict) else v) for k, v in varcomp(s, col).items()})
for cf in ['ae', 's2']:
    print(f'\n-- 分类器 {cf} --')
    for col in ['d_rep', 'd_gap', 'd_betti', 'd_fam']:
        vc = varcomp(s[s.clf_feat == cf], col)
        print(col, {k: (np.round(v, 3) if not isinstance(v, dict) else v) for k, v in vc.items()})
for kk in [50, 200, 1000]:
    print(f'\n-- k={kk} --')
    for col in ['d_rep', 'd_gap', 'd_betti', 'd_fam']:
        vc = varcomp(w[w.budget == kk], col)
        print(col, {k: (np.round(v, 3) if not isinstance(v, dict) else v) for k, v in vc.items()})

# 方法间相关（去掉 国×k×clf 均值后的残差相关，以及原始单元相关）
print('\n=== 不同方法的 Δ 之间的相关（k∈{50,200}）===')
cols = ['d_rep', 'd_gap', 'd_betti']
print('原始:\n', s[cols].corr().round(2))
res = s.copy()
for c in cols:
    res[c] = res[c] - res.groupby(['target', 'budget', 'clf_feat'])[c].transform('mean')
print('去掉 国×k×clf 均值后:\n', res[cols].corr().round(2))
# 分类器之间（同一选样、不同分类器）
p = s.pivot_table(index=['target', 'budget', 'seed'], columns='clf_feat', values=cols)
for c in cols:
    print(f'{c}: corr(ae 分类器, s2 分类器) = {p[c]["ae"].corr(p[c]["s2"]):.2f}')

# 逐国 × 分类器均值表
print('\n=== 逐国 × 分类器 × k 的均值（pp）===')
print(w.groupby(['target', 'clf_feat', 'budget'])[['d_rep', 'd_gap', 'd_betti']].mean().round(2).unstack('budget'))

# 同一划分内：两个独立随机选样之差的 SD（参照）vs 两个几何方法之差
print('\n=== 单个方法跨种子 SD（国×k×clf 内）：random / ae_rep / s2_rep / img_rep ===')
for c in ['random', 'ae_rep', 's2_rep', 'ae_gap', 's2_gap', 'img_rep_ae', 'img_rep_s2']:
    print(c, round(np.sqrt(s.groupby(['target', 'budget', 'clf_feat'])[c].var(ddof=1).mean()), 3))

# ------------------------------------------------------------ headroom
h = pd.read_csv(f'{R}/headroom.csv')
h['ba_ge30'] *= 100
h['oa'] *= 100
print('\n=== headroom：20 次随机的国内 SD（S2 分类器，r=10，检验 2 的划分）===')
hr = h[h.strategy == 'random'].groupby(['target', 'budget'])[['ba_ge30', 'oa']].agg(['mean', 'std'])
print(hr.round(3))
print('合并（均方根）SD：', h[h.strategy == 'random'].groupby(['target', 'budget'])[['ba_ge30', 'oa']].var()
      .mean().pow(.5).round(3).to_dict())
ho = h[h.strategy.str.startswith('oracle')].groupby(['target', 'budget', 'strategy'])[['ba_ge30', 'oa']].std()
print('神谕 5 次的 SD：\n', ho.round(3).unstack('strategy'))

# ------------------------------------------------------------ 阶段 A r 网格（只有 SRS）
g = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{R}/stageA/rows_rgrid_*.csv'))])
print('\n=== 阶段 A r 网格：SRS 跨单元（划分×重复）SD ===')
print(g.groupby(['target', 'k', 'r'])[['ba_ge30', 'oa']].agg(['mean', 'std']).round(3))
# 分解：划分间 vs 划分内重复间
out = []
for (t, k, r), s_ in g.groupby(['target', 'k', 'r']):
    within = s_.groupby('split')['ba_ge30'].var(ddof=1).mean()
    between = s_.groupby('split')['ba_ge30'].mean().var(ddof=1) - within / 3
    within_oa = s_.groupby('split')['oa'].var(ddof=1).mean()
    between_oa = s_.groupby('split')['oa'].mean().var(ddof=1) - within_oa / 3
    out.append(dict(target=t, k=k, r=r, sd_rep_within_split=np.sqrt(within),
                    sd_split=np.sqrt(max(between, 0)), sd_rep_oa=np.sqrt(within_oa),
                    sd_split_oa=np.sqrt(max(between_oa, 0))))
out = pd.DataFrame(out)
print(out[out.r == 50].round(3))
# 配对对比（r=50 − r=1，同一选样、只改权重）的跨单元 SD：展示"同一单元内配对"能消掉多少
pv = g.pivot_table(index=['target', 'split', 'rep', 'k'], columns='r', values='ba_ge30')
pv['d'] = pv[50] - pv[1]
print('\n配对对比 r50−r1（同一选样）跨单元 SD：')
print(pv.groupby(['target', 'k'])['d'].agg(['mean', 'std']).round(3))

w.to_csv('/Users/perng_mini/AlphaCrops_local/verify/guidance_review/check2_cell_diffs.csv', index=False)
out.to_csv('/Users/perng_mini/AlphaCrops_local/verify/guidance_review/rgrid_varcomp.csv', index=False)
