"""按 ANALYSIS_PLAN_check2.md 独立实现主要比较（未参考 analyze_check2.py）。
单元 = (target, budget, clf_feat)，先对 3 个种子取平均；差值单位 pp。"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = '/Users/perng_mini/AlphaCrops_local/results'
OUTD = '/Users/perng_mini/AlphaCrops_local/verify/check2'
METRIC = sys.argv[1] if len(sys.argv) > 1 else 'ba_ge30'
TG = ['PT', 'SE', 'SK', 'LV']

sel = pd.read_csv(f'{R}/pilot_selection.csv')
bet = pd.read_csv(f'{R}/pilot_betti_parcel.csv')
bet = bet.assign(strategy='betti_' + bet.strategy)
cols = ['target', 'strategy', 'budget', 'seed', 'clf_feat', 'ba', 'ba_ge30', 'macro_f1']
allr = pd.concat([sel[cols], bet[cols]], ignore_index=True)

strat_rows = allr[allr.budget.isin([50, 200, 1000])]
# 种子平均
cell = strat_rows.groupby(['target', 'budget', 'clf_feat', 'strategy'])[METRIC].mean().unstack('strategy')
nseed = strat_rows.groupby(['target', 'budget', 'clf_feat', 'strategy']).seed.nunique().unstack('strategy')
assert (nseed == 3).all().all() and len(cell) == 24


def boot_ci(dunit, B=5000, seed=0):
    """按目标国聚类 bootstrap：重抽 4 个目标国（放回），统计量为抽中目标国全部单元的平均差。"""
    rng = np.random.default_rng(seed)
    by_t = {t: dunit.xs(t, level='target').to_numpy() for t in TG}
    stats = []
    for _ in range(B):
        ts = rng.choice(TG, 4, replace=True)
        stats.append(np.concatenate([by_t[t] for t in ts]).mean())
    return np.percentile(stats, [2.5, 97.5])


def summarize(name, d):
    d = d * 100
    p = wilcoxon(d.to_numpy()).pvalue if (d != 0).any() else np.nan
    per_t = d.groupby(level='target').mean().reindex(TG)
    main_sign = np.sign(d.mean())
    n_consistent = int((np.sign(per_t) == main_sign).sum())
    lo, hi = boot_ci(d)
    verdict = (abs(d.mean()) >= 1) and (p < 0.05) and (n_consistent >= 3)
    return dict(comparison=name, n_units=len(d), mean_pp=d.mean(), median_pp=d.median(), wilcoxon_p=p,
                n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
                ci_lo=lo, ci_hi=hi,
                **{f'{t}_pp': per_t[t] for t in TG}, n_tgt_same_dir=n_consistent,
                useful_pos=bool(verdict and d.mean() > 0))


res = []
# 问题 1
d_rep = cell['ae_rep'] - cell['s2_rep']
d_gap = cell['ae_gap'] - cell['s2_gap']
res.append(summarize('Q1 ae_rep-s2_rep', d_rep))
res.append(summarize('Q1 ae_gap-s2_gap', d_gap))
res.append(summarize('Q1 combined(avg of rep,gap per unit; 24)', (d_rep + d_gap) / 2))
stack = pd.concat([d_rep, d_gap], keys=['rep', 'gap'], names=['pair']).reorder_levels(['target', 'budget', 'clf_feat', 'pair'])
res.append(summarize('Q1 combined(stacked 48 pairs)', stack))
res.append(summarize("Q1' betti img_rep_ae-img_rep_s2", cell['betti_img_rep_ae'] - cell['betti_img_rep_s2']))
# 问题 2
for s in ['geo_strat', 's2_rep', 'ae_rep', 's2_gap', 'ae_gap', 'uncertainty',
          'betti_admin_rep', 'betti_img_rep_ae', 'betti_img_rep_s2']:
    res.append(summarize(f'Q2 {s}-random', cell[s] - cell['random']))
out = pd.DataFrame(res)
pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.precision', 4)
print(f'== metric {METRIC} ==')
print(out.to_string(index=False))

# 按预算分层（描述性）
print('\n-- 各策略−random 按预算的平均差（pp）--')
bud = {}
for s in ['geo_strat', 's2_rep', 'ae_rep', 's2_gap', 'ae_gap', 'uncertainty',
          'betti_admin_rep', 'betti_img_rep_ae', 'betti_img_rep_s2']:
    bud[s] = ((cell[s] - cell['random']) * 100).groupby(level='budget').mean()
print(pd.DataFrame(bud).T.round(2))
print('\n-- 各策略−random 按 clf_feat 的平均差（pp）--')
print(pd.DataFrame({s: ((cell[s] - cell['random']) * 100).groupby(level='clf_feat').mean()
                    for s in bud}).T.round(2))

# 问题 3：补回比例
ref = allr[allr.strategy.isin(['foreign_only', 'ceiling_w25', 'target_full', 'foreign+target_nat'])]
refm = ref.groupby(['target', 'clf_feat', 'strategy'])[METRIC].mean().unstack('strategy')
rows = []
for (t, k, cf), v in cell['random'].items():
    fo, ce = refm.loc[(t, cf), 'foreign_only'], refm.loc[(t, cf), 'ceiling_w25']
    # 逐种子比例的平均（敏感性）
    rs = sel[(sel.target == t) & (sel.strategy == 'random') & (sel.budget == k) & (sel.clf_feat == cf)][METRIC]
    rows.append(dict(target=t, budget=k, clf_feat=cf, fo=fo, ceiling=ce, random=v,
                     recovery=(v - fo) / (ce - fo), recovery_seedwise=((rs - fo) / (ce - fo)).mean()))
rec = pd.DataFrame(rows)
print('\n-- Q3 random 补回比例 --')
print(rec.round(4).to_string(index=False))
print(rec.groupby('budget')[['recovery', 'recovery_seedwise']].agg(['mean', 'min', 'max']).round(4))
print(rec.pivot_table(index='budget', columns='target', values='recovery').round(3))
print(rec.pivot_table(index='budget', columns='clf_feat', values='recovery').round(3))
print('\n-- 参照线（种子平均）--')
print((refm * 100).round(2))
# 所有策略的补回比例（描述性）
allrec = cell.copy()
for (t, k, cf) in cell.index:
    fo, ce = refm.loc[(t, cf), 'foreign_only'], refm.loc[(t, cf), 'ceiling_w25']
    allrec.loc[(t, k, cf)] = (cell.loc[(t, k, cf)] - fo) / (ce - fo)
print('\n-- 全部策略补回比例（按预算平均）--')
print(allrec.groupby(level='budget').mean().T.round(3))

out.to_csv(f'{OUTD}/prereg_{METRIC}.csv', index=False)
rec.to_csv(f'{OUTD}/recovery_{METRIC}.csv', index=False)
(cell * 100).to_csv(f'{OUTD}/cell_means_{METRIC}.csv')
