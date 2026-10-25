import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv('tree_check.csv')
print('完成训练数：', len(d) // 5, ' 国家：', d.target.unique())
for met in ['ba_ge30', 'oa']:
    w = d.pivot_table(index=['target', 'budget', 'seed', 'iters'], columns='strategy', values=met).reset_index()
    w['d_rep'] = w.ae_rep - w.s2_rep; w['d_gap'] = w.ae_gap - w.s2_gap
    for s in ['ae_rep', 's2_rep', 'ae_gap', 's2_gap']:
        w[f'{s}-rnd'] = w[s] - w.random
    con = ['d_rep', 'd_gap', 'ae_rep-rnd', 's2_rep-rnd', 'ae_gap-rnd', 's2_gap-rnd']
    L = w.melt(id_vars=['target', 'budget', 'seed', 'iters'], value_vars=con, var_name='contrast')
    P = L.pivot_table(index=['target', 'budget', 'seed', 'contrast'], columns='iters', values='value')
    print(f'\n===== {met} =====')
    print('各树数下的平均准确率：', d.groupby('iters')[met].mean().round(2).to_dict())
    for it in [100, 150, 200, 250]:
        diff = P[it] - P[300]
        print(f'{it} vs 300：对比量相关 r={P[it].corr(P[300]):.3f}，Spearman={stats.spearmanr(P[it], P[300])[0]:.3f}，'
              f'差的均值 {diff.mean():+.3f}，差的 SD {diff.std():.3f}，同号比例 {(np.sign(P[it]) == np.sign(P[300])).mean():.2f}，'
              f'(对比量本身 SD {P[300].std():.3f})')
    # 只看 AE−S2 指导对比
    Q = P.loc[P.index.get_level_values('contrast').isin(['d_rep', 'd_gap'])]
    for it in [150, 200]:
        diff = Q[it] - Q[300]
        print(f'  [AE−S2 对比] {it} vs 300：r={Q[it].corr(Q[300]):.3f}，差均值 {diff.mean():+.3f}，差 SD {diff.std():.3f}，'
              f'同号比例 {(np.sign(Q[it]) == np.sign(Q[300])).mean():.2f}')
    # 每个单元内 5 个策略的排序一致性
    taus = {it: [] for it in [150, 200]}
    for key, g in w.groupby(['target', 'budget', 'seed']):
        g = g.set_index('iters')[['ae_rep', 's2_rep', 'ae_gap', 's2_gap', 'random']]
        for it in taus:
            taus[it].append(stats.kendalltau(g.loc[it], g.loc[300])[0])
    print('  单元内 5 策略排序 Kendall τ（相对 300 棵）：', {k: round(np.nanmean(v), 3) for k, v in taus.items()})
t = d.drop_duplicates(['target', 'strategy', 'budget', 'seed']).copy()
tt = d.pivot_table(index=['target', 'strategy', 'budget', 'seed'], columns='iters', values='t_at_iter')
tot = t.set_index(['target', 'strategy', 'budget', 'seed']).t_train_total
print('\n耗时：每次训练总时长中位数 %.0f s；到第 150/200 轮的时间占比中位数 %.2f / %.2f（占 300 轮总训练时长）' %
      (tot.median(), (tt[150] / tot).median(), (tt[200] / tot).median()))
