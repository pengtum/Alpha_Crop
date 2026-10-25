import numpy as np, pandas as pd
from scipy import stats
w = pd.read_csv('check2_cell_diffs.csv')
s = w[w.budget.isin([50, 200])]
cols = ['d_rep', 'd_gap', 'd_betti', 'd_fam']
print('=== 按分类器的均值（k∈{50,200}），国家均值的 t 检验（df=3）===')
for cf in ['ae', 's2']:
    cm = s[s.clf_feat == cf].groupby('target')[cols].mean()
    print(cf, cm.mean().round(3).to_dict(), ' t(d_fam)=%.2f' % stats.ttest_1samp(cm.d_fam, 0).statistic)
# 分类器平均（主效应）与交互（匹配效应）
p = s.pivot_table(index=['target', 'budget', 'seed'], columns='clf_feat', values=cols)
main = pd.DataFrame({c: (p[c]['ae'] + p[c]['s2']) / 2 for c in cols})
inter = pd.DataFrame({c: (p[c]['ae'] - p[c]['s2']) / 2 for c in cols})  # 半差：AE 分类器下额外 +x，S2 分类器下 −x
for nm, df in [('主效应（两分类器平均）', main), ('匹配交互（半差）', inter)]:
    cm = df.groupby('target').mean()
    print(f'\n{nm}: 逐国\n', cm.round(3))
    t = stats.ttest_1samp(cm, 0)
    print('均值', cm.mean().round(3).to_dict(), '\n国家层 t', np.round(t.statistic, 2), 'p', np.round(t.pvalue, 3))
    print('单元层 SD', df.std().round(3).to_dict(), ' 单元层 t(d_fam)=%.2f (n=%d, 忽略国家聚类)' % (df.d_fam.mean() / (df.d_fam.std() / np.sqrt(len(df))), len(df)))
# k=50 与 k=200 的 Δ 在同一 国×clf×种子 内的相关（去掉 国×clf×k 均值）
r = s.copy()
for c in cols:
    r[c] = r[c] - r.groupby(['target', 'budget', 'clf_feat'])[c].transform('mean')
q = r.pivot_table(index=['target', 'clf_feat', 'seed'], columns='budget', values=cols)
print('\n=== 同一种子下 k=50 与 k=200 的 Δ 残差相关 ===')
for c in cols:
    print(c, round(q[c][50].corr(q[c][200]), 2))
# 国家间 tau 的置信上界（d_fam，分类器平均，k 合并）
cm = main.groupby('target').d_fam.mean()
n_per = main.groupby('target').size().mean()
within = main.groupby('target').d_fam.var(ddof=1).mean()
v = cm.var(ddof=1)
print('\n=== 分类器平均 d_fam 的国家间 tau ===')
print('国家均值方差 %.4f，噪声部分 %.4f，tau 矩估计 %.3f' % (v, within / n_per, np.sqrt(max(v - within / n_per, 0))))
for q_ in [0.5, 0.8, 0.95]:
    up = 3 * v / stats.chi2.ppf(1 - q_, 3)
    print(f'  tau 的 {int(q_*100)}% 上界 ≈ {np.sqrt(max(up - within / n_per, 0)):.2f}')
