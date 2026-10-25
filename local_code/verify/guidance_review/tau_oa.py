import numpy as np, pandas as pd
from scipy import stats
w = pd.read_csv('check2_cell_diffs_oa.csv')
p = w.pivot_table(index=['target', 'budget', 'seed'], columns='clf_feat', values='d_fam')
main = ((p['ae'] + p['s2']) / 2).rename('m').reset_index()
cm = main.groupby('target').m.mean(); n_per = main.groupby('target').size().mean()
within = main.groupby('target').m.var(ddof=1).mean(); v = cm.var(ddof=1)
print('OA 分类器平均 d_fam 国家均值', cm.round(3).to_dict())
print('tau 矩估计 %.3f' % np.sqrt(max(v - within / n_per, 0)))
for q in [0.8, 0.95]:
    print(int(q*100), '% 上界', round(float(np.sqrt(max(3 * v / stats.chi2.ppf(1 - q, 3) - within / n_per, 0))), 2))
print('去掉 PT：tau 矩估计 %.3f' % np.sqrt(max(cm.drop('PT').var(ddof=1) - main[main.target != 'PT'].groupby('target').m.var().mean() / n_per, 0)))
inter = ((p['ae'] - p['s2']) / 2).groupby('target').mean()
print('OA 匹配交互（半差）逐国', inter.round(3).to_dict())
