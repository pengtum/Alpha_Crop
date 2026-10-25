import numpy as np, pandas as pd
from scipy.stats import wilcoxon
d = pd.read_csv('perclass_recall.csv')
d = d[d.strategy.isin(['random', 'uncertainty'])]
TG = ['PT', 'SE', 'SK', 'LV']
def run(g, lab):
    c2 = g.groupby(['target', 'budget', 'clf_feat', 'seed', 'strategy']).recall.mean().unstack('strategy')
    c2 = c2.groupby(level=['target', 'budget', 'clf_feat']).mean()
    dd = (c2['uncertainty'] - c2['random']) * 100
    pt = dd.groupby(level='target').mean()
    print(f'  {lab:28s} n={len(dd)} Δ={dd.mean():+.3f}  p={wilcoxon(dd).pvalue:.4f}  pos {int((dd>0).sum())}/24  '
          + ' '.join(f'{t}{pt[t]:+.2f}' for t in TG))
ge = d[d.n_test >= 30]
run(ge, 'ba_ge30 (all)')
for drop in sorted(ge.cls.unique()):
    run(ge[ge.cls != drop], f'drop {drop}')
for thr in [100, 300]:
    run(d[d.n_test >= thr], f'ba_ge{thr}')
# 去掉 SE 目标国
g = ge[ge.target != 'SE']
c2 = g.groupby(['target', 'budget', 'clf_feat', 'seed', 'strategy']).recall.mean().unstack('strategy').groupby(level=['target','budget','clf_feat']).mean()
dd = (c2['uncertainty'] - c2['random']) * 100
print('  without SE: n=%d Δ=%+.3f p=%.4f pos %d' % (len(dd), dd.mean(), wilcoxon(dd).pvalue, (dd>0).sum()))
# 每个单元的差
c2 = ge.groupby(['target', 'budget', 'clf_feat', 'seed', 'strategy']).recall.mean().unstack('strategy')
seedd = (c2['uncertainty'] - c2['random']) * 100
print('\n逐单元（种子平均）与逐种子 uncertainty−random（pp）:')
print(seedd.unstack('seed').assign(mean=lambda x: x.mean(1)).round(2))
