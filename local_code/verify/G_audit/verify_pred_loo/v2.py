import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/verify_pred_loo/pred_cells.csv')
# per country x method
cm = d.groupby(['target','method','clf','k']).d.mean().groupby(['target','method']).mean().unstack()
print('country x method (pred pathway):\n', cm.round(3))
print('method means across countries:', cm.mean().round(3).to_dict())
for m in cm.columns:
    s = cm[m]; t = stats.ttest_1samp(s, 0)
    print(m, 'mean', round(s.mean(),3), 'p', round(t.pvalue,4), 'pos', (s>0).sum(), 'median', round(s.median(),3))
# per country x split (unit level)
cs = d.groupby(['target','split']).d.mean().unstack()
print('\ncountry x split:\n', cs.round(3))
print('\nRO cells:\n', d[d.target=='RO'].sort_values(['method','clf','k','split']).to_string(index=False))
print('\nSE cells:\n', d[d.target=='SE'].sort_values(['method','clf','k','split']).to_string(index=False))
# within-country SE of country value from 16 cells, 2 units
v = d.groupby(['target','method','clf','k']).d.mean().groupby('target').mean()
u = d.groupby(['target','split']).d.mean()
print('\nunit-level SD within country (2 units):', u.groupby('target').std(ddof=1).round(3).to_dict())
# Grubbs-style z
z = (v - v.mean())/v.std(ddof=1)
print('z scores:', z.round(2).to_dict())
n=11; a=0.05; tcrit = stats.t.ppf(1-a/(2*n), n-2); G = (n-1)/np.sqrt(n)*np.sqrt(tcrit**2/(n-2+tcrit**2))
print('Grubbs crit n=11 a=0.05', round(G,3))
# contribution of RO+SE to mean
print('RO+SE contribution to mean', round((v['RO']+v['SE'])/11,3), 'of', round(v.mean(),3))
# Holm across 3 pathways: geom p=.543, pred p=.031, hyb p=.454
ps = np.array([0.543,0.031,0.454]); order=np.argsort(ps)
print('Bonferroni pred over 3 pathways', 0.031*3)
