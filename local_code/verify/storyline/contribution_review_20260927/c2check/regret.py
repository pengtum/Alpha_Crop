import pandas as pd, numpy as np
from scipy import stats
d=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/check1_per_target.csv')
p=d.pivot_table(index='target',columns='descriptor',values='regret',aggfunc='mean')  # avg over 3 classifier features
diff=p['mmd_ae']-p['mmd_s2']
print(p[['mmd_ae','mmd_s2','geo_km']].round(2))
v=diff.values; n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
print('regret AE-MMD minus S2-MMD: mean %.2f 95%% t CI [%.2f, %.2f], n_AEbetter=%d, wilcoxon p=%.3f'%(m,m-stats.t.ppf(.975,n-1)*se,m+stats.t.ppf(.975,n-1)*se,(v<0).sum(),stats.wilcoxon(v).pvalue))
rng=np.random.default_rng(0); bs=[rng.choice(v,n).mean() for _ in range(5000)]
print('bootstrap 95%%: [%.2f, %.2f]'%tuple(np.percentile(bs,[2.5,97.5])))
print('median diff', np.median(v), 'noRO', diff.drop('RO').mean().round(2))
g=p['mmd_ae']-p['geo_km']; print('AE-MMD minus geo regret mean %.2f, AE better %d/11'%(g.mean(),(g<0).sum()))
