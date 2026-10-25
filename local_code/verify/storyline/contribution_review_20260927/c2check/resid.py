import pandas as pd, numpy as np
from scipy import stats
a=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/check3_capacity_arms.csv')
print(a.cap.unique(), a.arm.unique())
w=a.pivot_table(index=['target','feat','cap'],columns='arm',values='ba_ge30')
w['resid']=w['W0']-w['CMB']
r=w['resid'].unstack('feat')
for cap in a.cap.unique():
    x=r.xs(cap,level='cap'); d=(x.ae-x.s2)
    v=d.values;n=len(v);m=v.mean();se=v.std(ddof=1)/np.sqrt(n)
    print(cap,'BA resid AE-S2 %.2f [%.2f, %.2f] AEsmaller %d/11 wilcoxon p=%.3f'%(m,m-stats.t.ppf(.975,10)*se,m+stats.t.ppf(.975,10)*se,(v<0).sum(),stats.wilcoxon(v).pvalue))
    print('   ', d.round(2).to_dict())
    # also AE-S2 of W0 (in-domain) and L0 (cross)
    for arm in ['L0','W0']:
        y=w[arm].unstack('feat').xs(cap,level='cap'); dd=y.ae-y.s2
        print('   %s AE-S2 %.2f  n+ %d'%(arm,dd.mean(),(dd>0).sum()))
