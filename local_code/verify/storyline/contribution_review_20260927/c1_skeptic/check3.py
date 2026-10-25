import pandas as pd, numpy as np
a=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/check3_arms.csv'); s=a[a.split=='spatial']
for f in ['ae','s2']:
    o=s[s.feat==f].pivot_table(index='target',columns='arm',values='oa')
    tot=o.Wnat-o.L0; comp=o.PM-o.L0; within=o.Wnat-o.W0
    sh=0.5*(comp+within)
    print(f, 'spatial chain', round(comp.sum()/tot.sum()*100,1), 'Shapley', round(sh.sum()/tot.sum()*100,1), 'within(Wnat-W0) mean', round(within.mean(),2), 'median per-country chain share', round(np.median(comp/tot*100),1))
    m=~o.index.isin(['RO','PT'])
    print('   exROPT chain', round(comp[m].sum()/tot[m].sum()*100,1), 'Shapley', round(sh[m].sum()/tot[m].sum()*100,1))
