import pandas as pd, numpy as np
from scipy.stats import spearmanr
R='/Users/perng_mini/AlphaCrops_local/results/'
dd=pd.read_csv(R+'domain_distance.csv'); pw=pd.read_csv(R+'pairwise_v2.csv')
sym=pd.concat([dd, dd.rename(columns={'c1':'c2','c2':'c1'})], ignore_index=True)
cross=pw[pw.source!=pw.target].merge(sym,left_on=['source','target'],right_on=['c1','c2'])
per=pd.read_csv(R+'check1_per_target.csv')
print(per[(per.descriptor=='c2st_ae')].pivot(index='target',columns='feature',values='rho').round(3).to_string())
rows=[]
for d in ['c2st_ae','climate_diff','c2st_s2','mmd_ae','mmd_s2','geo_km']:
    for feat in ['ae','s2','fusion']:
        for t in sorted(cross.target.unique()):
            g=cross[(cross.feature==feat)&(cross.target==t)]
            mn=g[d].min(); tied=g[np.isclose(g[d],mn,rtol=0,atol=1e-12)]
            best=g.ba.max()
            # expected top1/regret under random tie-breaking
            rows.append(dict(d=d,feat=feat,t=t,ntied=len(tied),picked=g.loc[g[d].idxmin(),'source'],
                             top1_first=int(g.loc[g[d].idxmin(),'ba']==best),
                             top1_rand=(tied.ba==best).mean(),
                             reg_first=100*(best-g.loc[g[d].idxmin(),'ba']), reg_rand=100*(best-tied.ba.mean())))
r=pd.DataFrame(rows)
print(r[r.ntied>1].groupby(['d','feat']).agg(n_targets_tied=('t','size')).to_string())
print(r.groupby(['d','feat'])[['top1_first','top1_rand','reg_first','reg_rand']].mean().round(3).to_string())
print(r[(r.d=='c2st_ae')&(r.ntied>1)&(r.feat=='ae')][['t','ntied','picked','top1_first','top1_rand']].to_string())
