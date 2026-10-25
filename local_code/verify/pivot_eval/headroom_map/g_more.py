import pandas as pd, numpy as np
from scipy import stats
D = pd.read_pickle('G_all.pkl')
key=['target','split','rep','clf','k']
B = D[(D.cond=='none')&(D.k.isin([50,200]))].set_index(key)
def ci(x):
    x=np.asarray(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(len(x)); t=stats.t.ppf(.975,len(x)-1)
    return f'{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] {int((x>0).sum())}/{len(x)}'
print('## 方法 − SRS 在 P0 / P2（先验校正到真实池先验）下，k=200；P0−P2 差 ≈ 可由"有效先验"解释的部分')
for cond in ['AE','S2','ORACLE','ORACLE2']:
    for meth in ['predstrat_srs','margin','typiclust','lpm2_nogeo','gist_margin','badge_c']:
        P=D[(D.cond==cond)&(D.method==meth)]
        if P.empty: continue
        P=P.groupby(key)[['ba_ge30','ba_P2','oa','oa_P2']].mean()
        j=P.join(B[['ba_ge30','ba_P2','oa','oa_P2']],rsuffix='_s',how='inner')
        for k in [50,200]:
            jj=j.xs(k,level='k').groupby('target').mean()
            print(f'{cond:7s} {meth:13s} k={k:3d} dBA_P0 {ci(jj.ba_ge30-jj.ba_ge30_s)} | dBA_P2 {ci(jj.ba_P2-jj.ba_P2_s)} | dOA {ci(jj.oa-jj.oa_s)}')
# in-sample best-of-16 per k: max over (cond,method) of 11-country mean
