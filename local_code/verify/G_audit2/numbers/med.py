import sys
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.argv=['x']
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
from guidance_G import E1, FAMILIES
M=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/guidance_G/mediators_G.csv')
print(M.groupby(['cond']).size(), M.groupby(['method','k']).size().head(20))
J=M[M.cond=='AE']
print('jac overall mean rows', J.jaccard.mean(), 'max', J.jaccard.max()); print(J.loc[J.jaccard.idxmax(), ['target','split','rep','method','k','jaccard']])
# per country-avg (as in report) method,k
jm=J.groupby(['method','k','target']).jaccard.mean().groupby(['method','k']).mean()
print(jm.round(4)); print('range', jm.min(), jm.max(), 'mean of cells', jm.mean())
# is Jaccard AE==S2 identical rows?
print('AE vs S2 jaccard same', np.allclose(M[M.cond=='AE'].sort_values(['target','split','rep','method','k']).jaccard.values, M[M.cond=='S2'].sort_values(['target','split','rep','method','k']).jaccard.values))
# E1 only
JE=J[J.method.isin(E1)]; print('E1 jac mean', JE.jaccard.mean(), JE.groupby(['method','k']).jaccard.mean().round(4).to_dict())
# correlation
Q=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/guidance_G/guidance_quality_G.csv')
print(Q.groupby('feat')[['knn10_ba','nll_pool','auroc_margin','ece_oof','nll_oof','prior_L1_argmax','temp']].mean())
kq=Q.pivot_table(index=['target','split'],columns='feat',values='knn10_ba')
adv=(kq.ae-kq.s2).groupby('target').mean()
R=A.load_rows(); vE1=A.country_values(A.deltas(R,'AE','S2',E1))
df=pd.DataFrame(dict(adv=adv,th=vE1)); print(df.round(3))
print('pearson', stats.pearsonr(df.adv,df.th), 'spearman', stats.spearmanr(df.adv,df.th))
d2=df.drop(['SE','PT']); print('noSE,PT pearson', stats.pearsonr(d2.adv,d2.th), stats.spearmanr(d2.adv,d2.th))
for c in ['SE','PT']:
    d1=df.drop([c]); print('drop',c, stats.pearsonr(d1.adv,d1.th)[0])
# alt: adv ratio or per-split?
