import sys, os
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
os.chdir(sys.path[0])
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
import guidance_G as G
from guidance_G import PRED, E1, HYB, pool_positions
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
V = "/Users/perng_mini/AlphaCrops_local/verify/G_audit/pred"
R = A.load_rows(OUT)
D = G.GData()
print('classes', D.classes)
# per-class recall deltas, classify big classes by test prevalence rank within unit
out=[]
for (tgt, split), g in R[(R.rep==0)&R.method.isin(PRED+E1+HYB)&R.cond.isin(['AE','S2'])].groupby(['target','split']):
    S = np.load(f'{OUT}/stage0/{tgt}_s{split}.npz')
    yT = D.y[S['test']]; n = np.bincount(yT, minlength=10); big = np.flatnonzero(n>=30)
    pt = n/n.sum()
    order = big[np.argsort(-pt[big])]
    rank = {c:i for i,c in enumerate(order)}
    a = g[g.cond=='AE'].set_index(['method','clf','k']); b = g[g.cond=='S2'].set_index(['method','clf','k'])
    for c in big:
        dr = 100*(a[f'rec{c}']-b[f'rec{c}'])
        for (m,clf,k), v in dr.items():
            out.append(dict(target=tgt, split=split, method=m, clf=clf, k=k, cls=c, cname=D.classes[c], rank=rank[c], nbig=len(big), prev=pt[c], drec=v))
C = pd.DataFrame(out)
C['half'] = np.where(C['rank'] < C['nbig']/2, 'majority_half','minority_half')
C['fam'] = C.method.map({**{m:'geom' for m in E1}, **{m:'pred' for m in PRED}, **{m:'hyb' for m in HYB}})
# country values of mean Δrec by half
for fam in ('pred','geom','hyb'):
    for h in ('majority_half','minority_half'):
        x = C[(C.fam==fam)&(C.half==h)].groupby(['target','method','clf','k','split']).drec.mean().groupby(['target','method','clf','k']).mean().groupby('target').mean()
        print(f'{fam:5s} {h:14s}', A.fmt(A.t_summary(x)))
# by class name (pred)
print('\npred path Δrec by class (country value mean over countries where class is big):')
x = C[C.fam=='pred'].groupby(['cname','target']).drec.mean().groupby('cname').agg(['mean','count','std'])
print(x.round(3).to_string())
# by method, top class drivers
for m in PRED:
    x = C[C.method==m].groupby(['cname','target']).drec.mean().groupby('cname').agg(['mean','count'])
    print(m); print(x.round(3).T.to_string())
# regression: drec on log prevalence, pred
g = C[C.fam=='pred'].groupby(['target','split','cls']).agg(drec=('drec','mean'), prev=('prev','first')).reset_index()
print('\npred: spearman Δrec vs test prevalence', stats.spearmanr(g.prev, g.drec))
C.to_csv(f'{V}/class_recall_deltas.csv', index=False)
