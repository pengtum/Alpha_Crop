"""只读：从 G 的 rows_t1 CSV 检查 4 个单元 (s0/s1 x r0/r1) 上 ba_i300 - ba_i200、ba_i250 的趋势（k=0 foreign_only 与 SRS k=200）。"""
import pandas as pd, numpy as np, glob
from scipy import stats
GOUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(f'{GOUT}/rows_t1_*.csv')])
R=R[((R.method=='foreign_only')&(R.k==0))|((R.method=='srs')&(R.cond=='none'))]
R['d300_200']=R.ba_i300-R.ba_i200
R['d250_200']=R.ba_i250-R.ba_i200
R['doa']=R.oa_i300-R.oa_i200
for (k),g in R.groupby('k'):
    for (sp,rp),h in g.groupby(['split','rep']):
        d=h.d300_200
        print(f'k={k:3d} s{sp}r{rp}: n={len(d)} mean d(300-200) {d.mean():+.2f}  neg {int((d<0).sum())}/{len(d)}  dOA {h.doa.mean():+.2f}')
    d=g.d300_200
    t=stats.ttest_1samp(g.groupby('target').d300_200.mean(),0)
    print(f'k={k:3d} ALL: n={len(d)} mean {d.mean():+.2f} neg {int((d<0).sum())}/{len(d)}; country-level mean {g.groupby("target").d300_200.mean().mean():+.2f} p={t.pvalue:.3f}')
    print('   per country mean d:', g.groupby('target').d300_200.mean().round(2).to_dict())
# SE 全部单元
print(R[R.target=='SE'][['split','rep','k','clf','ba_i200','ba_i250','ba_i300']].sort_values(['k','split','rep','clf']).round(2).to_string())
