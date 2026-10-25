"""审计 10：选样杠杆：SRS 学习曲线增益（k=0→50/200），与同单元内各选样（全部条件×方法）ba_ge30 的离散度对比。只读。"""
import glob, os
import numpy as np, pandas as pd
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t*_*.csv'))], ignore_index=True)
U = ['target', 'split', 'rep', 'clf']
k0 = R[R.method == 'foreign_only'].set_index(U).ba_ge30
out = []
for k in (50, 200):
    srs = R[(R.method == 'srs') & (R.k == k)].set_index(U).ba_ge30
    g = (srs - k0.reindex(srs.index)).groupby('target').mean()
    sel = R[(R.k == k) & (R.method != 'srs') & (R.rep == 0)]
    sd = sel.groupby(U).ba_ge30.std().groupby('target').mean()
    rng = sel.groupby(U).ba_ge30.agg(lambda x: x.max() - x.min()).groupby('target').mean()
    out.append(pd.DataFrame({f'SRS增益 k0→{k}': g, f'k={k} 单元内SD': sd, f'k={k} 单元内极差': rng}))
T = pd.concat(out, axis=1)
print(T.round(2).to_string()); print('中位数'); print(T.median().round(2).to_string())
