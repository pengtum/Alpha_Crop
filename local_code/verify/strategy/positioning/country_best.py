"""只读：逐国、逐（指导, 分类器, k）看哪种方法相对 SRS 最好/最差（rep0，ba_ge30 与 oa）。"""
import os, glob
import numpy as np, pandas as pd
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t1_*.csv'))], ignore_index=True)
R = R[R.rep == 0]
U = ['target', 'split', 'rep']
methods = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding', 'margin', 'predstrat_srs', 'gist_margin', 'badge_c']
srs = R[R.method == 'srs'].set_index(U + ['clf', 'k'])
out = []
for metric in ['ba_ge30', 'oa']:
    A = R[R.cond.isin(['AE', 'S2']) & R.method.isin(methods) & R.k.isin([50, 200])].copy()
    A['d'] = A[metric].values - srs[metric].reindex(pd.MultiIndex.from_frame(A[U + ['clf', 'k']])).values
    C = A.groupby(['cond', 'clf', 'k', 'target', 'method']).d.mean().reset_index()
    best = C.loc[C.groupby(['cond', 'clf', 'k', 'target']).d.idxmax()]
    worst = C.loc[C.groupby(['cond', 'clf', 'k', 'target']).d.idxmin()]
    print(f'== {metric}: 88 个（指导×分类器×k×国家）格中最佳方法计数 ==')
    print(best.method.value_counts().to_string())
    print(f'== {metric}: 最差方法计数 ==')
    print(worst.method.value_counts().to_string())
    # 同一 (clf,k,country) 下 AE 指导与 S2 指导的最佳方法是否相同
    b = best.pivot_table(index=['clf', 'k', 'target'], columns='cond', values='method', aggfunc='first')
    print('AE 与 S2 指导最佳方法相同的比例:', (b.AE == b.S2).mean().round(3), f'(n={len(b)})')
    # predstrat 相对其它方法的名次
    C['rank'] = C.groupby(['cond', 'clf', 'k', 'target']).d.rank(ascending=False)
    print('predstrat_srs 平均名次(1=最好,8=最差):', C[C.method == 'predstrat_srs']['rank'].mean().round(2),
          ' kcenter 平均名次:', C[C.method == 'kcenter']['rank'].mean().round(2))
    print('逐国：predstrat 在 k=200 相对 SRS（4 个指导×分类器格平均）:')
    print(C[(C.method == 'predstrat_srs') & (C.k == 200)].groupby('target').d.mean().round(2).to_dict())
