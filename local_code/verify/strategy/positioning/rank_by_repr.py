"""定位视角的轻量只读计算：表征（指导 cond × 分类器 clf）是否改变策略排序。
只读 G 的 rows_t1；重复 0 的 22 单元；方法 − SRS 先国内平均，再 11 国等权。"""
import os, sys, glob
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np, pandas as pd
from scipy import stats
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t1_*.csv'))], ignore_index=True)
R = R[R.rep == 0]
U = ['target', 'split', 'rep']
srs = R[(R.method == 'srs')].set_index(U + ['clf', 'k'])
methods = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding', 'margin', 'predstrat_srs', 'gist_margin', 'badge_c']
rows = []
for metric in ['ba_ge30', 'oa']:
    for cond in ['AE', 'S2']:
        for clf in ['s2', 'ae']:
            for k in [50, 200]:
                for m in methods:
                    a = R[(R.cond == cond) & (R.method == m) & (R.clf == clf) & (R.k == k)].set_index(U + ['clf', 'k'])[metric]
                    d = (a - srs[metric].reindex(a.index)).dropna()
                    cv = d.groupby(level='target').mean()
                    rows.append(dict(metric=metric, cond=cond, clf=clf, k=k, method=m, mean=cv.mean(),
                                     se=cv.std(ddof=1) / np.sqrt(len(cv)), npos=int((cv > 0).sum()), n=len(cv)))
T = pd.DataFrame(rows)
T.to_csv('/Users/perng_mini/AlphaCrops_local/verify/strategy/positioning/method_vs_srs_rep0.csv', index=False)
pd.set_option('display.width', 200)
for metric in ['ba_ge30', 'oa']:
    P = T[T.metric == metric].pivot_table(index='method', columns=['cond', 'clf', 'k'], values='mean').round(2)
    print(f'==== {metric}: 方法 − SRS（pp，11 国等权，rep0）====')
    print(P)
    # 排序一致性
    cols = list(P.columns)
    print('best method per column:', {c: P[c].idxmax() for c in cols})
    print('worst method per column:', {c: P[c].idxmin() for c in cols})
    # Kendall between guidance AE vs S2 (same clf,k)
    for clf in ['s2', 'ae']:
        for k in [50, 200]:
            t, p = stats.kendalltau(P[('AE', clf, k)], P[('S2', clf, k)])
            print(f'  Kendall τ 指导 AE vs S2 | clf={clf} k={k}: {t:+.2f} (p={p:.2f})')
    for cond in ['AE', 'S2']:
        for k in [50, 200]:
            t, p = stats.kendalltau(P[(cond, 's2', k)], P[(cond, 'ae', k)])
            print(f'  Kendall τ 分类器 s2 vs ae | cond={cond} k={k}: {t:+.2f} (p={p:.2f})')
    for cond in ['AE', 'S2']:
        for clf in ['s2', 'ae']:
            t, p = stats.kendalltau(P[(cond, clf, 50)], P[(cond, clf, 200)])
            print(f'  Kendall τ k=50 vs k=200 | cond={cond} clf={clf}: {t:+.2f} (p={p:.2f})')
# SRS learning curve per clf (country mean) from cond none
S = R[(R.method.isin(['srs', 'foreign_only']))]
C = S.groupby(['clf', 'k', 'target']).ba_ge30.mean().groupby(['clf', 'k']).mean().round(2)
print('SRS 曲线（ba_ge30，11 国均值，rep0）:'); print(C.unstack(0))
C2 = S.groupby(['clf', 'k', 'target']).oa.mean().groupby(['clf', 'k']).mean().round(2)
print('SRS 曲线（oa）:'); print(C2.unstack(0))
