"""审计 3：从 rows_*.csv 复算 H、逐方法 ORACLE−S2、ORACLE/AE/S2 − SRS、类别构成（comp_L1、覆盖类数）。只读。"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t*_*.csv'))], ignore_index=True)
print('行数', len(R), R.cond.value_counts().to_dict())
KEY = ['target', 'split', 'rep', 'method', 'clf', 'k']
MS = ['lpm2_nogeo', 'typiclust', 'margin']
o = R[R.cond == 'ORACLE']
print('ORACLE 行', len(o), '单元', o.groupby(['target', 'split', 'rep']).ngroups, 'reps', o.rep.unique())


def tsum(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return f'{m:+.3f} [{m - q * se:+.3f}, {m + q * se:+.3f}] p={2 * stats.t.sf(abs(m / se), n - 1):.3f} n>0={int((v > 0).sum())}/{n}'


def deltas(a, b, ms, metric='ba_ge30', reps=(0,)):
    A = R[(R.cond == a) & R.method.isin(ms) & R.rep.isin(reps)].set_index(KEY)[metric]
    B = R[(R.cond == b) & R.method.isin(ms) & R.rep.isin(reps)].set_index(KEY)[metric]
    return (A - B).dropna().rename('d').reset_index()


def cv(d):
    return d.groupby(['target', 'method', 'clf', 'k']).d.mean().groupby('target').mean()


print('\nH = ORACLE − S2（3 方法）:', tsum(cv(deltas('ORACLE', 'S2', MS))))
print('AE − S2 同方法:', tsum(cv(deltas('AE', 'S2', MS))))
print('ORACLE − AE:', tsum(cv(deltas('ORACLE', 'AE', MS))))
print('各国 H:', cv(deltas('ORACLE', 'S2', MS)).round(3).to_dict())
for m in MS:
    for k in (50, 200):
        d = deltas('ORACLE', 'S2', [m]); d = d[d.k == k]
        print(f'  ORACLE−S2 {m:11s} k={k}:', tsum(cv(d)))
    print(f'  ORACLE−S2 {m:11s} 合并 :', tsum(cv(deltas('ORACLE', 'S2', [m]))))
for met in ('oa', 'ba_P2', 'ba_far', 'f1_ge30'):
    print(f'H 用 {met}:', tsum(cv(deltas('ORACLE', 'S2', MS, met))))

# 相对 SRS（同单元、同分类器、同 k；重复 0）
srs = R[(R.method == 'srs') & (R.rep == 0)].set_index(['target', 'split', 'rep', 'clf', 'k']).ba_ge30
print('\n相对 SRS（重复 0，各国均值 → t）:')
tab = []
for cond in ('ORACLE', 'AE', 'S2'):
    for m in MS:
        x = R[(R.cond == cond) & (R.method == m) & (R.rep == 0)].set_index(['target', 'split', 'rep', 'clf', 'k']).ba_ge30
        d = (x - srs.reindex(x.index)).dropna().rename('d').reset_index()
        d['method'] = m
        for k in (50, 200):
            v = cv(d[d.k == k])
            tab.append(dict(cond=cond, method=m, k=k, mean=v.mean(), lo=v.mean() - stats.t.ppf(.975, 10) * v.std(ddof=1) / np.sqrt(11),
                            hi=v.mean() + stats.t.ppf(.975, 10) * v.std(ddof=1) / np.sqrt(11), npos=int((v > 0).sum())))
T = pd.DataFrame(tab)
print(T.round(3).to_string(index=False))

# SRS 学习曲线（重复 0 与 1 都有 k=50/100/200/400；k=0 foreign_only）
lc = R[R.method.isin(['srs', 'foreign_only'])].groupby(['target', 'k']).ba_ge30.mean().unstack()
print('\nSRS 学习曲线（ba_ge30，各国，重复与分类器平均）:')
print(lc.round(2).to_string())
print('均值:', lc.mean().round(2).to_dict())

# 类别构成
print('\n类别构成（重复 0，k 与 分类器 不重复计：取 clf=s2 行）:')
c = R[(R.rep == 0) & (R.clf == 's2') & (R.method.isin(MS + ['srs']))]
c = c[(c.cond.isin(['ORACLE', 'AE', 'S2'])) | (c.method == 'srs')]
print(c.groupby(['method', 'cond', 'k'])[['comp_L1', 'n_classes_sel']].mean().round(3).to_string())
