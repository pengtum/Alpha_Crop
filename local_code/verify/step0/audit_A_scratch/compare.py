# -*- coding: utf-8 -*-
"""主实现（rows_A.csv）与独立实现（indep_rows.csv 或诊断版）在共同对比上的国家层均值比较。
国家值 = 2 划分 × 20 次抽样平均；国家值 MC SE = sqrt(Σ_s var_s/n_s)/2；11/9 国均值 MC SE = sqrt(Σ_c se_c²)/n_c。
z = (主 − 复) / sqrt(SE_主² + SE_复²)。用法：python compare.py [indep_rows.csv 路径] [标签]"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IND = sys.argv[1] if len(sys.argv) > 1 else '/Users/perng_mini/AlphaCrops_local/verify/step0/partA_indep/indep_rows.csv'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'indep'
C11 = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
C9 = [c for c in C11 if c not in ('RO', 'PT')]
A = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partA/rows_A.csv')
B = pd.read_csv(IND)
# 统一列名
A = A.rename(columns={'s0_type': 'scen', 'draw': 'r', 'estimator': 'est', 'prior_L1_test': 'prior_L1',
                      's0_L1_frame': 'bias_S0', 'n_labels_S1': 'n_lab'})
A.loc[(A.est == 'E_S0'), 'arm'] = '-'
B = B.rename(columns={'n_labels': 'n_lab'})
B['k_or_B'] = np.where(B.arm.isin(['SRS', 'EXT-p']), 200, np.where(B.arm.isin(['EXT', 'SPR', 'SPR=']), 20, 0))
A = A[(A.scen == 'ref') | A.k_or_B.isin([0, 200, 20])]
A = A[~((A.arm.isin(['SRS', 'EXT-p'])) & (A.k_or_B != 200))]
A = A[~A.arm.str.startswith('Betti')]


def draws(df, scen, n0, feat, arm, est, col):
    q = df[(df.scen == scen) & (df.n0 == n0) & (df.feat == feat) & (df.arm == arm) & (df.est == est)]
    s = q.set_index(['target', 'split', 'r'])[col]
    assert len(s) == 440 and s.index.is_unique, (scen, n0, feat, arm, est, len(s))
    return s


def ref(df, feat, est, col='oa'):
    q = df[(df.scen == 'ref') & (df.feat == feat) & (df.est == est)]
    return q.groupby('target')[col].mean().reindex(C11)


def country(x):
    g = x.groupby(level=['target', 'split'])
    m = g.mean().groupby(level='target').mean().reindex(C11)
    se = (np.sqrt((g.var(ddof=1) / g.count()).groupby(level='target').sum()) / 2).reindex(C11)
    return m, se


items = []


def add(name, fa, fb):
    """fa/fb: 函数(df) → (国家值, 国家 MC SE)。"""
    ma, sa = fa(A); mb, sb = fb(B)
    for tag, cs in (('11', C11), ('9', C9)):
        d = ma.reindex(cs).mean() - mb.reindex(cs).mean()
        se = np.sqrt(((sa.reindex(cs) ** 2).sum() + (sb.reindex(cs) ** 2).sum())) / len(cs)
        zc = ((ma - mb) / np.sqrt(sa ** 2 + sb ** 2)).reindex(cs)
        items.append(dict(item=name, set=tag, main=ma.reindex(cs).mean(), indep=mb.reindex(cs).mean(), diff=d,
                          mc_se=se, z=d / se if se > 0 else (0.0 if abs(d) < 1e-9 else np.inf),
                          se_main=np.sqrt((sa.reindex(cs) ** 2).sum()) / len(cs),
                          se_indep=np.sqrt((sb.reindex(cs) ** 2).sum()) / len(cs),
                          worst_ctry=(zc.abs().idxmax() if zc.notna().any() else ''),
                          worst_z=(zc[zc.abs().idxmax()] if zc.notna().any() else np.nan),
                          n_ctry_absz_gt3=int((zc.abs() > 3).sum())))


zero = pd.Series(0.0, index=C11)
for f in ('ae', 's2'):
    for est in ('E_none', 'E_oracle'):
        add(f'ref {f} OA({est})', lambda d, f=f, e=est: (ref(d, f, e), zero), lambda d, f=f, e=est: (ref(d, f, e), zero))
        add(f'ref {f} BA({est})', lambda d, f=f, e=est: (ref(d, f, e, 'ba_ge30'), zero),
            lambda d, f=f, e=est: (ref(d, f, e, 'ba_ge30'), zero))


def D1(scen, n0, feat, arm, est, col='oa'):
    return lambda d: country(draws(d, scen, n0, feat, arm, est, col))


def DD(s1, s2, col='oa'):
    """同一抽样内（或按 r 配对的独立抽样）差值。s = (scen, n0, feat, arm, est)。"""
    return lambda d: country(draws(d, *s1, col) - draws(d, *s2, col))


for n0 in (100, 500, 2000):
    for col in ('bias_S0', 'prior_L1'):
        for sc in ('clus', 'rand'):
            add(f'A-c1 n0={n0} {sc} {col}(E_S0)', D1(sc, n0, 'ae', '-', 'E_S0', col), D1(sc, n0, 'ae', '-', 'E_S0', col))
        add(f'A-c1 n0={n0} clus−rand {col}(E_S0)', DD(('clus', n0, 'ae', '-', 'E_S0'), ('rand', n0, 'ae', '-', 'E_S0'), col),
            DD(('clus', n0, 'ae', '-', 'E_S0'), ('rand', n0, 'ae', '-', 'E_S0'), col))
    for f in ('ae', 's2'):
        for sc in ('clus', 'rand'):
            add(f'A-c1 n0={n0} {sc} {f} OA(E_S0)', D1(sc, n0, f, '-', 'E_S0'), D1(sc, n0, f, '-', 'E_S0'))
        add(f'A-c1 n0={n0} {f} OA(E_S0) clus−rand', DD(('clus', n0, f, '-', 'E_S0'), ('rand', n0, f, '-', 'E_S0')),
            DD(('clus', n0, f, '-', 'E_S0'), ('rand', n0, f, '-', 'E_S0')))
    for sc in ('clus', 'rand'):
        for f in ('ae', 's2'):
            a = (sc, n0, f, 'SRS', 'E_S1'); b = (sc, n0, f, 'SRS', 'E_pool'); c = (sc, n0, f, '-', 'E_S0')
            add(f'A-c2 {sc} n0={n0} {f} OA E_S1−E_pool', DD(a, b), DD(a, b))
            add(f'A-c2 {sc} n0={n0} {f} OA E_S1−E_S0', DD(a, c), DD(a, c))
        a = (sc, n0, 'ae', 'SRS', 'E_S1'); b = (sc, n0, 'ae', 'SRS', 'E_pool'); c = (sc, n0, 'ae', '-', 'E_S0')
        add(f'A-c2 {sc} n0={n0} L1 E_S1−E_pool', DD(a, b, 'prior_L1'), DD(a, b, 'prior_L1'))
        add(f'A-c2 {sc} n0={n0} L1 E_S1−E_S0', DD(a, c, 'prior_L1'), DD(a, c, 'prior_L1'))
    for est in ('E_pool', 'E_S1'):
        for arm in ('EXT', 'SPR', 'SPR='):
            add(f'A-c3 n0={n0} {est} L1 {arm}', D1('clus', n0, 'ae', arm, est, 'prior_L1'), D1('clus', n0, 'ae', arm, est, 'prior_L1'))
        for f in ('ae', 's2'):
            for x, y_ in (('SPR', 'EXT'), ('SPR=', 'EXT'), ('SPR=', 'SPR')):
                a = ('clus', n0, f, x, est); b = ('clus', n0, f, y_, est)
                add(f'A-c3 n0={n0} {est} {f} OA {x}−{y_}', DD(a, b), DD(a, b))
    add(f'A-c3 n0={n0} n_lab SPR', D1('clus', n0, 'ae', 'SPR', 'E_S1', 'n_lab'), D1('clus', n0, 'ae', 'SPR', 'E_S1', 'n_lab'))
    for f in ('ae', 's2'):
        for est in ('E_S1', 'E_pool'):
            add(f'EXT-p k200 n0={n0} {f} OA({est})', D1('clus', n0, f, 'EXT-p', est), D1('clus', n0, f, 'EXT-p', est))
for f in ('ae', 's2'):
    add(f'none SRS k200 {f} OA(E_S1)', D1('none', 0, f, 'SRS', 'E_S1'), D1('none', 0, f, 'SRS', 'E_S1'))
add('none SRS k200 L1(E_S1)', D1('none', 0, 'ae', 'SRS', 'E_S1', 'prior_L1'), D1('none', 0, 'ae', 'SRS', 'E_S1', 'prior_L1'))

R = pd.DataFrame(items)
R.to_csv(os.path.join(HERE, f'compare_{TAG}.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 60)
print(f'共 {len(R)} 条（{R.item.nunique()} 项 × 11/9 国）')
for tag in ('11', '9'):
    r = R[R.set == tag]
    fin = r[np.isfinite(r.z)]
    print(f'[{tag} 国] |z|>2: {int((fin.z.abs() > 2).sum())}，|z|>3: {int((fin.z.abs() > 3).sum())}，|z| 中位数 {fin.z.abs().median():.2f}；'
          f'无 MC 方差项 {int((r.mc_se == 0).sum())} 条，其中 |diff|>1e-9：{int(((r.mc_se == 0) & (r["diff"].abs() > 1e-9)).sum())}')
cols = ['item', 'set', 'main', 'indep', 'diff', 'mc_se', 'z', 'worst_ctry', 'worst_z', 'n_ctry_absz_gt3']
print(R[(R.z.abs() > 2)][cols].round(4).to_string())
