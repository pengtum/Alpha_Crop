"""(a) 各方法相对 SRS：指导 × 分类器 × k 分格、合并、收缩估计；对照条件。"""
from common_bm import *

R = load()
rows = []
for metric in ('ba_ge30', 'oa'):
    for cond in ('AE', 'S2'):
        for m in ALL8:
            d = vs_srs(R, cond, [m], metric)
            for clf in ('s2', 'ae'):
                for k in (50, 200):
                    s = ts(cv(d[(d.clf == clf) & (d.k == k)]))
                    rows.append(dict(metric=metric, method=m, cond=cond, clf=clf, k=k, n_units=d[(d.clf == clf) & (d.k == k)].shape[0], **s))
            s = ts(cv(d))
            rows.append(dict(metric=metric, method=m, cond=cond, clf='pool', k='pool', n_units=len(d), **s))
        # 合并 AE+S2
    for m in ALL8:
        d = pd.concat([vs_srs(R, c, [m], metric) for c in ('AE', 'S2')])
        s = ts(cv(d))
        rows.append(dict(metric=metric, method=m, cond='AE+S2', clf='pool', k='pool', n_units=len(d), **s))
        for clf in ('s2', 'ae'):
            for k in (50, 200):
                s = ts(cv(d[(d.clf == clf) & (d.k == k)]))
                rows.append(dict(metric=metric, method=m, cond='AE+S2', clf=clf, k=k, n_units=0, **s))
T = pd.DataFrame(rows)
T['p_holm_pool'] = np.nan
T['q_bh_pool'] = np.nan
for metric in ('ba_ge30', 'oa'):
    # 族：8 方法 × 2 指导 的合并值（16 项）
    msk = (T.metric == metric) & (T.clf == 'pool') & T.cond.isin(['AE', 'S2'])
    T.loc[msk, 'p_holm_pool'] = holm(T.loc[msk, 'p']).to_numpy()
    T.loc[msk, 'q_bh_pool'] = bh(T.loc[msk, 'p']).to_numpy()
    # 族：8 方法 × 2 指导 × 2 分类器 × 2 k 的分格值（64 项）
    msk2 = (T.metric == metric) & (T.clf != 'pool') & T.cond.isin(['AE', 'S2'])
    T.loc[msk2, 'p_holm_cells'] = holm(T.loc[msk2, 'p']).to_numpy()
    T.loc[msk2, 'q_bh_cells'] = bh(T.loc[msk2, 'p']).to_numpy()
T.to_csv(f'{OUTD}/a_vs_srs_cells.csv', index=False)

# 收缩估计（A.shrink_methods）：每种指导 8 方法；以及 AE+S2 合并
sh = []
for metric in ('ba_ge30', 'oa'):
    for cond in ('AE', 'S2', 'AE+S2'):
        if cond == 'AE+S2':
            d = all_vs_srs(R, metric)
            # 两种指导等权：把 cond 并入格子
            d = d.assign(clf=d.clf + '_' + d.cond)
        else:
            d = vs_srs(R, cond, ALL8, metric)
        M = A.shrink_methods(d)
        M.insert(0, 'cond', cond)
        M.insert(0, 'metric', metric)
        sh.append(M)
SH = pd.concat(sh)
SH.to_csv(f'{OUTD}/a_shrink.csv', index=False)

# 对照条件（重复 0）：ORACLE、ORACLE2、XY、S2XY、RAWZ、NATIVE 与同子集的 AE/S2
ctrl = []
spec = [('ORACLE', ['lpm2_nogeo', 'typiclust', 'margin'], (50, 200)),
        ('ORACLE2', ['lpm2_nogeo', 'typiclust'], (50, 200)),
        ('XY', ['lpm2_nogeo', 'kcenter'], (50, 200)),
        ('S2XY', ['lpm2_nogeo', 'kcenter'], (50, 200)),
        ('RAWZ_AE', E1, (50,)), ('RAWZ_S2', E1, (50,)),
        ('NATIVE_AE', ['lpm2_nogeo', 'typiclust'], (50,)), ('NATIVE_S2', ['lpm2_nogeo', 'typiclust'], (50,))]
for metric in ('ba_ge30', 'oa'):
    for cond, ms, ks in spec:
        for m in ms:
            d = A.vs_srs(R[R.rep == 0], cond, [m], metric, ks=ks)
            s = ts(cv(d))
            ctrl.append(dict(metric=metric, cond=cond, method=m, ks=str(ks), **s))
            for ref in ('AE', 'S2'):
                dr = A.vs_srs(R[R.rep == 0], ref, [m], metric, ks=ks)
                s = ts(cv(dr))
                ctrl.append(dict(metric=metric, cond=f'{ref}(同子集)', method=m, ks=str(ks), **s))
C = pd.DataFrame(ctrl).drop_duplicates(subset=['metric', 'cond', 'method', 'ks'])
C.to_csv(f'{OUTD}/a_controls.csv', index=False)

# 标签尺度的参照：SRS 翻倍（k→2k）与 foreign_only→SRS
S = R[R.method.isin(['srs', 'foreign_only'])].groupby(['target', 'split', 'rep', 'clf', 'k'])[['ba_ge30', 'oa']].mean().unstack('k')
ref = []
for metric in ('ba_ge30', 'oa'):
    X = S[metric]
    for a, b in ((0, 50), (50, 100), (100, 200), (200, 400), (50, 200), (0, 200)):
        d = (X[b] - X[a]).rename('d').reset_index().assign(method=f'srs{b}-{a}', k=b)
        s = ts(cv(d))
        ref.append(dict(metric=metric, contrast=f'SRS k={b} − k={a}', **s))
REF = pd.DataFrame(ref)
REF.to_csv(f'{OUTD}/a_srs_curve_ref.csv', index=False)

# 方法（k）− SRS（2k）：方法能否抵上翻倍随机标签（k=50 对 SRS100；k=200 对 SRS400）
dbl = []
for metric in ('ba_ge30', 'oa'):
    X = S[metric]
    for cond in ('AE', 'S2'):
        for m in ALL8:
            g = R[(R.cond == cond) & (R.method == m) & R.k.isin([50, 200]) & R.clf.isin(['s2', 'ae'])]
            g = g.set_index(['target', 'split', 'rep', 'clf', 'k'])[metric]
            ref2 = pd.Series([X.loc[(t, s, r, c), 2 * k] for (t, s, r, c, k) in g.index], index=g.index)
            d = (g - ref2).rename('d').reset_index().assign(method=m)
            for k in (50, 200):
                s_ = ts(cv(d[d.k == k]))
                dbl.append(dict(metric=metric, cond=cond, method=m, k=k, vs=f'SRS k={2 * k}', frac_runs_ge=(d[d.k == k].d >= 0).mean(), **s_))
DBL = pd.DataFrame(dbl)
DBL.to_csv(f'{OUTD}/a_vs_srs_double.csv', index=False)

pd.set_option('display.width', 250)
fmtc = ['mean', 'lo95', 'hi95', 'n_pos', 'p']
print('=== 合并（2 分类器 × 2 k 等权）方法 − SRS ===')
for metric in ('ba_ge30', 'oa'):
    P = T[(T.metric == metric) & (T.clf == 'pool')].pivot_table(index='method', columns='cond', values=['mean', 'lo95', 'hi95', 'n_pos'])
    print(metric)
    print(P.round(3).to_string())
    print(T[(T.metric == metric) & (T.clf == 'pool') & T.cond.isin(['AE', 'S2'])][['method', 'cond', 'mean', 'p', 'p_holm_pool', 'q_bh_pool']].round(4).to_string(index=False))
print('\n=== 分格（ba_ge30）===')
X = T[(T.metric == 'ba_ge30') & (T.clf != 'pool') & T.cond.isin(['AE', 'S2'])]
print(X.assign(s=X['mean'].round(2).astype(str) + ' [' + X.lo95.round(2).astype(str) + ',' + X.hi95.round(2).astype(str) + '] ' + X.n_pos.astype(str))
      .pivot_table(index='method', columns=['cond', 'clf', 'k'], values='s', aggfunc='first').to_string())
print('\n分格中 Holm/BH 校正后 <0.05 的：')
print(X[(X.p_holm_cells < 0.05) | (X.q_bh_cells < 0.05)][['method', 'cond', 'clf', 'k', 'mean', 'lo95', 'hi95', 'n_pos', 'p', 'p_holm_cells', 'q_bh_cells']].round(4).to_string(index=False))
X = T[(T.metric == 'oa') & (T.clf != 'pool') & T.cond.isin(['AE', 'S2'])]
print('\n=== 分格（OA）===')
print(X.assign(s=X['mean'].round(2).astype(str) + ' [' + X.lo95.round(2).astype(str) + ',' + X.hi95.round(2).astype(str) + '] ' + X.n_pos.astype(str))
      .pivot_table(index='method', columns=['cond', 'clf', 'k'], values='s', aggfunc='first').to_string())
print(X[(X.p_holm_cells < 0.05) | (X.q_bh_cells < 0.05)][['method', 'cond', 'clf', 'k', 'mean', 'lo95', 'hi95', 'n_pos', 'p', 'p_holm_cells', 'q_bh_cells']].round(4).to_string(index=False))
print('\n=== 收缩 ===')
print(SH.round(3).to_string(index=False))
print('\n=== 对照 ===')
print(C[['metric', 'cond', 'method', 'ks', 'mean', 'lo95', 'hi95', 'n_pos', 'p']].round(3).to_string(index=False))
print('\n=== SRS 曲线参照 ===')
print(REF[['metric', 'contrast', 'mean', 'lo95', 'hi95', 'n_pos', 'p']].round(3).to_string(index=False))
print('\n=== 方法(k) − SRS(2k) ===')
print(DBL.pivot_table(index='method', columns=['metric', 'cond', 'k'], values='mean').round(2).to_string())
print(DBL.pivot_table(index='method', columns=['metric', 'cond', 'k'], values='n_pos').to_string())
