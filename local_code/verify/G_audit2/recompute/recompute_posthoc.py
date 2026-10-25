"""独立重算 G 实验事后追加部分（不 import analyze_G / guidance_G）。只读 rows_t*.csv。"""
import glob, itertools, os
import numpy as np
import pandas as pd
from scipy import stats

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, 'rows_t*_*.csv')))], ignore_index=True)
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
PRED = ['margin', 'predstrat_srs']
HYB = ['gist_margin', 'badge_c']
UNIT = ['target', 'split', 'rep']
CELL = ['method', 'clf', 'k']
SESOI = 0.3


def pair(a, b, methods, metric='ba_ge30', ks=(50, 200), reps=None, clfs=('s2', 'ae')):
    """a − b 的配对差；显式检查两边行的集合完全相同。"""
    S = R[R.k.isin(ks) & R.clf.isin(clfs) & R.method.isin(methods)]
    if reps is not None:
        S = S[S.rep.isin(reps)]
    A = S[S.cond == a].set_index(UNIT + CELL)[metric].sort_index()
    B = S[S.cond == b].set_index(UNIT + CELL)[metric].sort_index()
    common = A.index.intersection(B.index)
    info = (len(A), len(B), len(common))
    d = (A.loc[common] - B.loc[common]).rename('d').reset_index()
    return d, info


def cval(d):
    # 两种聚合：先格子内对单元取均值再对格子等权；以及直接平均（平衡时相等）
    v1 = d.groupby(['target'] + CELL).d.mean().groupby('target').mean()
    v2 = d.groupby('target').d.mean()
    assert np.allclose(v1.values, v2.reindex(v1.index).values), 'unbalanced'
    return v1


def ts(v):
    v = np.asarray(v, float); n = len(v)
    m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    q95, q90 = stats.t.ppf(.975, n - 1), stats.t.ppf(.95, n - 1)
    return dict(n=n, mean=m, se=se, lo95=m - q95 * se, hi95=m + q95 * se, lo90=m - q90 * se, hi90=m + q90 * se,
                p=2 * stats.t.sf(abs(m / se), n - 1), npos=int((v > 0).sum()), nneg=int((v < 0).sum()))


def f(s):
    return (f"{s['mean']:+.4f} [{s['lo95']:+.4f},{s['hi95']:+.4f}] 90%[{s['lo90']:+.4f},{s['hi90']:+.4f}] "
            f"p={s['p']:.4f} pos {s['npos']}/{s['n']} neg {s['nneg']}")


def show(lab, d, info=None):
    s = ts(cval(d))
    print(f'{lab:<45} {f(s)}' + (f'  rows A/B/common={info}' if info else ''))
    return s


print('== 数据', len(R), 'rows; dup', R.duplicated(['cond'] + UNIT + CELL).sum())

# ---------------- θ_E1 基线
dE1, i = pair('AE', 'S2', E1)
vE1 = cval(dE1)
sE1 = show('θ_E1', dE1, i)
print('各国', vE1.round(3).to_dict())

# ---------------- ORACLE2
two = ['lpm2_nogeo', 'typiclust']
three = ['lpm2_nogeo', 'typiclust', 'margin']
print('\n== ORACLE2 / ORACLE')
for a, ms, lab in (('ORACLE2', two, 'ORACLE2−S2 (两方法)'), ('ORACLE', two, 'ORACLE−S2 (两方法)')):
    d, i = pair(a, 'S2', ms, reps=[0]); show(lab, d, i)
for m in two:
    d, i = pair('ORACLE2', 'S2', [m], reps=[0]); show(f'  ORACLE2−S2 {m}', d, i)
    d, i = pair('ORACLE2', 'ORACLE', [m], reps=[0]); show(f'  ORACLE2−ORACLE {m}', d, i)
d2, _ = pair('ORACLE2', 'S2', two, reps=[0])
dm, _ = pair('ORACLE', 'S2', ['margin'], reps=[0])
dH2 = pd.concat([d2, dm])
sH2 = show('H_ORACLE2 (3 方法等权)', dH2)
# 另一种等权：先每方法国家值再三方法平均
vH2_alt = pd.concat({m: cval(dH2[dH2.method == m]) for m in three}, axis=1).mean(1)
print('  H_ORACLE2 alt (方法国家值再平均)', f(ts(vH2_alt)))
dH, _ = pair('ORACLE', 'S2', three, reps=[0]); vH = cval(dH); sH = show('H_ORACLE (预注册)', dH)
dA3, _ = pair('AE', 'S2', three, reps=[0]); show('同三方法 AE−S2', dA3)

# ---------------- ORACLE 逐方法 + Friedman
print('\n== ORACLE 逐方法 (rep0)')
Mf = {}
for met in ('ba_ge30', 'oa', 'ba_P2'):
    cols = {}
    for m in three:
        d, _ = pair('ORACLE', 'S2', [m], met, reps=[0])
        v = cval(d); cols[m] = v
        s = ts(v); print(f'  {m:<11} {met:<8} {f(s)}')
    Mf[met] = pd.DataFrame(cols)
for met in ('ba_ge30', 'oa'):
    M = Mf[met]
    print(f'  Friedman {met}: p={stats.friedmanchisquare(*[M[m] for m in three]).pvalue:.5f}')
# 相对 SRS
S0 = R[(R.rep == 0) & R.clf.isin(['s2', 'ae']) & R.k.isin([50, 200])]
srs = S0[S0.method == 'srs'].set_index(UNIT + ['clf', 'k']).ba_ge30
for m in three:
    a = S0[(S0.cond == 'ORACLE') & (S0.method == m)].set_index(UNIT + ['clf', 'k']).ba_ge30
    dd = (a - srs.reindex(a.index)).rename('d').reset_index().assign(method=m)
    print(f'  {m} ORACLE−SRS', f(ts(cval(dd))))
# margin OA 全负?
d, _ = pair('ORACLE', 'S2', ['margin'], 'oa', reps=[0]); print('  margin OA 国家值', cval(d).round(3).to_dict())

# ---------------- RAWZ / NATIVE
print('\n== 层级 3 (k=50, rep0)')
for ca, cb, ms, lab in (('RAWZ_AE', 'RAWZ_S2', E1, 'RAWZ'), ('NATIVE_AE', 'NATIVE_S2', two, 'NATIVE')):
    for a, b in ((ca, cb), ('AE', 'S2'), (ca, 'AE'), (cb, 'S2')):
        d, i = pair(a, b, ms, ks=(50,), reps=[0]); show(f'{lab}: {a}−{b}', d, i)
    # 逐方法
    for m in ms:
        d, _ = pair(ca, cb, [m], ks=(50,), reps=[0]); show(f'   {lab} {m}: {ca}−{cb}', d)
    # 逐分类器
    for c in ('s2', 'ae'):
        d, _ = pair(ca, cb, ms, ks=(50,), reps=[0], clfs=(c,)); show(f'   {lab} clf={c}: {ca}−{cb}', d)

# ---------------- 异质性
print('\n== 异质性 θ_E1')
u = dE1.groupby(UNIT).d.mean().reset_index()
groups = [g.d.to_numpy() for _, g in u.groupby('target')]
F, pF = stats.f_oneway(*groups)
print(f'  单元值 n={len(u)}; F({len(groups)-1},{len(u)-len(groups)})={F:.4f}, p={pF:.4f}')
y = np.array([g.mean() for g in groups]); v = np.array([g.var(ddof=1) / len(g) for g in groups]); w = 1 / v
Q = (w * (y - (w * y).sum() / w.sum()) ** 2).sum()
print(f'  Cochran Q={Q:.3f} p={stats.chi2.sf(Q, len(y)-1):.4f} I2={(Q-(len(y)-1))/Q:.3f}')
# 去 PT 与 SE
u2 = u[~u.target.isin(['PT', 'SE'])]
print('  去 PT、SE 后 F p=', stats.f_oneway(*[g.d.to_numpy() for _, g in u2.groupby('target')]).pvalue.round(4))

# ---------------- 留一国
print('\n== 留一国 θ_E1')
lo = []
for c in vE1.index:
    s = ts(vE1.drop(c)); eq = s['lo90'] > -SESOI and s['hi90'] < SESOI and s['lo95'] <= 0 <= s['hi95']
    lo.append((c, s['mean'], s['lo90'], s['hi90'], eq))
lo = pd.DataFrame(lo, columns=['drop', 'mean', 'lo90', 'hi90', 'equiv'])
print(lo.round(3).to_string(index=False))

# ---------------- 族错误率 14 项
print('\n== 次要结果族 (14 项)')
fam = {}
g = dE1.pivot_table(index=UNIT + ['method', 'k'], columns='clf', values='d').dropna()
g['gamma'] = (g['ae'] - g['s2']) / 2
fam['γ'] = g.reset_index().groupby(['target', 'method', 'k']).gamma.mean().groupby('target').mean()
for c in ('s2', 'ae'):
    fam[f'E1|clf={c}'] = cval(dE1[dE1.clf == c])
for k in (50, 200):
    fam[f'E1|k={k}'] = cval(dE1[dE1.k == k])
fam['通路 pred'] = cval(pair('AE', 'S2', PRED, reps=[0])[0])
fam['通路 hyb'] = cval(pair('AE', 'S2', HYB, reps=[0])[0])
fam['Δ(P2)'] = cval(pair('AE', 'S2', E1, 'ba_P2')[0])
fam['Δ(P1)'] = cval(pair('AE', 'S2', E1, 'ba_P1')[0])
fam['E_far'] = cval(pair('AE', 'S2', E1, 'ba_far')[0])
xy = ['lpm2_nogeo', 'kcenter']
fam['AE−S2XY'] = cval(pair('AE', 'S2XY', xy, reps=[0])[0])
fam['AE−XY'] = cval(pair('AE', 'XY', xy, reps=[0])[0])
fam['S2XY−S2'] = cval(pair('S2XY', 'S2', xy, reps=[0])[0])
fam['H_ORACLE'] = vH
V = pd.DataFrame(fam).dropna()
print('  n 国', len(V), '项', V.shape[1])
praw = pd.Series({c: ts(V[c])['p'] for c in V})
o = praw.sort_values(); m = len(o)
hv = np.minimum(np.maximum.accumulate([(m - i) * p for i, p in enumerate(o.values)]), 1)
holm = pd.Series(hv, index=o.index).reindex(praw.index)
X = V.to_numpy(float); n = X.shape[0]
tstat = lambda A: A.mean(0) / (A.std(0, ddof=1) / np.sqrt(n))
t0 = np.abs(tstat(X))
mx = np.array([np.abs(tstat(X * np.array(sg)[:, None])).max() for sg in itertools.product([1, -1], repeat=n)])
maxT = pd.Series([(mx >= t - 1e-12).mean() for t in t0], index=V.columns)
# 逐项精确符号翻转（未校正）
single = pd.Series([(np.abs(np.array([tstat(X * np.array(sg)[:, None]) for sg in itertools.product([1, -1], repeat=n)])[:, j]) >= t0[j] - 1e-12).mean() for j in range(V.shape[1])], index=V.columns)
print(pd.DataFrame(dict(p_raw=praw, holm=holm, maxT=maxT, signflip_raw=single)).round(4).to_string())
# 预测通路稳健位置与留一国
vp = fam['通路 pred']
print('  pred 中位数', round(vp.median(), 4), '20% 截尾', round(stats.trim_mean(vp, .2), 4),
      'LOO', round(min(vp.drop(c).mean() for c in vp.index), 4), round(max(vp.drop(c).mean() for c in vp.index), 4))
print('  pred 各国', vp.round(3).to_dict())
for met in ('ba_P1', 'ba_P2', 'ba_far', 'oa', 'f1_ge30'):
    print(f'  pred {met}', f(ts(cval(pair('AE', 'S2', PRED, met, reps=[0])[0]))))

# ---------------- SE 的 typiclust 贡献
print('\n== SE 逐方法 (θ_E1)')
se_m = dE1[dE1.target == 'SE'].groupby('method').d.mean()
print(se_m.round(3).to_dict(), '去 typiclust 的 SE 值', round(se_m.drop('typiclust').mean(), 3))

# ---------------- 标签当量（锚定）
print('\n== 标签当量')
from sklearn.isotonic import IsotonicRegression


def curve(t, clf):
    g = R[(R.target == t) & (R.clf == clf) & (((R.method == 'srs') & (R.cond == 'none')) | (R.method == 'foreign_only'))]
    c = g.groupby('k').ba_ge30.mean().sort_index()
    ks = c.index.to_numpy(float)
    return ks, IsotonicRegression(increasing=True).fit_transform(ks, c.to_numpy()), c


def finv(ks, fv, yv):
    if yv > fv.max() + 1e-12:
        return np.inf
    if yv < fv.min() - 1e-12:
        return -np.inf
    # inf{x: f(x) >= y}
    if yv <= fv[0]:
        return ks[0]
    j = np.argmax(fv >= yv - 1e-12)
    x0, x1, y0, y1 = ks[j - 1], ks[j], fv[j - 1], fv[j]
    return x0 + (yv - y0) * (x1 - x0) / (y1 - y0)


def leq(ks, fv, k, delta):
    fk = np.interp(k, ks, fv)
    return finv(ks, fv, fk + delta) - finv(ks, fv, fk)


names = {'θ_E1': sE1['mean'], 'L95': sE1['lo95'], 'U95': sE1['hi95'], '+0.3': 0.3, '−0.3': -0.3}
rows = []
curves = {}
for t in sorted(R.target.unique()):
    for clf in ('s2', 'ae'):
        ks, fv, c = curve(t, clf)
        curves[(t, clf)] = (ks, fv, c)
        for k in (50, 200):
            for nm, dv in names.items():
                rows.append(dict(target=t, clf=clf, k=k, what=nm, L=leq(ks, fv, k, dv)))
L = pd.DataFrame(rows)


def summ(x):
    x = np.asarray(x, float); fin = x[np.isfinite(x)]
    return pd.Series(dict(median=np.median(x), n_pinf=(x == np.inf).sum(), n_ninf=(x == -np.inf).sum(),
                          q25=np.percentile(fin, 25) if len(fin) else np.nan, q75=np.percentile(fin, 75) if len(fin) else np.nan,
                          med_rel=np.median(x) / 1))
print(L.groupby(['what', 'k']).L.apply(summ).unstack().round(1).to_string())
# 相对预算的倍数（+0.3）
for k in (50, 200):
    x = L[(L.what == '+0.3') & (L.k == k)].L.to_numpy()
    print(f'  +0.3 在 k={k}：各曲线 L/k = {np.sort(np.round(x / k, 2))}')
# 平台：k=50 处 f(50)==f(0) 的曲线数
flat = sum(1 for (t, c), (ks, fv, _) in curves.items() if np.isclose(np.interp(50, ks, fv), fv[0]))
print('  f(50)=f(0)（PAV 平台，任何负 Δ 在 k=50 都为 −∞）的曲线数', flat)
atmax = sum(1 for (t, c), (ks, fv, _) in curves.items() if np.isclose(np.interp(200, ks, fv), fv.max()))
print('  f(200)=max 的曲线数', atmax)
# 学习曲线增益
G = R[(R.cond == 'none')].groupby(['target', 'clf', 'k']).ba_ge30.mean().unstack('k')
print('  增益 0→50 中位数', round((G[50] - G[0]).median(), 3), '50→400 中位数', round((G[400] - G[50]).median(), 3),
      '均值', round((G[400] - G[50]).mean(), 3), '不单调', int((G.diff(axis=1).iloc[:, 1:] < 0).any(axis=1).sum()))
# 跨国平均曲线
for clf in ('s2', 'ae'):
    c = G.xs(clf, level='clf').mean()
    ks = c.index.to_numpy(float); fv = IsotonicRegression().fit_transform(ks, c.to_numpy())
    print(f'  平均曲线 {clf}:', {(k, nm): round(leq(ks, fv, k, dv), 1) for k in (50, 200) for nm, dv in names.items()})
L.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit2/recompute/label_equiv_anchored.csv', index=False)
