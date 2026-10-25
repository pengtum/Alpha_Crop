"""独立核实：'等价缺乏检测敏感性'（refute 视角提出）。只读结果文件。"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import guidance_G as G  # noqa: E402

OUT = G.OUT
E1 = G.E1
UNIT = ['target', 'split', 'rep']
KEY = UNIT + ['method', 'clf', 'k']
R = pd.concat([pd.read_csv(f) for f in glob.glob(OUT + '/rows_t*_*.csv')], ignore_index=True)
pd.set_option('display.width', 200)


def tsum(v):
    v = np.asarray(v, float)
    n = len(v)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    q90 = stats.t.ppf(0.95, n - 1)
    p = 2 * stats.t.sf(abs(m / se), n - 1)
    return f'{m:+.3f} [{m - q * se:+.3f}, {m + q * se:+.3f}] 90%[{m - q90 * se:+.3f}, {m + q90 * se:+.3f}] p={p:.3f} n+={int((v > 0).sum())}/{n}'


def cv(d, col='d'):
    return d.groupby(['target', 'method', 'clf', 'k'])[col].mean().groupby('target').mean()


def pair(a_cond, b_cond, methods, reps=None, metric='ba_ge30'):
    a = R[(R.cond == a_cond) & R.method.isin(methods)].set_index(KEY)[metric]
    b = R[(R.cond == b_cond) & R.method.isin(methods)].set_index(KEY)[metric]
    d = (a - b).dropna().rename('d').reset_index()
    if reps is not None:
        d = d[d.rep.isin(reps)]
    return d


def vs_srs(cond, methods, reps=None, metric='ba_ge30'):
    a = R[(R.cond == cond) & R.method.isin(methods)].copy()
    s = R[R.method == 'srs'].set_index(UNIT + ['clf', 'k'])[metric]
    a['d'] = a[metric].to_numpy() - s.reindex(pd.MultiIndex.from_frame(a[UNIT + ['clf', 'k']])).to_numpy()
    a = a.dropna(subset=['d'])
    if reps is not None:
        a = a[a.rep.isin(reps)]
    return a


print('=== 0. 复现 θ_E1')
print('θ_E1', tsum(cv(pair('AE', 'S2', E1))))

print('\n=== 1. E1 方法 − SRS（国家值 t 检验）')
for c in ('AE', 'S2'):
    d = vs_srs(c, E1)
    print(f'  {c} 全部单元: 池化均值 {d.d.mean():+.3f}; 国家值', tsum(cv(d)))
    for k in (50, 200):
        print(f'     k={k}:', tsum(cv(d[d.k == k])))
print('  逐方法（AE / S2），国家值：')
for m in E1:
    for c in ('AE', 'S2'):
        print(f'    {m:12s} {c}:', tsum(cv(vs_srs(c, [m]))))
print('  其他通路 − SRS（重复 0）：')
for m in ['margin', 'predstrat_srs', 'gist_margin', 'badge_c']:
    for c in ('AE', 'S2'):
        print(f'    {m:14s} {c}:', tsum(cv(vs_srs(c, [m]))))

print('\n=== 2. ORACLE')
for ms, lab in ((['lpm2_nogeo', 'typiclust', 'margin'], '全部 3 方法'), (['lpm2_nogeo', 'typiclust'], '只几何'),
                (['margin'], 'margin'), (['lpm2_nogeo'], 'lpm2'), (['typiclust'], 'typiclust')):
    print(f'  ORACLE − S2 [{lab}]:', tsum(cv(pair('ORACLE', 'S2', ms, reps=[0]))))
    print(f'  ORACLE − AE [{lab}]:', tsum(cv(pair('ORACLE', 'AE', ms, reps=[0]))))
    print(f'  ORACLE − SRS [{lab}]:', tsum(cv(vs_srs('ORACLE', ms, reps=[0]))))
    print(f'  AE − S2 同方法同单元 [{lab}]:', tsum(cv(pair('AE', 'S2', ms, reps=[0]))))

print('\n=== 3. XY / S2XY 相对 SRS（重复 0）')
for c in ('XY', 'S2XY', 'AE', 'S2'):
    print(f'  {c} − SRS (lpm2,kcenter):', tsum(cv(vs_srs(c, ['lpm2_nogeo', 'kcenter'], reps=[0]))))

print('\n=== 4. SRS 学习曲线（单元均值；k=0 为 foreign_only）')
S = R[R.method.isin(['srs', 'foreign_only'])]
fo = S[S.method == 'foreign_only']
print('  foreign_only rep0 vs rep1 相同?', np.allclose(
    fo[fo.rep == 0].set_index(['target', 'split', 'clf']).ba_ge30.sort_index(),
    fo[fo.rep == 1].set_index(['target', 'split', 'clf']).ba_ge30.sort_index()))
C = S.groupby(['target', 'clf', 'k']).ba_ge30.mean().unstack('k')
C['g0_400'] = C[400] - C[0]
C['g0_50'] = C[50] - C[0]
C['g50_200'] = C[200] - C[50]
C['g200_400'] = C[400] - C[200]
print(C.round(2).to_string())
Cm = C.groupby('target').mean()
print('  两分类器平均：')
print(Cm[['g0_400', 'g0_50', 'g50_200', 'g200_400']].round(2).to_string())
print('  0→400 增益 <0.5 pp 的国家（两分类器平均）:', list(Cm.index[Cm.g0_400 < 0.5]))
print('  0→400 增益 <0.5 pp 的国家（两分类器都 <0.5）:',
      list(C.g0_400.unstack('clf').pipe(lambda x: x[(x < 0.5).all(1)]).index))
# SRS 学习曲线的国家值 t 检验
for a, b in ((0, 50), (50, 200), (200, 400), (0, 400)):
    print(f'  SRS k={b} − k={a}（国家值，两分类器平均）:', tsum(Cm[b] - Cm[a]))

print('\n=== 5. 单元格层面：AE−S2 的 SD 与 SRS 两次重复之差的 SD')
d = pair('AE', 'S2', E1)
print(f'  AE−S2 单元格 SD（E1，全部）= {d.d.std(ddof=1):.3f}，n={len(d)}；均值 {d.d.mean():+.3f}')
sr = S[S.method == 'srs'].pivot_table(index=['target', 'split', 'clf', 'k'], columns='rep', values='ba_ge30')
sd_srs = (sr[0] - sr[1]).std(ddof=1)
print(f'  SRS r0−r1 SD（k=50,100,200,400）= {sd_srs:.3f}；只 k∈{{50,200}}: {(sr[0]-sr[1]).loc[(slice(None), slice(None), slice(None), [50, 200])].std(ddof=1):.3f}')
# 同一单元内：不同 E1 方法（同指导）之间的差，作为'换方法'的参照
for c in ('AE', 'S2'):
    x = R[(R.cond == c) & R.method.isin(E1)].pivot_table(index=UNIT + ['clf', 'k'], columns='method', values='ba_ge30')
    print(f'  {c}：同单元内 4 个 E1 方法的 SD（均值）= {x.std(axis=1, ddof=1).mean():.3f}；方法 − SRS 单元格 SD = {vs_srs(c, E1).d.std(ddof=1):.3f}')
# 方差分解：Δ 的系统部分（国家×方法×分类器×k 的格子均值）占比
g = d.groupby(['target', 'method', 'clf', 'k']).d
cell_mean = g.transform('mean')
tot = d.d.var(ddof=0)
betw = cell_mean.var(ddof=0)
print(f'  Δ 总方差 {tot:.3f}；格子均值间方差 {betw:.3f}（占 {betw / tot:.1%}，含抽样噪声，4 单元均值）')
# 国家层面
cm = cv(d)
print(f'  国家值 SD {cm.std(ddof=1):.3f}，|θ| {abs(cm.mean()):.3f}；θ²/总方差 {cm.mean() ** 2 / tot:.4f}')

print('\n=== 6. 几何方法之间确实不同吗（方法选择是否影响精度）')
for c in ('AE', 'S2'):
    x = vs_srs(c, E1)
    vals = x.groupby(['target', 'method']).d.mean().unstack('method')
    print(f'  {c} 各方法 − SRS 的国家均值矩阵：')
    print(vals.round(2).to_string())
    # 方法间差异的 F 检验（国家 × 方法 的双向，无交互）
    long = vals.stack().rename('y').reset_index()
    gm = long.y.mean()
    mm = long.groupby('method').y.mean()
    tm = long.groupby('target').y.mean()
    ss_m = 11 * ((mm - gm) ** 2).sum()
    resid = long.y - long.method.map(mm) - long.target.map(tm) + gm
    ss_e = (resid ** 2).sum()
    F = (ss_m / 3) / (ss_e / 30)
    print(f'   方法主效应 F(3,30)={F:.2f} p={stats.f.sf(F, 3, 30):.4f}')

print('\n=== 7. comp_L1 / n_classes（AE vs S2 选样类别构成）')
for c in ('AE', 'S2', 'none'):
    x = R[(R.cond == c) & (R.method.isin(E1 + ['srs'])) & R.k.isin([50, 200])]
    print(c, x.groupby('k')[['comp_L1', 'n_classes_sel']].mean().round(3).to_dict())
