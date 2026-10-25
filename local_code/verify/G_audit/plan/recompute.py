"""独立重算 report_G.txt 的主要数字（不 import analyze_G）。"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in glob.glob(f'{OUT}/rows_t*_*.csv')], ignore_index=True)
print('rows', len(R), R.groupby(['cond']).size().to_dict())
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
PRED = ['margin', 'predstrat_srs']; HYB = ['gist_margin', 'badge_c']
K = ['target', 'split', 'rep', 'method', 'clf', 'k']
def dd(a, b, ms, met='ba_ge30', reps=None):
    A = R[(R.cond == a) & R.method.isin(ms)]
    B = R[(R.cond == b) & R.method.isin(ms)]
    if reps is not None:
        A = A[A.rep.isin(reps)]; B = B[B.rep.isin(reps)]
    m = A.merge(B, on=K, suffixes=('_a', '_b'))
    m['d'] = m[met + '_a'] - m[met + '_b']
    return m[K + ['d']]
def cv(d):
    # 国家值 = 该国全部配对差的简单平均（均衡时与分格平均相同）
    return d.groupby('target').d.mean()
def cv_cells(d):
    return d.groupby(['target', 'method', 'clf', 'k']).d.mean().groupby('target').mean()
def ts(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    t95, t90 = stats.t.ppf(.975, n - 1), stats.t.ppf(.95, n - 1)
    return dict(n=n, m=round(m, 3), lo95=round(m - t95 * se, 3), hi95=round(m + t95 * se, 3),
                lo90=round(m - t90 * se, 3), hi90=round(m + t90 * se, 3), p=round(2 * stats.t.sf(abs(m / se), n - 1), 3),
                npos=int((v > 0).sum()))
d = dd('AE', 'S2', E1)
print('E1 pairs', len(d), 'per country', d.groupby('target').size().unique(), 'units', d.groupby(['target','split','rep']).ngroups)
v = cv(d); v2 = cv_cells(d)
print('max|cv - cv_cells|', float((v - v2).abs().max()))
print('theta_E1', ts(v)); print(v.round(3).to_dict())
# R0
u = d.groupby(['target', 'split', 'rep']).d.mean()
wv = (u.groupby('target').var(ddof=1) / u.groupby('target').size()).mean()
y = v.to_numpy(); n = len(y)
from scipy.optimize import minimize_scalar
def nll(t2, vv):
    w = 1 / (vv + t2); mu = (w * y).sum() / w.sum()
    return .5 * (np.log(vv + t2).sum() + np.log(w.sum()) + (w * (y - mu) ** 2).sum())
vv = np.full(n, wv)
t2 = max(0, y.var(ddof=1) - wv)  # 等方差时 REML 闭式
w = 1 / (vv + t2); mu = (w * y).sum() / w.sum(); se = np.sqrt((w * (y - mu) ** 2).sum() / ((n - 1) * w.sum()))
h = stats.t.ppf(.9, n - 2) * np.sqrt(t2 + se ** 2)
print(f'R0 mu {mu:.3f} tau {np.sqrt(t2):.3f} se {se:.3f} PI [{mu-h:.3f},{mu+h:.3f}] within_var {wv:.4f}')
# 若按方案字面 t_{0.90,9} —— 同上；若 df=10 做对照
h10 = stats.t.ppf(.9, n - 1) * np.sqrt(t2 + se ** 2); print(f'  (df=10 对照 PI [{mu-h10:.3f},{mu+h10:.3f}])')
print('H2 OA', ts(cv(dd('AE', 'S2', E1, 'oa'))))
pv = {'geom': cv(dd('AE', 'S2', E1)), 'pred': cv(dd('AE', 'S2', PRED, reps=[0])), 'hyb': cv(dd('AE', 'S2', HYB, reps=[0]))}
for kk, vv_ in pv.items(): print('path', kk, ts(vv_))
print('H3 E2', ts(pd.concat(pv, axis=1).mean(1)))
# R1
S = R[R.method == 'srs'][['target', 'split', 'rep', 'clf', 'k', 'ba_ge30']].rename(columns={'ba_ge30': 'srs'})
def vs(cond):
    A = R[(R.cond == cond) & R.method.isin(E1)].merge(S, on=['target', 'split', 'rep', 'clf', 'k'])
    A['d'] = A.ba_ge30 - A.srs; return A
rA = vs('AE').assign(b=lambda x: x.d < -0.5).groupby('target').b.mean()
rS = vs('S2').assign(b=lambda x: x.d < -0.5).groupby('target').b.mean()
print('H4 R1', ts(rA - rS), 'AE', round(rA.mean(), 3), 'S2', round(rS.mean(), 3), 'wilcoxon p', round(stats.wilcoxon(rA - rS).pvalue, 3))
# gamma
g = d.pivot_table(index=['target', 'split', 'rep', 'method', 'k'], columns='clf', values='d')
print('gamma', ts(((g.ae - g.s2) / 2).groupby('target').mean()))
for c in ('s2', 'ae'): print('clf', c, ts(cv(d[d.clf == c])))
for k in (50, 200): print('k', k, ts(cv(d[d.k == k])))
for m in E1 + PRED + HYB:
    print('method', m, ts(cv(dd('AE', 'S2', [m])))['m'])
for met in ('ba_P2', 'ba_P1', 'ba_far'): print(met, ts(cv(dd('AE', 'S2', E1, met))))
a = dd('AE', 'S2', E1).set_index(K).d; b = dd('AE', 'S2', E1, 'ba_P2').set_index(K).d
print('P0-P2', ts((a - b).groupby('target').mean()))
for a_, b_ in (('AE', 'S2'), ('AE', 'S2XY'), ('AE', 'XY'), ('S2XY', 'S2')):
    print(a_, '-', b_, ts(cv(dd(a_, b_, ['lpm2_nogeo', 'kcenter'], reps=[0]))))
print('ORACLE H', ts(cv(dd('ORACLE', 'S2', ['lpm2_nogeo', 'typiclust', 'margin'], reps=[0]))))
print('AE-S2 same', ts(cv(dd('AE', 'S2', ['lpm2_nogeo', 'typiclust', 'margin'], reps=[0]))))
# R2
sd = {c: vs(c).groupby(['target', 'method', 'clf', 'k']).d.std(ddof=1) for c in ('AE', 'S2')}
print('R2 (on method-SRS)', ts(np.log(sd['AE'] / sd['S2']).groupby('target').mean()))
sd2 = {c: R[(R.cond == c) & R.method.isin(E1)].groupby(['target', 'method', 'clf', 'k']).ba_ge30.std(ddof=1) for c in ('AE', 'S2')}
print('R2 (on raw ba)', ts(np.log(sd2['AE'] / sd2['S2']).groupby('target').mean()))
for c in ('AE', 'S2'):
    x = vs(c); print('R3', c, round(x.d.mean(), 3), round((x.d < 0).mean(), 3), round((x.d < -0.5).mean(), 3))
    print('   per-country P(<SRS)', x.assign(b=x.d < 0).groupby('target').b.mean().round(2).to_dict())
# sensitivity
for lab, drop in (('noCHK2', ['PT', 'SE', 'SK', 'LV']), ('noRO', ['RO']), ('noDEV', ['PT', 'BE', 'SI'])):
    print(lab, ts(v.drop(drop)))
for it in (200, 250): print('iter', it, ts(cv(dd('AE', 'S2', E1, f'ba_i{it}'))))
# 一致性：ba_i300 == ba_ge30 ?
print('ba_i300==ba_ge30 max diff', float((R.ba_i300 - R.ba_ge30).abs().max()))
# 混合模型（单元层级，国家随机截距，均衡）：μ̂ 与 SE
uu = u.reset_index(); nc, nu = 11, 4
msb = nu * ((uu.groupby('target').d.mean() - uu.d.mean()) ** 2).sum() / (nc - 1)
print('balanced mixed model mu', round(uu.d.mean(), 3), 'SE', round(np.sqrt(msb / (nc * nu)), 4), 'vs t SE', round(v.std(ddof=1) / np.sqrt(11), 4))
# prior_src
print(R[R.k > 0].groupby('method').prior_src.agg(lambda s: dict(s.value_counts())).to_dict())
