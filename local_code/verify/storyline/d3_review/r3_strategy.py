"""D3 复核：H4 predstrat 相对 SRS 与 Holm p；预测通路 AE−S2；几何法 F 检验；kcenter OA。只读 G rows。"""
import glob
import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
from scipy import stats

GD = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(GD, 'rows_t*_*.csv')))], ignore_index=True)
UNIT = ['target', 'split', 'rep']
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
M8 = ['badge_c', 'gist_margin', 'kcenter', 'kherding', 'lpm2_nogeo', 'margin', 'predstrat_srs', 'typiclust']


def tsum(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return m, m - q * se, m + q * se, 2 * stats.t.sf(abs(m / se), n - 1), int((v > 0).sum())


def holm(p):
    p = pd.Series(p); o = p.sort_values(); m = len(p)
    vals = np.maximum.accumulate([(m - i) * pv for i, pv in enumerate(o.to_numpy())])
    return pd.Series(np.minimum(vals, 1), index=o.index).reindex(p.index)


def vs_srs(cond, methods, metric='ba_ge30', ks=(50, 200)):
    sub = R[R.k.isin(ks) & R.clf.isin(['ae', 's2'])]
    a = sub[(sub.cond == cond) & sub.method.isin(methods)].copy()
    s = sub[sub.method == 'srs'].set_index(UNIT + ['clf', 'k'])[metric]
    a['d'] = a[metric].to_numpy() - s.reindex(pd.MultiIndex.from_frame(a[UNIT + ['clf', 'k']])).to_numpy()
    return a.dropna(subset=['d'])


def cv(d):
    return d.groupby(['target', 'method', 'clf', 'k']).d.mean().groupby('target').mean()


P = {}
for cond in ('AE', 'S2'):
    for m in M8:
        d = vs_srs(cond, [m])
        r = tsum(cv(d))
        P[f'{m}|{cond}'] = r[3]
        if m == 'predstrat_srs':
            print(f'predstrat|{cond}: {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}] p={r[3]:.4f} {r[4]}/11; reps={sorted(d.rep.unique())}')
H = holm(P)
print('Holm 16 项（最小 5 项）:', H.sort_values().head(5).round(4).to_dict())
print('predstrat Holm:', H['predstrat_srs|AE'].round(4), H['predstrat_srs|S2'].round(4))
# 合并（两种指导平均）
dA = cv(vs_srs('AE', ['predstrat_srs'])); dS = cv(vs_srs('S2', ['predstrat_srs']))
print('predstrat pooled', ['%.4f' % x for x in tsum((dA + dS) / 2)])
# 几何法相对 SRS
for cond in ('AE', 'S2'):
    print('geom vs SRS', cond, ['%.4f' % x for x in tsum(cv(vs_srs(cond, E1)))])
# kcenter OA（两种指导合并？）
for cond in ('AE', 'S2'):
    print('kcenter OA', cond, ['%.4f' % x for x in tsum(cv(vs_srs(cond, ['kcenter'], 'oa')))])
kc = (cv(vs_srs('AE', ['kcenter'], 'oa')) + cv(vs_srs('S2', ['kcenter'], 'oa'))) / 2
print('kcenter OA pooled', ['%.4f' % x for x in tsum(kc)], kc.round(2).to_dict())
# 4 几何法之间的 F（单元值，方法为因素？）——复现两种单元集合
for reps, lab in (([0], 'rep0'), ([0, 1], 'all')):
    groups = []
    for m in E1:
        d = pd.concat([vs_srs(c, [m]) for c in ('AE', 'S2')])
        d = d[d.rep.isin(reps)]
        u = d.groupby(UNIT).d.mean()
        groups.append(u.to_numpy())
    F, p = stats.f_oneway(*groups)
    print(f'4 几何法 单因素 F（{lab}）：F={F:.3f} p={p:.3f} n/组={len(groups[0])}')
