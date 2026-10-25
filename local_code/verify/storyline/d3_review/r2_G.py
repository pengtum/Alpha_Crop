"""D3 复核：从 G 逐行结果独立重算抽样行与若干 headline 行。
只读 results/guidance_G/rows_t*_*.csv（不读 guidance_G_ml）。不 import numbers_table。"""
import glob
import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
from scipy import stats

GD = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
assert 'guidance_G_ml' not in GD
fs = sorted(glob.glob(os.path.join(GD, 'rows_t*_*.csv')))
assert all('guidance_G_ml' not in f for f in fs)
R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
UNIT = ['target', 'split', 'rep']
DROP = ['RO', 'PT']


def tsum(v):
    v = np.asarray(v, float)
    n = len(v)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
    q95, q90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    return dict(n=n, m=m, lo95=m - q95 * se, hi95=m + q95 * se, lo90=m - q90 * se, hi90=m + q90 * se,
                p=2 * stats.t.sf(abs(m / se), n - 1), npos=int((v > 0).sum()), se=se)


def show(name, v):
    s = tsum(v)
    s2 = tsum(v.drop([c for c in DROP if c in v.index]))
    print(f'{name}: {s["m"]:+.4f} [{s["lo95"]:+.4f}, {s["hi95"]:+.4f}] 90%[{s["lo90"]:+.4f}, {s["hi90"]:+.4f}] '
          f'p={s["p"]:.4f} {s["npos"]}/{s["n"]} | 去RO/PT {s2["m"]:+.4f} [{s2["lo95"]:+.4f}, {s2["hi95"]:+.4f}] '
          f'p={s2["p"]:.4f} {s2["npos"]}/{s2["n"]}')
    return s


# ---------- θ_E1（AE 指导 − S2 指导，E1 方法，k∈{50,200}，clf∈{ae,s2}）
def theta(metric='ba_ge30', ks=(50, 200), R=R):
    sub = R[R.method.isin(E1) & R.k.isin(ks) & R.clf.isin(['ae', 's2'])]
    key = UNIT + ['method', 'clf', 'k']
    a = sub[sub.cond == 'AE'].set_index(key)[metric]
    b = sub[sub.cond == 'S2'].set_index(key)[metric]
    d = (a - b).dropna().rename('d').reset_index()
    assert len(d) == 44 * 4 * 2 * len(ks), len(d)
    return d.groupby(['target', 'method', 'clf', 'k']).d.mean().groupby('target').mean()


vE1 = theta()
s = show('theta_E1 ba_ge30', vE1)
n = s['n']
se = s['se']
p_tost = max(stats.t.sf((s['m'] + 0.3) / se, n - 1), stats.t.sf((0.3 - s['m']) / se, n - 1))
print(f'  TOST p={p_tost:.4f}; 最小可通过界值={max(abs(s["lo90"]), abs(s["hi90"])):.4f}')
loo = [vE1.drop(c).mean() for c in vE1.index]
print(f'  LOO 均值范围 {min(loo):+.4f} ({vE1.index[int(np.argmin(loo))]}) 至 {max(loo):+.4f} ({vE1.index[int(np.argmax(loo))]})')
loo_eq = 0
for c in vE1.index:
    t = tsum(vE1.drop(c))
    loo_eq += (t['lo90'] > -0.3 and t['hi90'] < 0.3 and t['lo95'] < 0 < t['hi95'])
print(f'  LOO 判等价 {loo_eq}/11')
show('theta_E1 oa', theta('oa'))

# ---------- SRS 学习曲线（22 条）
srs = R[R.method.isin(['srs', 'foreign_only']) & R.clf.isin(['ae', 's2'])]
cur = srs.groupby(['target', 'clf', 'k']).ba_ge30.mean().unstack('k')
g400 = cur[400] - cur[0]
g50 = cur[50] - cur[0]
print(f'SRS k0→400 中位数 {g400.median():+.4f}，均值 {g400.mean():+.4f}；去RO/PT 中位数 '
      f'{g400.drop(DROP, level=0).median():+.4f}（n={len(g400.drop(DROP, level=0))}）')
print(f'SRS k0→50 中位数 {g50.median():+.4f}')
nonmono = sum(((cur.loc[i, [0, 50, 100, 200, 400]].diff().dropna() < 0).any()) for i in cur.index)
print(f'  不单调曲线 {nonmono}/{len(cur)}')

# ---------- B2：G(L,k)，单元值 → 国家值
def unit_curve(metric):
    u = srs.groupby(UNIT + ['clf', 'k'])[metric].mean().unstack('k')
    return u


for metric in ['ba_ge30']:
    u = unit_curve(metric)
    for k in (200,):
        g = (u[k] - u[0])
        avg = g.groupby(UNIT).mean().groupby('target').mean()
        show(f'B2 G(LGB,{k}) avg {metric}', avg)
        for c in ('ae', 's2'):
            show(f'B2 G(LGB,{k}) {c} {metric}', g.xs(c, level='clf').groupby('target').mean())

# ---------- B1：Δ_feat
for metric in ['ba_ge30', 'oa']:
    u = unit_curve(metric)
    df = (u.xs('ae', level='clf') - u.xs('s2', level='clf'))
    for k in (0, 50, 100, 200, 400):
        v = df[k].groupby('target').mean()
        show(f'Δ_feat {metric} k={k}', v)
        if k in (0, 400):
            print('     负的国家', sorted(v[v < 0].index))
    ch = (df[400] - df[0]).groupby('target').mean()
    show(f'Δ_feat 变化量 {metric} (400−0)', ch)
    allneg = [c for c in df.index.get_level_values(0).unique()
              if all(df[k].groupby('target').mean()[c] < 0 for k in (0, 50, 100, 200, 400))]
    print('     五档都为负:', allneg)
# 检查 k=0 两个重复是否相同
f0 = R[R.method == 'foreign_only'].pivot_table(index=['target', 'split', 'clf'], columns='rep', values='ba_ge30')
print('k=0 两重复最大差', float((f0[0] - f0[1]).abs().max()))
