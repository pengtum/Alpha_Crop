"""
补遗 B2 独立复核的方案外敏感性：共同可达口径（只在 Z0、PK-k、RT-k 三臂都可达的 o 上比较），
检查 PK 臂在窗口顶端的部分可达（OA 上限略低于 OA★，DE/AT/LT）是否影响 A_k、R_k 与读法。
只读 indep_curves.csv（新网格曲线）；包络函数与 indep_B2.py 相同的定义（此处重写一份，不导入主脚本以免加载数据）。
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-9
WOFF = np.arange(-40, 1) / 10.0
NINE = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'SE', 'SI', 'SK']
ALL = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']


def env_one(oa, ba, grid):
    res = np.full(len(grid), np.nan)
    a0, a1, c0, c1 = oa[:-1], oa[1:], ba[:-1], ba[1:]
    lo, hi = np.minimum(a0, a1), np.maximum(a0, a1)
    for gi, o in enumerate(grid):
        s = np.flatnonzero((o >= lo - TOL) & (o <= hi + TOL))
        if len(s) == 0:
            continue
        d = a1[s] - a0[s]
        flat = np.abs(d) <= TOL
        lam = np.clip((o - a0[s]) / np.where(flat, 1.0, d), 0.0, 1.0)
        res[gi] = np.where(flat, np.maximum(c0[s], c1[s]), c0[s] + lam * (c1[s] - c0[s])).max()
    return res


def tci(v):
    v = np.asarray(pd.Series(v, dtype=float).dropna())
    n = len(v)
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return f'{m:+.2f} [{m - h:+.2f}, {m + h:+.2f}] {int((v > 0).sum())}/{n}'


def main():
    C = pd.read_csv(os.path.join(HERE, 'indep_curves.csv'))
    rows = []
    for (tgt, unit, f), g in C.groupby(['target', 'unit', 'clf']):
        cv = {a: (x.sort_values('tau').oa.values, x.sort_values('tau').ba_ge30.values) for a, x in g.groupby('arm')}
        tz = g[(g.arm == 'Z0') & (g.tau == 0.0)]
        grid = float(tz.oa.iloc[0]) + WOFF
        E = {a: env_one(*cv[a], grid) for a in ['Z0', 'PKo'] + [f'{p}{k}' for p in ('PK', 'RT', 'HD') for k in (50, 200, 400)]}
        r = dict(target=tgt, unit=unit, clf=f)
        for k in (50, 200, 400):
            m = np.isfinite(E['Z0']) & np.isfinite(E[f'PK{k}']) & np.isfinite(E[f'RT{k}'])
            cov = m.mean()
            ok = cov >= 0.5
            r[f'cov{k}'] = cov
            r[f'A_{k}'] = np.mean(E[f'PK{k}'][m] - E['Z0'][m]) if ok else np.nan
            r[f'R_{k}'] = np.mean(E[f'RT{k}'][m] - E[f'PK{k}'][m]) if ok else np.nan
            r[f'dRT_{k}'] = np.mean(E[f'RT{k}'][m] - E['Z0'][m]) if ok else np.nan
        rows.append(r)
    U = pd.DataFrame(rows)
    U.to_csv(os.path.join(HERE, 'indep_commonreach_units.csv'), index=False)
    out = ['补遗 B2 独立复核 —— 方案外敏感性：共同可达口径（Z0、PK-k、RT-k 三臂都可达的 o；共同可达 <50% 记缺失）', '']
    for k in (50, 200, 400):
        out.append(f'k={k}：共同可达 <1 的单元 × 特征 {int((U[f"cov{k}"] < 1).sum())}，<0.5 的 {int((U[f"cov{k}"] < 0.5).sum())}')
        for col in (f'A_{k}', f'R_{k}', f'dRT_{k}'):
            cvf = {f: U[U.clf == f].groupby('target')[col].mean() for f in ('ae', 's2')}
            both = pd.concat([cvf['ae'], cvf['s2']], axis=1).mean(axis=1, skipna=False)
            out.append(f'  {col:7s} both 9 国 {tci(both.reindex(NINE))}   11 国 {tci(both.reindex(ALL))}')
        a9 = pd.concat([U[U.clf == f].groupby('target')[f'A_{k}'].mean() for f in ('ae', 's2')], axis=1).mean(axis=1).reindex(NINE)
        d9 = pd.concat([U[U.clf == f].groupby('target')[f'dRT_{k}'].mean() for f in ('ae', 's2')], axis=1).mean(axis=1).reindex(NINE)
        out.append(f'  A_{k} 占比（both 9 国）= {a9.mean():.3f}/{d9.mean():.3f} = {a9.mean() / d9.mean():.3f}')
    open(os.path.join(HERE, 'indep_commonreach.txt'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
