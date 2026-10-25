"""
只读（不训练）：第 0 步 B2 的同 OA 标签价值估计量，AE − S2 配对差（每个特征以自己的 Z0 OA★ 为锚；LightGBM；描述，C 级）。
输入：/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/fa_rows_B2.csv（grid = new）。
运行：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B ae_s2_step0B2_paired.py
"""
import numpy as np, pandas as pd
from scipy import stats
C = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
C9 = [c for c in C if c not in ('RO', 'PT')]
def tsum(v):
    v = pd.Series(v).dropna(); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return dict(n=n, mean=m, lo95=m - h, hi95=m + h, p=2 * stats.t.sf(abs(m / se), n - 1), npos=int((v > 0).sum()))
def fmt(s):
    return f"{s['mean']:+.2f} [{s['lo95']:+.2f}, {s['hi95']:+.2f}]，p={s['p']:.3f}，>0 {s['npos']}/{s['n']}"
f = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/fa_rows_B2.csv')
f = f[f.grid == 'new']
W = f.pivot_table(index=['target', 'split', 'rep', 'clf'], columns='arm', values='FA')
E = pd.DataFrame(index=W.index)
for k in (50, 200, 400):
    E[f'dFA_RT{k}'] = W[f'RT{k}'] - W['Z0']
    E[f'A_{k}'] = W[f'PK{k}'] - W['Z0']
    E[f'R_{k}'] = W[f'RT{k}'] - W[f'PK{k}']
    E[f'dFA_HD{k}'] = W[f'HD{k}'] - W['Z0']
E['FA_Z0'] = W['Z0']
cm = E.groupby(['target', 'clf']).mean().unstack('clf')
print('# 第 0 步 B2：AE − S2 配对差（国家值 = 可用单元平均；正 = AE 更大）')
for q in ['FA_Z0', 'dFA_RT50', 'dFA_RT200', 'dFA_RT400', 'A_50', 'A_200', 'A_400', 'R_50', 'R_200', 'R_400', 'dFA_HD200', 'dFA_HD400']:
    a, s = cm[(q, 'ae')], cm[(q, 's2')]
    d = (a - s).loc[C]
    print(f'  {q:10s} AE 9国 {a.loc[C9].mean():+.2f} / S2 9国 {s.loc[C9].mean():+.2f}；AE−S2 11 国 {fmt(tsum(d))} | 9 国 {fmt(tsum(d.loc[C9]))}')
print('  注：FA_Z0 是各自 OA★ 窗口内的零标签 BA 包络均值，两特征的窗口不同（锚不同），AE−S2 只作参照，不是同 OA 比较。')
