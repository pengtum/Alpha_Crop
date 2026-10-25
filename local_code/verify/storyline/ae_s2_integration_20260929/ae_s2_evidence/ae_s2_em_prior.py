"""
只读（不训练）：检验 3 中不用标签的 EM 先验估计（PEM）与真实先验（PO）、50 标签先验（PK50）对 OA 的作用，AE − S2 配对差（逐国）。
恢复率 = (arm − L0)/(PO − L0)。只作描述；check3 为随机 70/30 无缓冲、早停设置。
运行：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B ae_s2_em_prior.py
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
d = pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/check3_arms.csv')
d = d[d.split == 'random']
W = d.pivot_table(index=['target', 'feat'], columns='arm', values='oa')
pk = W[[c for c in W.columns if c.startswith('PK50_s')]].mean(1)
G = pd.DataFrame({'PO_gain': W.PO - W.L0, 'PEM_gain': W.PEM - W.L0, 'PK50_gain': pk - W.L0})
G = G.unstack('feat')
print('# 检验 3（随机 70/30，早停）OA 增益（相对 L0），AE 与 S2 及配对差')
for q in ('PO_gain', 'PEM_gain', 'PK50_gain'):
    a, s = G[(q, 'ae')].loc[C], G[(q, 's2')].loc[C]
    print(f'  {q:9s} AE {a.mean():+.2f} / S2 {s.mean():+.2f}；AE−S2 11 国 {fmt(tsum(a - s))} | 9 国 {fmt(tsum((a - s).loc[C9]))}')
    print(f'        逐国 AE−S2 {(a - s).round(2).to_dict()}')
for f in ('ae', 's2'):
    rec = G[('PEM_gain', f)].sum() / G[('PO_gain', f)].sum()
    rec9 = G[('PEM_gain', f)].loc[C9].sum() / G[('PO_gain', f)].loc[C9].sum()
    print(f'  PEM 合并恢复率 {f}: 11 国 {100*rec:.1f}%（NUMBERS 对应 H5.pem_recovery），9 国 {100*rec9:.1f}%')
