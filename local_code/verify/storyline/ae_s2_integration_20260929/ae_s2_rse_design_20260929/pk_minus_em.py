"""
只读（不训练）：先验通道的"表征 × 标签"交互。
量：50 个随机标签估先验（PK50）相对不用标签的 EM 先验（PEM）的 OA 增益，AE − S2 配对差（逐国）。
即 [PK50 − PEM]_AE − [PK50 − PEM]_S2；正值 = 在 AE 下 50 个标签对先验通道更"必要"。
另报 PK50/PEM 相对 PO 的逐国恢复率。数据：results/check3_arms.csv，split=random（检验 3，随机 70/30 无缓冲，早停）。C 级描述。
运行：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B pk_minus_em.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np, pandas as pd
from scipy import stats
C = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
C9 = [c for c in C if c not in ('RO', 'PT')]
def tsum(v):
    v = pd.Series(v).dropna(); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return dict(n=n, mean=m, lo95=m - h, hi95=m + h, p=2 * stats.t.sf(abs(m / se), n - 1), npos=int((v > 0).sum()), sd=v.std(ddof=1))
def fmt(s):
    return f"{s['mean']:+.2f} [{s['lo95']:+.2f}, {s['hi95']:+.2f}]，p={s['p']:.3f}，>0 {s['npos']}/{s['n']}，SD={s['sd']:.2f}"
d = pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/check3_arms.csv')
d = d[d.split == 'random']
W = d.pivot_table(index=['target', 'feat'], columns='arm', values='oa')
pk = W[[c for c in W.columns if c.startswith('PK50_s')]].mean(1)
X = pd.DataFrame({'pk_minus_em': pk - W.PEM, 'rec_pk': (pk - W.L0) / (W.PO - W.L0), 'rec_em': (W.PEM - W.L0) / (W.PO - W.L0)}).unstack('feat')
print('# [PK50 − PEM] 的 OA，AE 与 S2 及配对差（检验 3 random，早停）')
a, s = X[('pk_minus_em', 'ae')].loc[C], X[('pk_minus_em', 's2')].loc[C]
print(f'  AE 均值 {a.mean():+.2f} / S2 {s.mean():+.2f}')
print(f'  AE−S2 11 国 {fmt(tsum(a - s))}')
print(f'  AE−S2  9 国 {fmt(tsum((a - s).loc[C9]))}')
print(f'  逐国 AE−S2 {(a - s).round(2).to_dict()}')
print('# 逐国恢复率（中位数）')
for f in ('ae', 's2'):
    print(f"  {f}: PK50 中位 {100*X[('rec_pk', f)].loc[C].median():.1f}%，EM 中位 {100*X[('rec_em', f)].loc[C].median():.1f}%；9 国 PK50 {100*X[('rec_pk', f)].loc[C9].median():.1f}%，EM {100*X[('rec_em', f)].loc[C9].median():.1f}%")
