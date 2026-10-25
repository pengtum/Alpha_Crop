"""
只读（不训练）：为确认运行中 AE–S2 假设估计国家层 SD 与 9 国 / 11 国 t 检验的最小可检效应（MDE，80% 功效，双侧 α=0.05）。
输入：/private/tmp/ae_s2_evidence/matched_oa_units.csv（同 OA 的 Z0 AE−S2）；/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/fa_rows_B2.csv（B2 的 ΔFA、A、R）。
注意：G s0/s1 已复用，SD 只作功效规划，属 C 级。确认运行在 F1/F2 两半上平均，单元内噪声会变，国家间异质性不变。
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np, pandas as pd
from scipy import stats
C = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
C9 = [c for c in C if c not in ('RO', 'PT')]
def mde(sd, n, alpha=0.05, power=0.8):
    df = n - 1
    # 非中心 t 近似：MDE ≈ (t_{1-α/2} + t_{power}) * sd / sqrt(n)
    return (stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * sd / np.sqrt(n)
def tost_halfwidth(sd, n):
    return stats.t.ppf(0.95, n - 1) * sd / np.sqrt(n)
u = pd.read_csv('/private/tmp/ae_s2_evidence/matched_oa_units.csv')
print('matched_oa_units 列：', list(u.columns)[:20])
col = [c for c in u.columns if c.startswith('W4_min')]
print('W4 列：', col)
g = u.groupby('target')['W4_min_d'].mean()
out = {}
for nm, cs in (('11', C), ('9', C9)):
    v = g.loc[cs]; out[nm] = (v.mean(), v.std(ddof=1))
    print(f'H_R1 同 OA Z0 AE−S2（W4_min）{nm} 国：均值 {v.mean():+.2f}，SD {v.std(ddof=1):.2f}，MDE80 {mde(v.std(ddof=1), len(v)):.2f}，95% 下限预期 {v.mean() - stats.t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)):+.2f}')
# 由已报 95% CI 反推 SD（9 国，df=8）：sd = halfwidth * 3 / t_{0.975,8}
t8 = stats.t.ppf(0.975, 8)
rows = {
    'ΔFA(RT200) AE−S2（B2，9 国）': (+0.59, (1.40 + 0.21) / 2),
    'A_200 AE−S2（B2，9 国）': (+0.23, (1.31 + 0.85) / 2),
    'R_200 AE−S2（B2，9 国）': (+0.37, (1.14 + 0.41) / 2),
    'Δhead BA k=200 AE−S2（D1，9 国）': (+0.05, (1.72 + 1.61) / 2),
    'BA 缺口 tgt300−k0 AE−S2（9 国）': (-0.98, (2.13 + 4.09) / 2),
    '[PK50−PEM] OA AE−S2（检验 3，9 国）': (+1.11, (2.40 + 0.18) / 2),
}
for k, (m, hw) in rows.items():
    sd = hw * 3 / t8
    tw = tost_halfwidth(sd, 9)
    print(f'{k}：点估计 {m:+.2f}，反推 SD {sd:.2f}，MDE80 {mde(sd, 9):.2f}，TOST 90% 半宽 {tw:.2f} → 若真值=0，±1 等价需 半宽<1：{"可" if tw < 1 else "否"}；以当前点估计，能通过的最小界 {abs(m) + tw:.2f}')
