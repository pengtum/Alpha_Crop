"""
只读（不训练）：D1 决策层（类均衡头 λ=0.1）与表征层剩余的 AE − S2 配对差（逐国，国家为单位 t 区间，描述）。
输入：storyline/decision_curve/decision_curve_country.csv、decision_curve_repr_residual_s0r0.csv。
运行：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B ae_s2_decision_layer.py
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
D = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/decision_curve'
cc = pd.read_csv(f'{D}/decision_curve_country.csv')
out = []
print('# D1：AE − S2 配对差（正 = AE 更大）')
for met in ('ba_ge30', 'oa'):
    for k in (50, 200, 400):
        s = cc[(cc.metric == met) & (cc.k == k)].pivot(index='target', columns='clf')
        for q in ('dhead_PT', 'head', 'dfull_PT', 'full'):
            d = (s[(q, 'ae')] - s[(q, 's2')]).loc[C]
            if q in ('dfull_PT', 'full') and k != 50:
                continue
            print(f'  {met:7s} k={k:<3d} {q:9s} 11 国 {fmt(tsum(d))} | 9 国 {fmt(tsum(d.loc[C9]))}')
            if q in ('dfull_PT', 'full', 'head'):
                print(f'        逐国 {d.round(2).to_dict()}')
rr = pd.read_csv(f'{D}/decision_curve_repr_residual_s0r0.csv').pivot(index='target', columns='clf')
print('\n# 表征层剩余（tgt300 − D1 全池头，划分 0 重复 0）与决策层可挽回量（全池头 − k0），AE − S2')
for q in ('dec_ba', 'res_ba', 'dec_oa', 'res_oa', 'gap_ba', 'gap_oa'):
    d = (rr[(q, 'ae')] - rr[(q, 's2')]).loc[C]
    print(f'  {q:7s} AE {rr[(q, "ae")].mean():+.2f} / S2 {rr[(q, "s2")].mean():+.2f}；AE−S2 11 国 {fmt(tsum(d))} | 9 国 {fmt(tsum(d.loc[C9]))}')
    print(f'        逐国 {d.round(2).to_dict()}')
