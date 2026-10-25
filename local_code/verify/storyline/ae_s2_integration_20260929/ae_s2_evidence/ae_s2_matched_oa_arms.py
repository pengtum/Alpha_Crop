"""
只读（不训练）：同 OA 下 AE − S2 的 BA 差，扩展到第 0 步 B2 的其他臂（RT-k：SRS k 个标签并入重训；HD-k：类均衡决策头）。
窗口 W4_min = [min(两特征该臂 τ=0 的 OA) − 4, min(...)]。LightGBM，44 单元，国家均值，t 区间。
运行：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B ae_s2_matched_oa_arms.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from ae_s2_matched_oa import ba_at, tsum, fmt, C, C9, L

cur = pd.read_csv(f'{L}/verify/step0/partB2/curves_B2.csv')
out = []
for arm in ['Z0', 'RT50', 'RT200', 'RT400', 'HD50', 'HD200', 'HD400']:
    z = cur[cur.arm == arm].sort_values(['unit', 'clf', 'tau'])
    rows = []
    for unit, g in z.groupby('unit'):
        a = g[g.clf == 'ae']; s = g[g.clf == 's2']
        t0a = a[np.isclose(a.tau, 0)].iloc[0]; t0s = s[np.isclose(s.tau, 0)].iloc[0]
        anchor = min(t0a.oa, t0s.oa)
        grid = np.round(np.arange(anchor - 4, anchor + 1e-9, 0.1), 6)
        ba_a = ba_at(a.oa.to_numpy(), a.ba_ge30.to_numpy(), grid)
        ba_s = ba_at(s.oa.to_numpy(), s.ba_ge30.to_numpy(), grid)
        ok = np.isfinite(ba_a) & np.isfinite(ba_s)
        rows.append(dict(target=g.target.iloc[0], d_tau0_ba=t0a.ba_ge30 - t0s.ba_ge30, d_tau0_oa=t0a.oa - t0s.oa,
                         d_matched=float(np.mean(ba_a[ok] - ba_s[ok])) if ok.mean() >= 0.5 else np.nan))
    U = pd.DataFrame(rows).groupby('target').mean().loc[C]
    for col in ['d_tau0_ba', 'd_tau0_oa', 'd_matched']:
        s11, s9 = tsum(U[col]), tsum(U.loc[C9, col])
        out.append(dict(arm=arm, quantity=col, **{f'{k}_11': v for k, v in s11.items()}, **{f'{k}_9': v for k, v in s9.items()}))
        print(f'{arm:6s} {col:10s} 11 国 {fmt(s11)} | 9 国 {fmt(s9)}')
    print(f'       逐国 d_matched：{U.d_matched.round(2).to_dict()}')
pd.DataFrame(out).round(4).to_csv('matched_oa_by_arm.csv', index=False)
