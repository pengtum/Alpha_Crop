import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT, 'calib_rows.csv')).rename(columns={'head': 'hd'})
NOPT = ['RO', 'PT']


def cmean(d, cols=('ba', 'oa')):
    """先在国家内平均（单元、分类器、种子），再 11 国等权；返回均值、95% CI（国家层 t）、正向国家数。"""
    c = d.groupby('target')[list(cols)].mean()
    out = {}
    for col in cols:
        v = c[col].dropna()
        m = v.mean()
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        out[col] = (m, m - h, m + h, int((v > 0).sum()), len(v))
    return out, c


def fmt(o):
    return '  '.join(f'{k} {m:+.2f} [{lo:+.2f},{hi:+.2f}] {p}/{n}' for k, (m, lo, hi, p, n) in o.items())


lines = []
P = lines.append

# ---------- 绝对水平
P('== 绝对水平（11 国等权，单元与两种分类器平均）==')
lv = df[df.hd.isin(['k0', 'retrain', 'bal_l0.1', 'nat_l0.1', 'bal+prior', 'prior_only', 'retrain_deskew',
                      'retrain_i200', 'bal_full', 'nat_full'])]
tab = lv.groupby(['design', 'k', 'hd']).apply(lambda g: pd.Series(
    {'ba': g.groupby('target').ba.mean().mean(), 'oa': g.groupby('target').oa.mean().mean(),
     'ba_noROPT': g[~g.target.isin(NOPT)].groupby('target').ba.mean().mean(),
     'oa_noROPT': g[~g.target.isin(NOPT)].groupby('target').oa.mean().mean(), 'n_ctry': g.target.nunique()}))
P(tab.round(2).to_string())

# ---------- 配对差：相对 G 的 SRS 重训（同单元、同分类器、同 k；SRS 选样相同）
P('\n== 配对差：冻结外国模型 + 决策层（同一批 SRS 标签） − G 重训 SRS（r=50, 300 轮）==')
key = ['unit', 'target', 'clf', 'k']
ref = df[(df.design == 'srs_G') & (df.hd == 'retrain')].set_index(key)[['ba', 'oa']]
for head in ['bal_l0.02', 'bal_l0.1', 'bal_l0.5', 'nat_l0.1', 'bal+prior', 'prior_only', 'retrain_deskew',
             'retrain_i200']:
    x = df[(df.design == 'srs_G') & (df.hd == head)].set_index(key)[['ba', 'oa']]
    j = (x - ref).dropna().reset_index()
    for k in (50, 100, 200, 400):
        jj = j[j.k == k]
        o, c = cmean(jj)
        o2, _ = cmean(jj[~jj.target.isin(NOPT)])
        P(f'{head:15s} k={k:3d}: {fmt(o)}   | 去 RO/PT: ba {o2["ba"][0]:+.2f} oa {o2["oa"][0]:+.2f}')

P('\n== 逐国：bal_l0.1 − 重训 SRS（k=200，ba / oa）==')
x = df[(df.design == 'srs_G') & (df.hd == 'bal_l0.1')].set_index(key)[['ba', 'oa']]
j = (x - ref).dropna().reset_index()
P(j[j.k == 200].groupby('target')[['ba', 'oa']].mean().round(2).T.to_string())
x = df[(df.design == 'srs_G') & (df.hd == 'retrain_deskew')].set_index(key)[['ba', 'oa']]
j = (x - ref).dropna().reset_index()
P('retrain_deskew − retrain（k=200）')
P(j[j.k == 200].groupby('target')[['ba', 'oa']].mean().round(2).T.to_string())

# ---------- 相对 k0
P('\n== 相对 k=0（外国模型）的增益：重训 SRS vs 冻结 + 决策层 ==')
k0 = df[df.hd == 'k0'].set_index(['unit', 'target', 'clf'])[['ba', 'oa']]
for head in ['retrain', 'bal_l0.1', 'nat_l0.1', 'bal+prior', 'retrain_deskew']:
    x = df[(df.design == 'srs_G') & (df.hd == head)].set_index(['unit', 'target', 'clf', 'k'])[['ba', 'oa']]
    x = x.reset_index().set_index(['unit', 'target', 'clf'])
    j = x[['ba', 'oa']] - k0.reindex(x.index)[['ba', 'oa']]
    j['k'] = x['k']
    j = j.reset_index()
    s = []
    for k in (50, 100, 200, 400):
        o, _ = cmean(j[j.k == k])
        o2, _ = cmean(j[(j.k == k) & ~j.target.isin(NOPT)])
        s.append(f'k{k}: ba {o["ba"][0]:+.2f} oa {o["oa"][0]:+.2f} (去RO/PT ba {o2["ba"][0]:+.2f} oa {o2["oa"][0]:+.2f})')
    P(f'{head:15s} ' + ' | '.join(s))

# ---------- 设计效应：在同一决策层下，哪种选样更好（只有重复 0）
P('\n== 设计 × 标签用法（重复 0 的 22 个单元；与同单元 SRS_G 比较）==')
key0 = ['unit', 'target', 'clf', 'k']
d0 = df[df.rep == 0]
for head in ['retrain', 'retrain_deskew', 'bal_l0.1', 'nat_l0.1', 'bal+prior']:
    ref0 = d0[(d0.design == 'srs_G') & (d0.hd == head)].set_index(key0)[['ba', 'oa']]
    for des in ['predstrat_AE', 'predstrat_S2', 'oracle_sqrt', 'oracle_equal', 'srs_extra']:
        x = d0[(d0.design == des) & (d0.hd == head)]
        if x.empty:
            continue
        x = x.groupby(key0)[['ba', 'oa']].mean()
        j = (x - ref0).dropna().reset_index()
        s = []
        for k in (50, 200):
            o, _ = cmean(j[j.k == k])
            s.append(f'k{k}: {fmt(o)}')
        P(f'{head:15s} {des:13s} ' + ' || '.join(s))

# ---------- 全池上界
P('\n== 决策层上界（全池 3 万标签拟合温度+偏置；冻结外国模型）相对 k0 ==')
for head in ['bal_full', 'nat_full']:
    x = df[(df.design == 'allpool') & (df.hd == head)].set_index(['unit', 'target', 'clf'])[['ba', 'oa']]
    j = (x - k0.reindex(x.index)).dropna().reset_index()
    o, _ = cmean(j)
    o2, _ = cmean(j[~j.target.isin(NOPT)])
    P(f'{head}: {fmt(o)} | 去 RO/PT ba {o2["ba"][0]:+.2f} oa {o2["oa"][0]:+.2f}')

# ---------- 前沿：同一组 SRS 标签下，bal 与 nat 两个头的 (ΔOA, ΔBA)
P('\n== 同一组 k=200 SRS 标签的三个头 vs 重训（ΔBA, ΔOA，相对重训）见上；λ 敏感性见上 ==')

open(os.path.join(OUT, 'calib_report.txt'), 'w').write('\n'.join(lines))
print('\n'.join(lines))
