"""由实测耗时估算三种覆盖方案的拟合次数与墙钟时间（只读 bench_*.jsonl）。
每次拟合耗时 = 拟合 + 预测；按测试集大小加权：BE（3.0 万测试地块）代表 BE、DE 两国，LV（1.2 万）代表其余 9 国。
k=0 用 k=0 实测，k>0（50–400）一律用 k=200 实测。"""
import json
import numpy as np
import pandas as pd

H = '/Users/perng_mini/AlphaCrops_local/verify/multilearner/feasibility_compute/'
d4 = pd.read_json(H + 'bench_4thr.jsonl', lines=True)
d4 = d4[d4.get('error').isna()] if 'error' in d4 else d4
W = {'BE': 2 / 11, 'LV': 9 / 11}

# 每种特征的拟合次数（两种特征各一半）：k0 = 22（国家×划分，两次重复共用）
N = {'A': dict(k0=22, kpos=704 + 176), 'B': dict(k0=22, kpos=1232), 'C': dict(k0=22, kpos=176)}


def per_fit(df, learner, feat, kk):
    s = df[(df.learner == learner) & (df.feat == feat) & (df.k == kk)].set_index('tgt').t_total
    if not set(W) <= set(s.index):
        return np.nan
    return sum(W[c] * s[c] for c in W)


def table(df, label):
    rows = []
    for L in sorted(df.learner.unique()):
        r = dict(learner=L, threads=label)
        for f in ('ae', 's2'):
            r[f't0_{f}'] = per_fit(df, L, f, 0)
            r[f't200_{f}'] = per_fit(df, L, f, 200)
        for plan, n in N.items():
            r[f'n_{plan}'] = 2 * (n['k0'] + n['kpos'])
            r[f'cpu_s_{plan}'] = sum(n['k0'] * r[f't0_{f}'] + n['kpos'] * r[f't200_{f}'] for f in ('ae', 's2'))
        rows.append(r)
    return pd.DataFrame(rows)


T4 = table(d4, '4thr')
# 墙钟：方式 i = 2 个进程 × 4 线程（8 核）；每进程的串行时间 = 总耗时 / 2
for plan in N:
    T4[f'wall_h_{plan}_2x4'] = T4[f'cpu_s_{plan}'] / 2 / 3600
pd.set_option('display.width', 250)
print(T4.round(2).to_string())
T4.to_csv(H + 'estimate_4thr.csv', index=False)
try:
    d1 = pd.read_json(H + 'bench_1thr_c4.jsonl', lines=True)
    s1 = d1.groupby(['learner', 'feat']).t_total.median().unstack()
    s4 = d4[(d4.tgt == 'BE') & (d4.k == 200)].groupby(['learner', 'feat']).t_total.median().unstack()
    R = (s1 / s4).rename(columns=lambda c: f'ratio_1thr_c4_over_4thr_{c}')
    print('\nBE k=200：1 线程（4 路并发）耗时 / 4 线程耗时\n', pd.concat([s1.add_prefix('t1c4_'), s4.add_prefix('t4_'), R], axis=1).round(2).to_string())
except FileNotFoundError:
    pass
