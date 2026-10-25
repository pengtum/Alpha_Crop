"""三种覆盖方案的拟合次数与墙钟估计（只读 bench_*.jsonl）。
单次耗时 = 拟合 + 预测；BE（3.0 万测试地块）代表 BE、DE，LV（1.2 万）代表其余 9 国；k>0 一律用 k=200 实测。
方式 i：2 进程 × 4 线程（4 线程实测）。方式 ii：N 进程 × 1 线程；单次耗时 = 4 线程实测 × BE k=200 上"1 线程（4 路并发）/4 线程"之比。
4×1 为实测配置；8×1 为线性外推（未实测，≤4 线程限制）。"""
import numpy as np, pandas as pd
H = '/Users/perng_mini/AlphaCrops_local/verify/multilearner/feasibility_compute/'
d4 = pd.concat([pd.read_json(H + 'bench_4thr.jsonl', lines=True), pd.read_json(H + 'bench_4thr_extra.jsonl', lines=True)])
d4 = d4[d4.learner != 'lr_csel']
d1 = pd.read_json(H + 'bench_1thr_c4.jsonl', lines=True)
W = {'BE': 2 / 11, 'LV': 9 / 11}
N = {'A': dict(k0=22, kpos=880), 'B': dict(k0=22, kpos=1232), 'C': dict(k0=22, kpos=176)}
ratio_src = {'rf_msl5': 'rf', 'et_msl5': 'et'}

def t4(L, f, k):
    s = d4[(d4.learner == L) & (d4.feat == f) & (d4.k == k)].groupby('tgt').t_total.mean()
    return sum(W[c] * s[c] for c in W) if set(W) <= set(s.index) else np.nan

rows = []
for L in sorted(d4.learner.unique()):
    r = dict(learner=L)
    for f in ('ae', 's2'):
        b4 = d4[(d4.learner == L) & (d4.feat == f) & (d4.k == 200) & (d4.tgt == 'BE')].t_total.mean()
        b1 = d1[(d1.learner == ratio_src.get(L, L)) & (d1.feat == f)].t_total.mean()
        b4src = d4[(d4.learner == ratio_src.get(L, L)) & (d4.feat == f) & (d4.k == 200) & (d4.tgt == 'BE')].t_total.mean()
        r[f'ratio_{f}'] = b1 / b4src
        r[f't4_k0_{f}'], r[f't4_k200_{f}'] = t4(L, f, 0), t4(L, f, 200)
        if np.isnan(r[f't4_k0_{f}']):
            r[f't4_k0_{f}'] = r[f't4_k200_{f}']  # msl5 变体只测了 k=200
    for p, n in N.items():
        s4 = sum(n['k0'] * r[f't4_k0_{f}'] + n['kpos'] * r[f't4_k200_{f}'] for f in ('ae', 's2'))
        s1 = sum((n['k0'] * r[f't4_k0_{f}'] + n['kpos'] * r[f't4_k200_{f}']) * r[f'ratio_{f}'] for f in ('ae', 's2'))
        r[f'n_{p}'] = 2 * (n['k0'] + n['kpos'])
        r[f'{p}_2x4_h'] = s4 / 2 / 3600
        r[f'{p}_4x1_h'] = s1 / 4 / 3600
        r[f'{p}_8x1_h'] = s1 / 8 / 3600
    mem = pd.concat([d4[d4.learner == L].rss_after_pred_mb, d1[d1.learner == L].rss_after_pred_mb]).max()
    r['rss_max_mb'] = mem
    rows.append(r)
T = pd.DataFrame(rows).set_index('learner')
pd.set_option('display.width', 250)
print(T[['t4_k200_ae', 't4_k200_s2', 'ratio_ae', 'ratio_s2', 'rss_max_mb']].round(2).to_string())
print(T[[c for c in T.columns if c.startswith(('A_', 'B_', 'C_', 'n_'))]].round(2).to_string())
T.to_csv(H + 'estimate_plans.csv')
