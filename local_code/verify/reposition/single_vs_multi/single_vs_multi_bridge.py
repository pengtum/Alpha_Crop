# 只读计算：单源两两迁移（pairwise_v2）与多源合并（G k=0/SRS）之间的桥接
# 1) AE 偏移的"标签当量"：S2 的 SRS 曲线要多少个目标标签才追平 AE 的 k=0（线性插值，>400 记为删失）
# 2) 多源 AE−S2 与单源 AE−S2（最近源、近源、全部源）的国家层对应
# 3) 多源 AE−S2 与目标国"孤立度"（最近源纬度差、最近 3 源平均、平均）的关系
import os, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
PW = '/Users/perng_mini/AlphaCrops_local/verify/reposition/overlap_audit/pairwise_with_distance.csv'
CEN = '/Users/perng_mini/AlphaCrops_local/verify/reposition/overlap_audit/country_centroids.csv'
CUR = os.path.join(OUT, 'srs_curve_ae_minus_s2_by_country.csv')

cur = pd.read_csv(CUR)
res = {}

# ---------- 1) 标签当量 ----------
def k_to_reach(ks, ys, target):
    # ys 随 k 不一定单调：取首次达到 target 的插值 k；若 k=0 已达到返回 0；到 400 仍未达到返回 inf
    if ys[0] >= target:
        return 0.0
    for i in range(1, len(ks)):
        if ys[i] >= target:
            k0, k1, y0, y1 = ks[i - 1], ks[i], ys[i - 1], ys[i]
            return float(k0 + (target - y0) / (y1 - y0) * (k1 - k0)) if y1 != y0 else float(k1)
    return float('inf')

eq_rows = []
for (lr, met), g in cur.groupby(['learner', 'metric']):
    for t, gt in g.groupby('target'):
        gt = gt.sort_values('k')
        ks = gt['k'].to_numpy(); s2 = gt['s2'].to_numpy(); ae = gt['ae'].to_numpy()
        eq_rows.append(dict(learner=lr, metric=met, target=t, ae_k0=ae[0], s2_k0=s2[0], s2_k400=s2[-1],
                            k_s2_reach_ae0=k_to_reach(ks, s2, ae[0]),
                            k_ae_reach_s2_400=k_to_reach(ks, ae, s2[-1])))
eq = pd.DataFrame(eq_rows)
eq.round(2).to_csv(os.path.join(OUT, 'label_equivalent_ae_offset.csv'), index=False)
summ = {}
for (lr, met), g in eq.groupby(['learner', 'metric']):
    v = g['k_s2_reach_ae0']
    summ[f'{lr}_{met}'] = dict(
        n_ae_k0_below_s2_k0=int((g.ae_k0 < g.s2_k0).sum()),
        n_censored_gt400=int(np.isinf(v).sum()),
        median_k=float(np.median(v)) if len(v) else None,
        per_country={r.target: (None if np.isinf(r.k_s2_reach_ae0) else round(r.k_s2_reach_ae0, 0)) for r in g.itertuples()},
        # 跨国均值曲线上的当量
        mean_curve=None)
    m = cur[(cur.learner == lr) & (cur.metric == met)].groupby('k')[['ae', 's2']].mean()
    summ[f'{lr}_{met}']['mean_curve'] = dict(
        ae=[round(x, 2) for x in m['ae']], s2=[round(x, 2) for x in m['s2']],
        k_s2_reach_ae0=round(k_to_reach(m.index.to_numpy(), m['s2'].to_numpy(), m['ae'].iloc[0]), 1))
    # 去 RO、PT
    m2 = cur[(cur.learner == lr) & (cur.metric == met) & (~cur.target.isin(['RO', 'PT']))].groupby('k')[['ae', 's2']].mean()
    summ[f'{lr}_{met}']['mean_curve_drop_RO_PT'] = dict(
        ae=[round(x, 2) for x in m2['ae']], s2=[round(x, 2) for x in m2['s2']],
        k_s2_reach_ae0=round(k_to_reach(m2.index.to_numpy(), m2['s2'].to_numpy(), m2['ae'].iloc[0]), 1))
res['label_equivalent'] = summ

# ---------- 2) 单源与多源的对应 ----------
pw = pd.read_csv(PW)
cen = pd.read_csv(CEN).set_index('country')
dcol = 'd_ae_s2' if 'd_ae_s2' in pw.columns else None
assert dcol, pw.columns
multi = cur[(cur.learner == 'lgbm') & (cur.metric == 'ba_ge30')].pivot(index='target', columns='k', values='d')
rows = []
for t in multi.index:
    p = pw[pw.target == t].copy()
    p = p.sort_values('dlat')
    rows.append(dict(target=t,
                     multi_k0=multi.loc[t, 0], multi_k400=multi.loc[t, 400],
                     single_nearest=p['d_ae_s2'].iloc[0],
                     single_near3=p['d_ae_s2'].iloc[:3].mean(),
                     single_lt6=p.loc[p.dlat < 6, 'd_ae_s2'].mean() if (p.dlat < 6).any() else np.nan,
                     single_all=p['d_ae_s2'].mean(),
                     min_dlat=p['dlat'].iloc[0], near3_dlat=p['dlat'].iloc[:3].mean(), mean_dlat=p['dlat'].mean(),
                     n_src_lt6=int((p.dlat < 6).sum())))
br = pd.DataFrame(rows).set_index('target')
br.round(3).to_csv(os.path.join(OUT, 'single_vs_multi_by_target.csv'))
corr = {}
for a in ['single_nearest', 'single_near3', 'single_lt6', 'single_all']:
    for b in ['multi_k0', 'multi_k400']:
        x = br[[a, b]].dropna()
        r, p = stats.spearmanr(x[a], x[b])
        corr[f'{a}~{b}'] = [round(float(r), 3), round(float(p), 4), len(x)]
for a in ['min_dlat', 'near3_dlat', 'mean_dlat']:
    for b in ['multi_k0', 'multi_k400']:
        r, p = stats.spearmanr(br[a], br[b])
        corr[f'{a}~{b}'] = [round(float(r), 3), round(float(p), 4), 11]
res['single_multi_corr'] = corr
res['single_multi_means'] = br[['multi_k0', 'multi_k400', 'single_nearest', 'single_near3', 'single_lt6', 'single_all']].mean().round(3).to_dict()
# 多源 − 单源平均 的配对差（同一目标国）
d = br['multi_k0'] - br['single_all']
res['multi_k0_minus_single_all'] = dict(mean=round(d.mean(), 3), npos=int((d > 0).sum()),
                                         wilcoxon_p=round(float(stats.wilcoxon(d).pvalue), 4))
d2 = br['multi_k0'] - br['single_near3']
res['multi_k0_minus_single_near3'] = dict(mean=round(d2.mean(), 3), npos=int((d2 > 0).sum()),
                                           wilcoxon_p=round(float(stats.wilcoxon(d2).pvalue), 4))

json.dump(res, open(os.path.join(OUT, 'single_vs_multi_bridge.json'), 'w'), indent=1, ensure_ascii=False, default=str)
print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
print(br.round(2).to_string())
