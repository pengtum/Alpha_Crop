"""(b) 风险画像：P(方法<SRS)、P(<−0.5)、最差国家/单元、国家间 SD；与"纯抽样噪声"的零假设对照。"""
from common_bm import *

R = load()
Phi = stats.norm.cdf

# ---- 抽样噪声：同一候选池（EE、LV 两次重复共用候选池，A4）上两次 SRS 的差，只含抽样随机性
S = R[R.method == 'srs'].pivot_table(index=['target', 'split', 'clf', 'k'], columns='rep', values=['ba_ge30', 'oa'])
noise = []
for metric in ('ba_ge30', 'oa'):
    dd = (S[(metric, 0)] - S[(metric, 1)]).rename('d').reset_index()
    for scope, g in (('EE+LV(同池)', dd[dd.target.isin(['EE', 'LV'])]), ('11 国(池也不同)', dd)):
        for k in (50, 200):
            x = g[g.k == k].d
            sd1 = np.sqrt((x ** 2).mean() / 2)
            noise.append(dict(metric=metric, scope=scope, k=k, n=len(x), sd_one_draw=sd1,
                              sd_diff=np.sqrt(2) * sd1, P0_lt0=0.5, P0_lt_m05=Phi(-0.5 / (np.sqrt(2) * sd1))))
# 方法自身重复之间的差（E1，EE/LV 同池）
for metric in ('ba_ge30', 'oa'):
    for cond in ('AE', 'S2'):
        for m in E1:
            g = R[(R.cond == cond) & (R.method == m) & R.target.isin(['EE', 'LV']) & R.k.isin([50, 200]) & R.clf.isin(['s2', 'ae'])]
            P = g.pivot_table(index=['target', 'split', 'clf', 'k'], columns='rep', values=metric)
            x = (P[0] - P[1])
            noise.append(dict(metric=metric, scope=f'{m}|{cond} EE+LV 重复间', k='50+200', n=len(x), sd_one_draw=np.sqrt((x ** 2).mean() / 2)))
N = pd.DataFrame(noise)
N.to_csv(f'{OUTD}/b_noise.csv', index=False)

sd_null = {(r.metric, r.k): r.sd_diff for r in N[(N.scope == 'EE+LV(同池)')].itertuples()}

rows = []
worst_rows = []
for metric in ('ba_ge30', 'oa'):
    for cond in ('AE', 'S2', 'AE+S2'):
        for m in ALL8:
            if cond == 'AE+S2':
                d = pd.concat([vs_srs(R, c, [m], metric) for c in ('AE', 'S2')], ignore_index=True)
            else:
                d = vs_srs(R, cond, [m], metric)
            v = cv(d)
            # 零假设（真实效应为 0、只有抽样噪声）下的期望比例
            p0 = np.mean([Phi(-0.5 / sd_null[(metric, k)]) for k in d.k])
            # 以国家均值为真值、加抽样噪声时的期望比例（风险中有多少是"系统性"的）
            cm = d.groupby(['target', 'clf', 'k']).d.transform('mean')
            p_sys = np.mean(Phi((-0.5 - cm) / d.k.map(lambda k: sd_null[(metric, k)])))
            w = d.loc[d.d.idxmin()]
            rows.append(dict(metric=metric, cond=cond, method=m, n_runs=len(d),
                             mean=v.mean(), P_lt0=(d.d < 0).mean(), P_lt_m05=(d.d < -0.5).mean(), P_lt_m1=(d.d < -1).mean(),
                             P_gt_p05=(d.d > 0.5).mean(), P0_lt_m05_null=p0, P_lt_m05_given_country_means=p_sys,
                             sd_run=d.d.std(ddof=1), sd_country=v.std(ddof=1),
                             n_country_lt_m05=int((v < -0.5).sum()), n_country_lt_m1=int((v < -1).sum()),
                             n_country_gt_p05=int((v > 0.5).sum()),
                             worst_country=v.idxmin(), worst_country_val=v.min(), best_country=v.idxmax(), best_country_val=v.max(),
                             worst_run=f'{w.target}_s{w.split}_r{w.rep}_{w.cond}_{w.clf}_k{w.k}', worst_run_val=w.d,
                             q05_run=d.d.quantile(0.05), q95_run=d.d.quantile(0.95)))
            if metric == 'ba_ge30' and cond == 'AE+S2':
                worst_rows.append(v.rename(m))
B = pd.DataFrame(rows)
B.to_csv(f'{OUTD}/b_risk.csv', index=False)
CV = pd.concat(worst_rows, axis=1)
CV.to_csv(f'{OUTD}/b_country_values_ba_pooled.csv')

pd.set_option('display.width', 250)
print('=== 抽样噪声 ===')
print(N.round(3).to_string(index=False))
for metric in ('ba_ge30', 'oa'):
    print(f'\n=== 风险画像 {metric} ===')
    X = B[B.metric == metric]
    print(X[['cond', 'method', 'n_runs', 'mean', 'P_lt0', 'P_lt_m05', 'P0_lt_m05_null', 'P_lt_m05_given_country_means', 'P_lt_m1', 'P_gt_p05',
             'sd_run', 'sd_country', 'n_country_lt_m05', 'n_country_lt_m1', 'n_country_gt_p05', 'worst_country', 'worst_country_val',
             'best_country', 'best_country_val', 'worst_run', 'worst_run_val']].round(3).to_string(index=False))
print('\n=== 国家值（ba_ge30，AE+S2 合并）===')
print(CV.round(2).to_string())
