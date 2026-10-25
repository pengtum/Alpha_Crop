# 只读计算：多源合并训练下，AE 分类器 − S2 分类器 随目标标签数 k 的变化
# 数据：G 的 foreign_only (k=0) 与 SRS (k=50/100/200/400) 行（LightGBM, r=50, 300 轮）
# 另：LR（guidance_G_ml，多学习器中期结果，只读 SRS/k0 行，不涉及 θ_E1）
import os, glob, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
ML = '/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml'
CEN = '/Users/perng_mini/AlphaCrops_local/verify/reposition/overlap_audit/multisource_k0_by_target.csv'

def load(pattern):
    fs = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)

def srs_rows(d):
    s = d[d.method.isin(['foreign_only', 'srs'])].copy()
    return s

def country_delta(s, metric):
    # 每 (target, split, rep, k) 的 AE − S2；k=0 在两个 rep 间共用，按 (target, split) 去重
    p = s.pivot_table(index=['target', 'split', 'rep', 'k'], columns='clf', values=metric, aggfunc='mean')
    p['d'] = p['ae'] - p['s2']
    p = p.reset_index()
    k0 = p[p.k == 0].drop_duplicates(['target', 'split', 'k'])
    p = pd.concat([k0, p[p.k > 0]])
    c = p.groupby(['target', 'k'])[['ae', 's2', 'd']].mean().reset_index()
    return c

def tsum(x):
    x = np.asarray(x, float)
    n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    tq = stats.t.ppf(0.975, n - 1)
    p = stats.ttest_1samp(x, 0).pvalue
    return dict(n=n, mean=round(m, 3), lo95=round(m - tq * se, 3), hi95=round(m + tq * se, 3),
                p=round(float(p), 4), npos=int((x > 0).sum()), median=round(float(np.median(x)), 3))

cen = pd.read_csv(CEN).set_index('target')
res = {}
tables = []
for name, pat in [('lgbm', f'{G}/rows_t*.csv'), ('lr', f'{ML}/rows_lr_*.csv')]:
    d = load(pat)
    s = srs_rows(d)
    res[name] = {'n_rows_srs_k0': int(len(s))}
    for metric in ['ba_ge30', 'oa']:
        c = country_delta(s, metric)
        c['learner'] = name; c['metric'] = metric
        tables.append(c)
        out = {}
        for k, g in c.groupby('k'):
            g = g.set_index('target')
            r = tsum(g['d'])
            r['mean_ae'] = round(g['ae'].mean(), 2); r['mean_s2'] = round(g['s2'].mean(), 2)
            # 去 RO、PT
            r['drop_RO_PT'] = tsum(g.drop(index=[t for t in ['RO', 'PT'] if t in g.index])['d'])
            # 与"到外国池平均纬度差"的 Spearman（国家层，n=11，朴素 p）
            gg = g.join(cen[['mean_dlat_to_pool', 'mean_km_to_pool']])
            rho, pr = stats.spearmanr(gg['d'], gg['mean_dlat_to_pool'])
            rho2, pr2 = stats.spearmanr(gg['d'], gg['mean_km_to_pool'])
            r['spearman_vs_mean_dlat'] = [round(float(rho), 3), round(float(pr), 4)]
            r['spearman_vs_mean_km'] = [round(float(rho2), 3), round(float(pr2), 4)]
            r['per_country'] = {t: round(v, 2) for t, v in g['d'].items()}
            out[int(k)] = r
        # k=0→400 的变化：d(400) − d(0)
        w = c.pivot(index='target', columns='k', values='d')
        out['change_0_to_400'] = tsum(w[400] - w[0])
        out['change_0_to_200'] = tsum(w[200] - w[0])
        # 学习曲线增益：AE 与 S2 各自 k0→400
        a = c.pivot(index='target', columns='k', values='ae'); b = c.pivot(index='target', columns='k', values='s2')
        out['gain_0_400_ae'] = tsum(a[400] - a[0]); out['gain_0_400_s2'] = tsum(b[400] - b[0])
        res[name][metric] = out

pd.concat(tables).round(3).to_csv(os.path.join(OUT, 'srs_curve_ae_minus_s2_by_country.csv'), index=False)
json.dump(res, open(os.path.join(OUT, 'srs_curve_ae_minus_s2_summary.json'), 'w'), indent=1, ensure_ascii=False)

for name in res:
    print('=====', name, res[name]['n_rows_srs_k0'])
    for metric in ['ba_ge30', 'oa']:
        print('--', metric)
        o = res[name][metric]
        for k in [0, 50, 100, 200, 400]:
            r = o[k]
            print(f"k={k:>3} AE {r['mean_ae']:6.2f} S2 {r['mean_s2']:6.2f}  d {r['mean']:+.2f} [{r['lo95']:+.2f},{r['hi95']:+.2f}] p={r['p']:.3f} pos {r['npos']}/{r['n']} med {r['median']:+.2f} | 去RO,PT {r['drop_RO_PT']['mean']:+.2f} [{r['drop_RO_PT']['lo95']:+.2f},{r['drop_RO_PT']['hi95']:+.2f}] {r['drop_RO_PT']['npos']}/9 | rho(dlat) {r['spearman_vs_mean_dlat']} rho(km) {r['spearman_vs_mean_km']}")
        for kk in ['change_0_to_200', 'change_0_to_400', 'gain_0_400_ae', 'gain_0_400_s2']:
            r = o[kk]
            print(f"  {kk}: {r['mean']:+.2f} [{r['lo95']:+.2f},{r['hi95']:+.2f}] p={r['p']:.3f} pos {r['npos']}/{r['n']} med {r['median']:+.2f}")
        print('  per country k0:', o[0]['per_country'])
        print('  per country k400:', o[400]['per_country'])
