import glob, json
import numpy as np, pandas as pd
from scipy import stats
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_fit'
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
L = []
mx = []
for p in glob.glob(f'{OUT}/jobs/theta_*.json'):
    r = json.load(open(p)); j = r['job']; mx.append(r['check_maxabs'])
    for it, m in r['test'].items():
        L.append(dict(tgt=j['tgt'], cond=j['cond'], clf=j['clf'], it=int(it), ba=m['ba'], oa=m['oa'], src='new'))
d = pd.DataFrame(L)
print('作业数', len(mx), '第 200 轮与 G probs 缓存 maxabs 最大', max(mx))
rr = pd.concat([pd.read_csv(f) for f in glob.glob(f'{GRES}/rows_t1_*_s1_r0.csv')])
rr = rr[(rr.method == 'kcenter') & (rr.k == 50) & rr.cond.isin(['AE', 'S2'])]
for it in (200, 250, 300):
    x = rr[['target', 'cond', 'clf', f'ba_i{it}', f'oa_i{it}']].rename(columns={'target': 'tgt', f'ba_i{it}': 'ba', f'oa_i{it}': 'oa'})
    x['it'] = it; x['src'] = 'G'
    d = pd.concat([d, x], ignore_index=True)
# 核对第 200 轮
a = d[(d.it == 200)].pivot_table(index=['tgt', 'cond', 'clf'], columns='src', values='ba')
print('第 200 轮 ba 新训练 vs G rows 最大差', float((a['new'] - a['G']).abs().max()) if 'new' in a else None)
d = d[~((d.it == 200) & (d.src == 'G'))]
p = d.pivot_table(index=['tgt', 'clf', 'it'], columns='cond', values='ba')
th = (p['AE'] - p['S2']).groupby(level=['tgt', 'it']).mean().unstack()
print('国家数', th.dropna().shape[0])
for it in th.columns:
    v = th[it].dropna(); ci = stats.t.interval(0.95, len(v) - 1, v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
    print(f'θ_ba @{it}: {v.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  >0 {int((v > 0).sum())}/{len(v)}')
print('逐国：\n', th.round(2).to_string())
print('国家间相关：\n', th.corr().round(2).to_string())
# 单元层（国家×分类器）
u = (p['AE'] - p['S2']).unstack()
print('单元层（国家×分类器）相关 100 vs 300:', round(u[100].corr(u[300]), 2), ' 25 vs 300:', round(u[25].corr(u[300]), 2),
      ' 200 vs 300:', round(u[200].corr(u[300]), 2))
th.to_csv(f'{OUT}/theta_s1_kcenter_k50.csv')
