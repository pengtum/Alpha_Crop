"""免训练：用 G 全部 rows（两划分 × 两重复 × 4 个 E1 方法 × k∈{50,200} × 两分类器）看 θ(AE−S2) 在 200/250/300 轮的稳定性，
并在不同聚合层级上算 200 vs 300 的相关（对照设计审查的 0.14）。"""
import glob
import numpy as np, pandas as pd
from scipy import stats
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
d = pd.concat([pd.read_csv(f) for f in glob.glob(f'{GRES}/rows_t1_*.csv')])
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
g = d[d.method.isin(E1) & d.cond.isin(['AE', 'S2'])]
key = ['target', 'split', 'rep', 'method', 'k', 'clf']
out = []
piv = g.pivot_table(index=key, columns='cond', values=[f'ba_i{i}' for i in (200, 250, 300)] + ['oa_i300'])
th = pd.DataFrame({i: piv[f'ba_i{i}']['AE'] - piv[f'ba_i{i}']['S2'] for i in (200, 250, 300)}).dropna()
print('单元数（国家×划分×重复×方法×k×分类器）:', len(th))
print('θ_ba 平均 (所有单元)：', th.mean().round(3).to_dict())
for lvl, grp in (('单元', key), ('国家×划分×重复×k×分类器（方法平均）', ['target', 'split', 'rep', 'k', 'clf']),
                 ('国家×划分×重复（方法、k、分类器平均）', ['target', 'split', 'rep']), ('国家', ['target'])):
    a = th.groupby(level=grp).mean()
    r2 = a[200].corr(a[300]); r25 = a[250].corr(a[300])
    sg = (np.sign(a[200]) == np.sign(a[300])).mean()
    print(f'{lvl}: n={len(a)}  corr(200,300)={r2:.2f}  corr(250,300)={r25:.2f}  同号比例(200 vs 300)={sg:.2f}  '
          f'sd(θ300)={a[300].std():.2f}  sd(θ300−θ200)={(a[300]-a[200]).std():.2f}')
# 国家层面的 E1 θ 与 95% CI（国家为单位）
c = th.groupby(level='target').mean()
for i in (200, 250, 300):
    v = c[i]; ci = stats.t.interval(0.95, len(v) - 1, v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
    print(f'E1 θ_ba @{i}: {v.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  >0 国家 {(v>0).sum()}/11')
print('SE 单元层 corr(200,300)：', th.xs('SE', level='target')[200].corr(th.xs('SE', level='target')[300]).round(2))
# 只看 fit 代理同样的子集：k=200, lpm2_nogeo+typiclust, s0 r0
sub = th.reset_index()
sub = sub[(sub.k == 200) & sub.method.isin(['lpm2_nogeo', 'typiclust']) & (sub.split == 0) & (sub.rep == 0)]
cs = sub.groupby('target')[[200, 250, 300]].mean()
print('fit 子集国家层 θ 平均：', cs.mean().round(3).to_dict(), ' corr(200,300)=', round(cs[200].corr(cs[300]), 2))
# SRS 与 k0 的 ba 随 200/250/300
s = d[(d.method.isin(['srs', 'foreign_only']))]
print(s.groupby(['method', 'k', 'clf'])[['ba_i200', 'ba_i250', 'ba_i300']].mean().round(2).to_string())
th.reset_index().to_csv('/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_fit/theta_units_G.csv', index=False)
