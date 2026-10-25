import glob, json, os
import numpy as np, pandas as pd
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_fit'
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CLS = ['grassland', 'maize', 'orchards', 'other_crops', 'potatoes', 'root_crops', 'strawberries', 'tree_wood_forest',
       'vegetables', 'winter_cereals']
L = []
chk = []
for p in glob.glob(f'{OUT}/jobs/*.json'):
    if os.path.basename(p).startswith(('theta_', 'alt_')):
        continue
    r = json.load(open(p)); j = r['job']
    for it, m in r['test'].items():
        L.append(dict(kind=j['kind'], tgt=j['tgt'], feat=j['feat'], ev='test', it=int(it), oa=m['oa'], ba=m['ba'], nll=m['nll'],
                      **{c: m['rec'][i] for i, c in enumerate(CLS)}))
    for it, m in r['train'].items():
        L.append(dict(kind=j['kind'], tgt=j['tgt'], feat=j['feat'], ev='train', it=int(it), oa=m['oa'], ba=m['ba'], nll=m['nll']))
    if 'check_maxabs' in r:
        chk.append(dict(tgt=j['tgt'], feat=j['feat'], maxabs=r['check_maxabs'], agree=r['check_agree']))
df = pd.DataFrame(L)
df.to_csv(f'{OUT}/vf_long.csv', index=False)
pd.set_option('display.width', 250)
print('== 复现核对 k0 划分1 第300轮 vs 阶段0 P0test_i300 ==')
c = pd.DataFrame(chk); print(len(c), 'maxabs max', c.maxabs.max() if len(c) else None, 'agree min', c.agree.min() if len(c) else None)
t = df[df.ev == 'test']
print('\n== k0 L63 划分 1：目标测试（国家平均）==')
k0 = t[t.kind == 'k0']
print('国家数', k0.tgt.nunique())
print(k0.groupby(['feat', 'it'])[['ba', 'oa', 'nll']].mean().round(2).unstack(0).to_string())
w = k0.pivot_table(index=['tgt', 'feat'], columns='it', values='ba')
print('ba@300 < ba@100 的 国家×特征：', int((w[300] < w[100]).sum()), '/', len(w), '；平均差 ', round((w[300] - w[100]).mean(), 2))
print('每个 国家×特征 的 ba 最优检查点分布：', w.idxmax(axis=1).value_counts().sort_index().to_dict())
tr = df[(df.ev == 'train') & (df.kind == 'k0')]
print('训练集（外国 5 万）：\n', tr.groupby(['feat', 'it'])[['oa', 'ba']].mean().round(2).to_string())
print('\n逐类召回（k0，11 国平均，NaN 忽略）50/100/300 轮：')
for f in ('ae', 's2'):
    print(f, '\n', k0[k0.feat == f].groupby('it')[CLS].mean().loc[[50, 100, 300]].T.mul(100).round(1).to_string())
print('\n== 只用目标池（tgt，池频率类别权重）划分 1 ==')
tg = t[t.kind == 'tgt']
print('国家数', tg.tgt.nunique())
print(tg.groupby(['feat', 'it'])[['ba', 'oa']].mean().round(2).unstack(0).to_string())
m = tg.pivot_table(index=['tgt', 'feat'], columns='it', values=['ba', 'oa']).join(
    k0.pivot_table(index=['tgt', 'feat'], columns='it', values=['ba', 'oa']), lsuffix='_tgt', rsuffix='_k0')
gap = pd.DataFrame({'gap_ba_300': m[('ba_tgt', 300)] - m[('ba_k0', 300)], 'gap_oa_300': m[('oa_tgt', 300)] - m[('oa_k0', 300)],
                    'gap_ba_100': m[('ba_tgt', 100)] - m[('ba_k0', 100)],
                    'gap_ba_best': tg.pivot_table(index=['tgt', 'feat'], columns='it', values='ba').max(1) -
                                   k0.pivot_table(index=['tgt', 'feat'], columns='it', values='ba').max(1)})
print(gap.round(1).unstack(1).to_string())
print('平均：\n', gap.groupby(level='feat').mean().round(2).to_string())
print('gap_ba_300 > 0 的 国家×特征：', int((gap.gap_ba_300 > 0).sum()), '/', len(gap))
tt = tg.pivot_table(index='tgt', columns=['feat'], values='ba', aggfunc=lambda s: s.iloc[0] if len(s) else np.nan)
print('\n== 5 国子集（AT, DE, LV, PT, SE）==')
sub = t[t.tgt.isin(['AT', 'DE', 'LV', 'PT', 'SE'])]
g = sub.groupby(['kind', 'feat', 'it'])[['ba', 'oa']].mean().round(2)
for kd in ('k0', 'k0uw', 'k0L15', 'tgt', 'tgtfw', 'tgt400'):
    if kd in g.index.get_level_values(0):
        print(f'-- {kd} (n国={sub[sub.kind == kd].tgt.nunique()})\n', g.loc[kd].unstack(0).to_string())
print('\n逐类召回 k0uw（无类别权重，5 国）50/100/300：')
for f in ('ae', 's2'):
    x = sub[(sub.kind == 'k0uw') & (sub.feat == f)]
    if len(x):
        print(f, '\n', x.groupby('it')[CLS].mean().loc[[50, 100, 300]].T.mul(100).round(1).to_string())
trn = df[(df.ev == 'train') & df.tgt.isin(['AT', 'DE', 'LV', 'PT', 'SE'])]
print('\n训练集拟合（5 国）：\n', trn.groupby(['kind', 'feat', 'it'])[['oa', 'ba']].mean().round(1).unstack(1).to_string())
# G rows：外国 + SRS400（划分 1 rep0）对照
rr = pd.concat([pd.read_csv(f) for f in glob.glob(f'{GRES}/rows_t1_*_s1_r0.csv')])
rr = rr[(rr.method.isin(['srs', 'foreign_only']))]
print('\nG rows 划分1 rep0（5 国）外国+SRS 的 ba_ge30：\n',
      rr[rr.target.isin(['AT', 'DE', 'LV', 'PT', 'SE'])].groupby(['clf', 'k'])['ba_ge30'].mean().round(2).unstack(0).to_string())
# ---- 配对比较（5 国，同国同特征）
print('\n== 配对比较（5 国，同国同特征）：相对 k0 L63 ==')
P = sub.pivot_table(index=['tgt', 'feat', 'it'], columns='kind', values=['ba', 'oa'])
for kd in ('k0uw', 'k0L15'):
    dd = pd.DataFrame({'dba': P[('ba', kd)] - P[('ba', 'k0')], 'doa': P[('oa', kd)] - P[('oa', 'k0')]}).dropna()
    print(f'-- {kd} − k0：\n', dd.groupby(level=['feat', 'it']).mean().round(2).unstack(0).to_string())
    best = sub[sub.kind.isin([kd, 'k0'])].groupby(['tgt', 'feat', 'kind']).ba.max().unstack()
    print(f'   神谕最优轮 ba：{kd} − k0 =', (best[kd] - best['k0']).groupby(level='feat').mean().round(2).to_dict())
dd = pd.DataFrame({'dba': P[('ba', 'tgtfw')] - P[('ba', 'tgt')], 'doa': P[('oa', 'tgtfw')] - P[('oa', 'tgt')]}).dropna()
print('-- tgtfw − tgt（目标池，类别权重来源 外国 vs 池）：\n', dd.groupby(level=['feat', 'it']).mean().round(2).unstack(0).to_string())
ae_s2 = sub.pivot_table(index=['tgt', 'kind', 'it'], columns='feat', values='ba')
x = (ae_s2['ae'] - ae_s2['s2']).groupby(level=['kind', 'it']).mean().unstack(0).round(2)
print('\n分类器 AE−S2 ba（5 国均）：\n', x.to_string())
print('\nk0uw 逐国 ba@300 vs k0 ba@300：\n', sub[(sub.it == 300) & sub.kind.isin(['k0', 'k0uw'])].pivot_table(index=['tgt'], columns=['kind', 'feat'], values='ba').round(1).to_string())
