from common_mv import *
import pandas as pd
pd.set_option('display.width', 250)
df = pd.read_csv('d2_stacker_rows.csv')
c = df.groupby(['arm', 'k', 'target'])[['ba', 'oa']].mean().reset_index()
k0 = c[c.arm.isin(['k0_ae', 'k0_s2'])].pivot(index='target', columns='arm', values='ba')
k0oa = c[c.arm.isin(['k0_ae', 'k0_s2'])].pivot(index='target', columns='arm', values='oa')
# G：SRS 重训（r=50），两种分类器
G = pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/headroom_map/G_all.pkl')
g = G[(G.cond == 'none') & (G.method.isin(['srs', 'foreign_only']))]
gm = g.groupby(['clf', 'k', 'target'])[['ba_ge30', 'oa']].mean()
out = []
for t in COUNTRIES:
    r = dict(target=t, k0_ae=k0.loc[t, 'k0_ae'], k0_s2=k0.loc[t, 'k0_s2'])
    best0 = max(r['k0_ae'], r['k0_s2'])
    for arm in ('orc_both_bal', 'orc_ae_bal', 'orc_s2_bal', 'orc_both_unw'):
        v = c[(c.arm == arm) & (c.target == t)]
        r[arm + '_dBA'] = v.ba.values[0] - best0
    for k in (50, 200, 400):
        for arm in ('cal_both', 'cal_ae', 'cal_s2'):
            v = c[(c.arm == arm) & (c.target == t) & (c.k == k)]
            r[f'{arm}_k{k}_dBA'] = v.ba.values[0] - best0
        for clf in ('ae', 's2'):
            r[f'G_srs_{clf}_k{k}_dBA'] = gm.loc[(clf, k, t), 'ba_ge30'] - gm.loc[(clf, 0, t), 'ba_ge30']
    out.append(r)
o = pd.DataFrame(out).set_index('target')
o.to_csv('d2_summary_by_country.csv')
cols1 = ['k0_ae', 'k0_s2', 'orc_ae_bal_dBA', 'orc_s2_bal_dBA', 'orc_both_bal_dBA', 'orc_both_unw_dBA']
print(o[cols1].round(2).to_string()); print('mean', o[cols1].mean().round(2).to_dict())
for k in (50, 200, 400):
    cc = [f'cal_ae_k{k}_dBA', f'cal_s2_k{k}_dBA', f'cal_both_k{k}_dBA', f'G_srs_ae_k{k}_dBA', f'G_srs_s2_k{k}_dBA']
    print(f'\n== k={k}（相对 k0 较好单视图；G 列相对各自分类器 k0）')
    print(o[cc].round(2).to_string())
    m = o[cc].mean(); m2 = o.drop(['RO', 'PT'])[cc].mean()
    print('mean', m.round(2).to_dict()); print('mean_noROPT', m2.round(2).to_dict())
    d = o[f'cal_both_k{k}_dBA'] - o[[f'cal_ae_k{k}_dBA', f'cal_s2_k{k}_dBA']].max(axis=1)
    print('both - max(single) mean %.2f, pos %d/11' % (d.mean(), (d > 0).sum()))
    d2 = o[f'cal_both_k{k}_dBA'] - o[f'cal_ae_k{k}_dBA']
    print('both - ae mean %.2f, pos %d/11' % (d2.mean(), (d2 > 0).sum()))
# OA
co = c.pivot_table(index='target', columns=['arm', 'k'], values='oa')
for k in (50, 200, 400):
    print(f'OA k={k}: cal_both - k0_ae mean %.2f' % (co[('cal_both', k)] - co[('k0_ae', 0)]).mean(),
          ' G srs ae dOA %.2f' % np.mean([gm.loc[('ae', k, t), 'oa'] - gm.loc[('ae', 0, t), 'oa'] for t in COUNTRIES]))
# 抽样之间的 SD
sd = df[df.arm == 'cal_both'].groupby(['target', 'split', 'k']).ba.std().groupby('k').mean()
print('cal_both 抽样间 SD', sd.round(2).to_dict())
