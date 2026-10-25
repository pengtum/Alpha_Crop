"""(f) 与检验 2（4 国，无缓冲，r 固定 25%）和阶段 A PT 期中的方向一致性。"""
from common_bm import *

R = load()
RES = '/Users/perng_mini/AlphaCrops_local/results'
lines = []

# ---------------- 检验 2
P = pd.read_csv(f'{RES}/pilot_selection.csv')
P = P[P.budget.isin([50, 200, 1000])]
P['ba_ge30'] *= 100
strat = ['geo_strat', 's2_rep', 'ae_rep', 's2_gap', 'ae_gap', 'uncertainty']
base = P[P.strategy == 'random'].set_index(['target', 'budget', 'seed', 'clf_feat']).ba_ge30
rows = []
for s in strat:
    g = P[P.strategy == s].set_index(['target', 'budget', 'seed', 'clf_feat'])
    d = (g.ba_ge30 - base.reindex(g.index)).rename('d').reset_index()
    d['n_cls_d'] = (g.n_classes_sel - P[P.strategy == 'random'].set_index(['target', 'budget', 'seed', 'clf_feat']).n_classes_sel.reindex(g.index)).to_numpy()
    for b in (50, 200, 1000):
        x = d[d.budget == b].groupby('target').d.mean()
        rows.append(dict(strategy=s, k=b, **{c: x.get(c) for c in ['LV', 'PT', 'SE', 'SK']}, mean4=x.mean(), n_pos=int((x > 0).sum()),
                         d_ncls=d[d.budget == b].n_cls_d.mean()))
C2 = pd.DataFrame(rows)
C2.to_csv(f'{OUTD}/f_check2_vs_random.csv', index=False)
lines.append('=== 检验 2：策略 − random（ba_ge30 pp；3 种子 × 2 分类器平均）===\n' + C2.round(2).to_string(index=False))

# G 中同 4 国（合并指导/分类器；k=50、200）
D = all_vs_srs(R, 'ba_ge30')
g4 = D[D.target.isin(['LV', 'PT', 'SE', 'SK'])]
G4 = g4.groupby(['method', 'k', 'target', 'cond', 'clf']).d.mean().groupby(['method', 'k', 'target']).mean().unstack('target')
G4['mean4'] = G4.mean(axis=1)
G4['n_pos'] = (G4[['LV', 'PT', 'SE', 'SK']] > 0).sum(axis=1)
G4.to_csv(f'{OUTD}/f_G_same4.csv')
lines.append('\n=== G：同 4 国，方法 − SRS（ba_ge30，AE+S2 合并）===\n' + G4.round(2).to_string())

# 映射（按机制相近，粗）：uncertainty ↔ margin / gist_margin；*_rep ↔ typiclust / kherding；*_gap ↔ kcenter；geo_strat ↔ XY 条件的 lpm2_nogeo/kcenter
mp = {'uncertainty': ['margin', 'gist_margin'], 'ae_rep': ['typiclust', 'kherding'], 's2_rep': ['typiclust', 'kherding'],
      'ae_gap': ['kcenter'], 's2_gap': ['kcenter']}
cmp = []
for s, ms in mp.items():
    for b in (50, 200):
        c2 = C2[(C2.strategy == s) & (C2.k == b)].iloc[0]
        for m in ms:
            gg = G4.loc[(m, b)]
            agree = sum(np.sign(c2[c]) == np.sign(gg[c]) for c in ['LV', 'PT', 'SE', 'SK'])
            cmp.append(dict(check2=s, G=m, k=b, check2_mean4=c2.mean4, G_mean4=gg.mean4, sign_agree_countries=agree,
                            check2_SE=c2.SE, G_SE=gg.SE))
# geo_strat ↔ XY
xy = pd.concat([A.vs_srs(R[R.rep == 0], 'XY', ['lpm2_nogeo', 'kcenter'])]).assign(cond='XY')
xy4 = xy[xy.target.isin(['LV', 'PT', 'SE', 'SK'])].groupby(['k', 'target', 'method', 'clf']).d.mean().groupby(['k', 'target']).mean().unstack('target')
for b in (50, 200):
    c2 = C2[(C2.strategy == 'geo_strat') & (C2.k == b)].iloc[0]
    gg = xy4.loc[b]
    cmp.append(dict(check2='geo_strat', G='XY(lpm2/kcenter)', k=b, check2_mean4=c2.mean4, G_mean4=gg.mean(),
                    sign_agree_countries=sum(np.sign(c2[c]) == np.sign(gg[c]) for c in ['LV', 'PT', 'SE', 'SK']), check2_SE=c2.SE, G_SE=gg.SE))
CMP = pd.DataFrame(cmp)
CMP.to_csv(f'{OUTD}/f_check2_mapping.csv', index=False)
lines.append('\n=== 检验 2 ↔ G 粗映射（方向）===\n' + CMP.round(2).to_string(index=False))

# 检验 2 自身：SE 的 uncertainty 在两种分类器下
u = P[(P.strategy.isin(['uncertainty', 'random'])) & P.budget.isin([50, 200])].pivot_table(index=['target', 'budget', 'clf_feat'], columns='strategy', values='ba_ge30', aggfunc='mean')
u['d'] = u.uncertainty - u.random
lines.append('\n检验 2 uncertainty − random（按国家、k、分类器）：\n' + u.d.unstack('clf_feat').round(2).to_string())

# ---------------- 阶段 A PT 期中
S = pd.read_csv(f'{RES}/stageA/stageA_merged_interim_PT.csv')
srsA = S[S.method == 'srs'].set_index(['split', 'rep', 'k'])[['ba_ge30', 'oa']]
S = S.join(srsA, on=['split', 'rep', 'k'], rsuffix='_srs')
S['d_ba'] = S.ba_ge30 - S.ba_ge30_srs
S['d_oa'] = S.oa - S.oa_srs
mapA = {'lpm2': 'lpm2_nogeo', 'kcenter': 'kcenter', 'typiclust': 'typiclust', 'kherding': 'kherding', 'margin': 'margin',
        'predstrat': 'predstrat_srs', 'gist_margin': 'gist_margin', 'badge': 'badge_c'}
units = [(0, 0), (0, 1), (1, 0)]  # 与 G 共用的单元（SRS 逐位相同）
out = []
Gpt = R[(R.target == 'PT') & (R.clf == 's2') & (R.k == 50)]
srsG = Gpt[Gpt.method == 'srs'].set_index(['split', 'rep'])[['ba_ge30', 'oa']]
for a, gm in mapA.items():
    sa = S[(S.method == a) & (S.k == 50)]
    sa_c = sa[[tuple(x) in units for x in sa[['split', 'rep']].to_numpy()]]
    row = dict(stageA=a, G=gm, A_all4=sa.d_ba.mean(), A_common3=sa_c.d_ba.mean(), A_oa_common3=sa_c.d_oa.mean())
    for cond in ('AE', 'S2', 'RAWZ_AE'):
        gg = Gpt[(Gpt.cond == cond) & (Gpt.method == gm)].set_index(['split', 'rep'])
        if len(gg) == 0:
            continue
        dd = (gg.ba_ge30 - srsG.ba_ge30.reindex(gg.index))
        do = (gg.oa - srsG.oa.reindex(gg.index))
        row[f'G_{cond}_common3'] = dd.reindex(units).mean()
        row[f'G_{cond}_oa_common3'] = do.reindex(units).mean()
        row[f'G_{cond}_same_sel'] = int((gg.sel_hash.reindex(units).to_numpy() == None).sum()) if False else None
    out.append(row)
SA = pd.DataFrame(out)
SA.to_csv(f'{OUTD}/f_stageA_PT.csv', index=False)
lines.append('\n=== 阶段 A PT（S2 分类器，k=50）与 G PT（s2 分类器，k=50）同 3 个单元 ===\n' + SA.round(2).drop(columns=[c for c in SA if c.endswith('same_sel')]).to_string(index=False))
# G 全 11 国对应值（同设置：clf=s2, k=50）
g11 = []
for gm in mapA.values():
    for cond in ('AE', 'S2'):
        d = vs_srs(R, cond, [gm], 'ba_ge30', ks=(50,), clfs=('s2',))
        v = cv(d)
        g11.append(dict(method=gm, cond=cond, mean11=v.mean(), n_pos=int((v > 0).sum()), PT=v['PT']))
lines.append('\nG 11 国（clf=s2，k=50）：\n' + pd.DataFrame(g11).round(2).to_string(index=False))
# 阶段 A 的方法排名与 G PT 排名的一致性（k=50，同 3 单元）
x = SA.set_index('G')
for cond in ('AE', 'S2'):
    r = stats.spearmanr(x.A_common3, x[f'G_{cond}_common3'])
    lines.append(f'阶段 A 与 G（PT，{cond} 指导）8 方法排名 Spearman = {r[0]:+.2f}（p={r[1]:.3f}）')
open(f'{OUTD}/f_crosscheck.log', 'w').write('\n'.join(lines))
print('\n'.join(lines))
