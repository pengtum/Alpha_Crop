import numpy as np
import pandas as pd

L = '/Users/perng_mini/AlphaCrops_local'
OUTD = f'{L}/verify/check2'
s = pd.read_csv(f'{OUTD}/selection_class_counts.csv')
classes = ['grassland', 'maize', 'orchards', 'other_crops', 'potatoes', 'root_crops', 'strawberries',
           'tree_wood_forest', 'vegetables', 'winter_cereals']
sel = pd.read_csv(f'{L}/results/pilot_selection.csv')
bet = pd.read_csv(f'{L}/results/pilot_betti_parcel.csv')
bet = bet.assign(strategy='betti_' + bet.strategy)
ref = pd.concat([sel[sel.budget.isin([50, 200, 1000])], bet])[['target', 'strategy', 'budget', 'seed', 'clf_feat', 'n_classes_sel']]
st = s[s.strategy != '_pool_'].copy()
st['n_cls_mine'] = (st[classes] > 0).sum(1)
m = st.merge(ref, on=['target', 'strategy', 'budget', 'seed', 'clf_feat'], how='outer', indicator=True)
print('merge:', m._merge.value_counts().to_dict())
m['ok'] = m.n_cls_mine == m.n_classes_sel
print('n_classes_sel 复现一致:', int(m.ok.sum()), '/', len(m))
print(m[~m.ok][['target', 'strategy', 'budget', 'seed', 'clf_feat', 'n_cls_mine', 'n_classes_sel']].to_string())
print('n_sel==budget:', (st.n_sel == st.budget).all(), ' unique==n_sel:', (st.n_unique == st.n_sel).all())

pd.set_option('display.width', 250, 'display.max_columns', 30)
print('\n== n_classes_sel（CSV）按策略 × 预算平均 ==')
print(ref.groupby(['strategy', 'budget']).n_classes_sel.mean().unstack('budget').round(2))

pool = s[s.strategy == '_pool_'].set_index('target')[classes]
poolp = pool.div(pool.sum(1), axis=0) * 100
print('\n== 候选池类别占比（%）==')
print(poolp.round(1))

share = st.copy()
share[classes] = share[classes].div(share[classes].sum(1), axis=0) * 100
g = share.groupby(['target', 'budget', 'strategy'])[classes].mean()
for t in ['PT', 'SE', 'SK', 'LV']:
    print(f'\n== {t} 选样类别占比（%，种子与 clf_feat 平均）；最后一行 = 候选池 ==')
    x = g.loc[t].round(1)
    print(x.to_string())
    print('pool'.ljust(20), poolp.loc[t].round(1).to_dict())

# uncertainty − random 的类别占比差（按目标国，预算平均）
u = g.xs('uncertainty', level='strategy') - g.xs('random', level='strategy')
print('\n== uncertainty − random 选样类别占比差（pp），按目标国 × 预算 ==')
print(u.round(1).to_string())
# SE ae vs s2 的 uncertainty 选样差别
print('\n== SE uncertainty 选样类别占比，按 clf_feat（%）==')
print(share[(share.target == 'SE') & (share.strategy.isin(['uncertainty', 'random']))]
      .groupby(['budget', 'strategy', 'clf_feat'])[classes].mean().round(1).to_string())
