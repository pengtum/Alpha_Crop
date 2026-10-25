"""uncertainty − random 的逐类 recall 差（从预测文件算）；看优势是否集中在某个类别。"""
import numpy as np
import pandas as pd

L = '/Users/perng_mini/AlphaCrops_local'
P = f'{L}/results/preds'
OUTD = f'{L}/verify/check2'
df = pd.read_parquet(f'{L}/data/dataset_v2.parquet', columns=['label', 'has_ae', 'has_s2'])
df = df[df.has_ae & df.has_s2].reset_index(drop=True)
classes = sorted(df.label.unique())
y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
TG = ['PT', 'SE', 'SK', 'LV']
STRATS = ['random', 'uncertainty', 'ae_rep', 's2_rep', 'geo_strat']

rows = []
for t in TG:
    ti = np.load(f'{P}/{t}_test_idx.npy')
    yt = y[ti]
    pres, cnt = np.unique(yt, return_counts=True)
    for s in STRATS:
        for k in [50, 200, 1000]:
            for cf in ['ae', 's2']:
                for seed in [0, 1, 2]:
                    pr = np.load(f'{P}/{t}_{s}_{k}_{seed}_{cf}.npy').astype(int)
                    for c, n in zip(pres, cnt):
                        m = yt == c
                        rows.append(dict(target=t, strategy=s, budget=k, clf_feat=cf, seed=seed, cls=classes[c], n_test=n,
                                         recall=(pr[m] == c).mean(),
                                         pred_share=(pr == c).mean()))
    # 参照 foreign_only
    for cf in ['ae', 's2']:
        for seed in [0, 1, 2]:
            pr = np.load(f'{P}/{t}_foreign_only_0_{seed}_{cf}.npy').astype(int)
            for c, n in zip(pres, cnt):
                m = yt == c
                rows.append(dict(target=t, strategy='foreign_only', budget=0, clf_feat=cf, seed=seed, cls=classes[c], n_test=n,
                                 recall=(pr[m] == c).mean(), pred_share=(pr == c).mean()))
d = pd.DataFrame(rows)
d.to_csv(f'{OUTD}/perclass_recall.csv', index=False)
ge = d[d.n_test >= 30]
cm = ge.groupby(['target', 'budget', 'clf_feat', 'cls', 'strategy']).recall.mean().unstack('strategy')
ncls = ge.groupby('target').cls.nunique()
diff = (cm['uncertainty'] - cm['random']) * 100
# 对 ba_ge30 的贡献 = 类 recall 差 / 该目标国 ≥30 类别数
contrib = diff / diff.index.get_level_values('target').map(ncls).to_numpy()
pd.set_option('display.width', 250)
print('每目标国 ≥30 类数', ncls.to_dict())
print('\n== uncertainty − random 逐类 recall 差（pp），按目标国平均（6 个单元）==')
print(diff.groupby(['target', 'cls']).mean().unstack('target').round(2))
print('\n== 对 ba_ge30 差的贡献（pp），按目标国 ==')
ct = contrib.groupby(['target', 'cls']).mean().unstack('target')
print(ct.round(3))
print('合计（应等于各目标国 ba_ge30 差）:', ct.sum().round(3).to_dict())
print('\n== 全 24 单元平均贡献（pp）==')
tot = contrib.groupby(['cls']).sum() / 24
print(tot.sort_values(ascending=False).round(3), 'sum', round(tot.sum(), 4))
print('\n== uncertainty − random 逐类 recall 差（pp），按预算 × 目标国 ==')
print(diff.groupby(['target', 'budget', 'cls']).mean().unstack('cls').round(1))
# 去掉贡献最大类后
print('\n== 各类“留一”后 uncertainty−random 的 ba_ge30 差（重算：剩余类 recall 平均）==')
for drop in sorted(ge.cls.unique()):
    g2 = ge[ge.cls != drop]
    c2 = g2.groupby(['target', 'budget', 'clf_feat', 'seed', 'strategy']).recall.mean().unstack('strategy')
    c2 = c2.groupby(level=['target', 'budget', 'clf_feat']).mean()
    dd = (c2['uncertainty'] - c2['random']) * 100
    from scipy.stats import wilcoxon
    pt = dd.groupby(level='target').mean()
    print(f'  drop {drop:17s} Δ={dd.mean():+.3f}  p={wilcoxon(dd).pvalue:.4f}  pos {int((dd>0).sum())}/24  '
          + ' '.join(f'{t}{pt[t]:+.2f}' for t in TG))
# 预测份额变化
print('\n== uncertainty − random：预测为各类的份额差（pp，测试集全部地块）==')
ps = d[d.strategy.isin(['uncertainty', 'random'])].groupby(['target', 'cls', 'strategy']).pred_share.mean().unstack('strategy')
print(((ps['uncertainty'] - ps['random']) * 100).unstack('target').round(2))
print('\n== 各类 recall 水平（foreign_only / random / uncertainty，全部预算与特征平均）==')
lv = d.groupby(['target', 'cls', 'strategy']).recall.mean().unstack('strategy')[['foreign_only', 'random', 'uncertainty']] * 100
lv['n_test'] = d.groupby(['target', 'cls']).n_test.first()
print(lv.round(1))
