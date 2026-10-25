"""任务 B 机制诊断（事后探索，不属于预注册分析）：
加入目标国样本后，原来判对、现在判错的测试地块（"翻错"），新预测是否就是它在特征空间里最近的那个目标国样本的标签？
若是，说明高权重的少数目标国样本在其邻域"盖章"（局部过拟合）；
对照：同一翻错地块，新预测等于"随机一个被选目标国样本的标签"的概率（= 选中样本的类别构成，反映整体先验偏移）。
不训练，只读 preds_taskB 与主试点 preds。"""
import os
import sys

sys.dont_write_bytecode = True
CODE = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/'
        '我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.path.insert(0, CODE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from common import load_dataset, feature_cols  # noqa: E402
from pilot_target_selection import setup_target  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = '/Users/perng_mini/AlphaCrops_local/results/preds'
SH = ['natural', '0.01', '0.05', '0.25', '1.0']

df = load_dataset()
classes = sorted(df['label'].unique())
y_all = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
ctry = df['country'].to_numpy()
X = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in ['ae', 's2']}
df = df[['longitude', 'latitude', 'area_ha']].copy()
rows = []
for t in ['LV', 'SE']:
    foreign, pool, test, _ = setup_target(df, ctry, t)
    yt = y_all[test]
    big = [c for c in np.unique(yt) if (yt == c).sum() >= 30]
    for cf in ['ae', 's2']:
        mu, sd = X[cf][foreign].mean(0), X[cf][foreign].std(0) + 1e-9
        Zt = (X[cf][test] - mu) / sd
        for k in [50, 200, 1000]:
            for s in range(3):
                sel = pool[np.random.default_rng(1000 * s + k).choice(len(pool), k, replace=False)]
                ysel = y_all[sel]
                nn_lab = ysel[NearestNeighbors(n_neighbors=1).fit((X[cf][sel] - mu) / sd)
                              .kneighbors(Zt, return_distance=False).ravel()]
                prior = pd.Series(ysel).value_counts(normalize=True)
                fo = np.load(os.path.join(PRED, f'{t}_foreign_only_0_{s}_{cf}.npy'))
                for sh in SH:
                    p = np.load(os.path.join(HERE, 'preds_taskB', f'{t}_random_{k}_{s}_{cf}_share{sh}.npy'))
                    lost = (fo == yt) & (p != yt)      # 翻错
                    gain = (fo != yt) & (p == yt)      # 翻对
                    inbig = np.isin(yt, big)
                    for tag, m in [('lost', lost), ('gain', gain)]:
                        mm = m & inbig
                        n = int(mm.sum())
                        rows.append(dict(
                            target=t, clf_feat=cf, budget=k, seed=s, share=sh, kind=tag, n=n,
                            n_minor=int((mm & ~np.isin(yt, [classes.index(c) for c in
                                                             ['grassland', 'other_crops', 'winter_cereals']])).sum()),
                            # 新预测 = 最近目标国样本的标签
                            p_nn=float((p[mm] == nn_lab[mm]).mean()) if n else np.nan,
                            # 对照：新预测 = 随机一个选中样本的标签 的期望概率
                            p_prior=float(np.mean([prior.get(v, 0.0) for v in p[mm]])) if n else np.nan,
                            # 翻错地块的真实类别是否在选中样本中出现过
                            p_true_sampled=float(np.isin(yt[mm], ysel).mean()) if n else np.nan,
                        ))
r = pd.DataFrame(rows)
r.to_csv(os.path.join(HERE, 'taskB_mechanism.csv'), index=False)
pd.set_option('display.width', 250)
print('ba_ge30 类别上的翻错/翻对地块数（ae/s2 × 3 种子平均）与 翻错中少数类的比例')
g = r.groupby(['target', 'budget', 'share', 'kind'])
x = g.n.mean().unstack('kind')
x['net'] = x['gain'] - x['lost']
x['lost_minor_frac'] = (g.n_minor.sum() / g.n.sum()).unstack('kind')['lost']
print(x.reindex(pd.MultiIndex.from_product([['LV', 'SE'], [50, 200, 1000], SH])).round(2).to_string())
print('\n翻错地块：新预测=最近目标国样本标签 的比例 p_nn，对照 p_prior（随机选中样本标签），以及真实类别在选中样本中出现过的比例')
lost = r[(r.kind == 'lost') & (r.n > 0)]
w = lost.assign(pn=lost.p_nn * lost.n, pp=lost.p_prior * lost.n, pt=lost.p_true_sampled * lost.n)
agg = w.groupby(['target', 'budget', 'share'])[['pn', 'pp', 'pt', 'n']].sum()
agg = pd.DataFrame({'p_nn': agg.pn / agg.n, 'p_prior': agg.pp / agg.n, 'p_true_sampled': agg.pt / agg.n, 'n_total': agg.n})
print(agg.reindex(pd.MultiIndex.from_product([['LV', 'SE'], [50, 200, 1000], SH])).round(3).to_string())
