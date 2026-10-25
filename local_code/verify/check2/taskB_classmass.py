"""各份额下训练集中每类的总权重（外国+目标国），看缩放是否破坏类别平衡（不训练，只算权重）。"""
import sys, os
sys.dont_write_bytecode = True
CODE = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/'
        '我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from pilot_target_selection import setup_target
df = pd.read_parquet('/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet',
                     columns=['country', 'label', 'has_ae', 'has_s2', 'longitude', 'latitude', 'area_ha'])
df = df[df.has_ae & df.has_s2].reset_index(drop=True)
classes = sorted(df['label'].unique())
y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
ctry = df.country.to_numpy()
rows = []
for tgt in ['LV', 'SE']:
    foreign, pool, test, _ = setup_target(df, ctry, tgt)
    pp = pd.Series(y[pool]).value_counts(normalize=True).sort_index()
    print(tgt, 'pool class share %:', {classes[c][:10]: round(v * 100, 2) for c, v in pp.items()})
    fp = pd.Series(y[foreign]).value_counts(normalize=True).sort_index()
    print(tgt, 'foreign class share %:', {classes[c][:10]: round(v * 100, 2) for c, v in fp.items()})
    for k in [50, 200, 1000]:
        for seed in range(3):
            sel = pool[np.random.default_rng(1000 * seed + k).choice(len(pool), k, replace=False)]
            yy = np.r_[y[foreign], y[sel]]; is_t = np.r_[np.zeros(len(foreign), bool), np.ones(k, bool)]
            for share in [0.01, 0.05, 0.25, 1.0, 'natural']:
                w = compute_sample_weight('balanced', yy)
                if share != 'natural':
                    w[is_t] *= share * w[~is_t].sum() / w[is_t].sum()
                tot = pd.Series(w).groupby(yy).sum()
                ft = pd.Series(w[~is_t]).groupby(yy[~is_t]).sum()
                for c in range(len(classes)):
                    rows.append(dict(target=tgt, budget=k, seed=seed, share=str(share), cls=classes[c],
                                     n_sel=int((y[sel] == c).sum()),
                                     mass_ratio=float(tot.get(c, 0) / ft.get(c, np.nan)),
                                     max_w_ratio=float(w[is_t & (yy == c)].max() / w[~is_t].mean()) if (y[sel] == c).any() else 0))
r = pd.DataFrame(rows)
r.to_csv('taskB_classmass.csv', index=False)
pd.set_option('display.width', 250)
t = r.groupby(['target', 'budget', 'share', 'cls']).mass_ratio.mean().unstack('cls')
print('\n每类总权重 / 该类外国权重（=1 表示目标国没贡献；3 个种子平均）')
print(t.round(2).to_string())
