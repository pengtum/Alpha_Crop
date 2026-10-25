"""strat_split 修复：稀有类（<2）强制进训练集。检查与原实现在无稀有类时逐元素一致，且 PT 不再报错。"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']; RS = 42
df = pd.read_parquet('/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet',
                     columns=['country','label','has_ae','has_s2','longitude','latitude','area_ha'])
df = df[df.has_ae & df.has_s2].reset_index(drop=True)
classes = sorted(df['label'].unique()); y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
ctry = df['country'].to_numpy()

def strat_split_old(idx, y, test_size, seed):
    counts = np.bincount(y[idx], minlength=y.max() + 1)
    strat = np.where(counts[y[idx]] >= 2, y[idx], -1)
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)

def strat_split_new(idx, y, test_size, seed):
    """分层划分；样本数 <2 的类别全部放进训练集（保证验证集没有训练集未见的标签）。"""
    counts = np.bincount(y[idx], minlength=y.max() + 1)
    rare = counts[y[idx]] < 2
    tr, va = train_test_split(idx[~rare], test_size=test_size, random_state=seed, stratify=y[idx[~rare]])
    return np.concatenate([tr, idx[rare]]), va

for tgt in COUNTRIES:
    rng = np.random.default_rng(RS)
    foreign = np.concatenate([rng.choice(np.flatnonzero(ctry == c), min(10_000, int((ctry == c).sum())), replace=False)
                              for c in COUNTRIES if c != tgt])
    t_idx = np.flatnonzero(ctry == tgt)
    t_idx = t_idx[~df.iloc[t_idx].duplicated(['longitude', 'latitude', 'area_ha']).to_numpy()]
    pool, test = train_test_split(t_idx, test_size=0.3, random_state=RS)
    out = []
    for name, idx in (('foreign', foreign), ('pool', pool)):
        a, b = strat_split_new(idx, y, 0.1, RS)
        unseen = set(y[b]) - set(y[a])
        try:
            a0, b0 = strat_split_old(idx, y, 0.1, RS); same = np.array_equal(a0, a) and np.array_equal(b0, b)
        except ValueError:
            same = '原实现报错'
        out.append(f'{name}: 与原实现一致={same}, 验证集未见类={unseen or "无"}')
    print(tgt, ' | '.join(out))
