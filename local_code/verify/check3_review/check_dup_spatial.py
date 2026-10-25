import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']; RS = 42
df = pd.read_parquet('/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet',
                     columns=['country','label','has_ae','has_s2','longitude','latitude','area_ha'])
df = df[df.has_ae & df.has_s2].reset_index(drop=True)
classes = sorted(df['label'].unique()); y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
ctry = df['country'].to_numpy()
key = pd.util.hash_pandas_object(df[['longitude', 'latitude', 'area_ha']], index=False).to_numpy()
def strat_split(idx, y, test_size, seed):
    counts = np.bincount(y[idx], minlength=y.max() + 1); rare = counts[y[idx]] < 2
    tr, va = train_test_split(idx[~rare], test_size=test_size, random_state=seed, stratify=y[idx[~rare]])
    return np.concatenate([tr, idx[rare]]), va
print('tgt  外国f_tr∩f_va重复键  pool→test重复键  test到pool最近邻中位距离(m)  1-NN坐标分类OA  随机同标签率Σπ²')
for tgt in COUNTRIES:
    rng = np.random.default_rng(RS)
    foreign = np.concatenate([rng.choice(np.flatnonzero(ctry == c), min(10_000, int((ctry == c).sum())), replace=False)
                              for c in COUNTRIES if c != tgt])
    t_idx = np.flatnonzero(ctry == tgt)
    t_idx = t_idx[~df.iloc[t_idx].duplicated(['longitude', 'latitude', 'area_ha']).to_numpy()]
    pool, test = train_test_split(t_idx, test_size=0.3, random_state=RS)
    f_tr, f_va = strat_split(foreign, y, 0.1, RS)
    fdup = len(set(key[f_tr]) & set(key[f_va])); tdup = len(set(key[pool]) & set(key[test]))
    rad = np.radians(df[['latitude', 'longitude']].to_numpy())
    dist, nn = BallTree(rad[pool], metric='haversine').query(rad[test], k=1)
    oa_nn = (y[pool][nn[:, 0]] == y[test]).mean()
    pi = np.bincount(y[test], minlength=10) / len(test)
    print(f'{tgt}   {fdup:6d}              {tdup:4d}            {np.median(dist) * 6371e3:8.0f}              {oa_nn:.3f}        {(pi**2).sum():.3f}')
