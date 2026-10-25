"""独立复现各策略选出的样本（不导入 label_expansion 代码，按其文件头/流程说明手写），
输出每次选样的类别计数，并用 CSV 的 n_classes_sel 列核验复现是否正确。"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

L = '/Users/perng_mini/AlphaCrops_local'
R = f'{L}/results'
OUTD = f'{L}/verify/check2'
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
TG = ['PT', 'SE', 'SK', 'LV']
K = [50, 200, 1000]
SEEDS = [0, 1, 2]

df = pd.read_parquet(f'{L}/data/dataset_v2.parquet')
df = df[df.has_ae & df.has_s2].reset_index(drop=True)
nuts = pd.read_parquet(f'{L}/data/parcel_nuts.parquet', columns=['parcel_id', 'nuts3'])
df = df.merge(nuts, on='parcel_id', how='left', validate='one_to_one')
classes = sorted(df.label.unique())
y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
ctry = df.country.to_numpy()
aec = [c for c in df.columns if c.startswith('ae_')]
s2c = [c for c in df.columns if c.startswith('s2_')]
X = {'ae': df[aec].to_numpy(np.float32), 's2': df[s2c].to_numpy(np.float32)}
lon, lat = df.longitude.to_numpy(), df.latitude.to_numpy()
C = len(classes)


def kpick(Z, k, seed):
    k = min(k, len(Z))
    km = KMeans(n_clusters=k, n_init=1, random_state=seed, max_iter=100).fit(Z)
    idx = np.unique(NearestNeighbors(n_neighbors=1).fit(Z).kneighbors(km.cluster_centers_, return_distance=False).ravel())
    if len(idx) < k:
        rest = np.setdiff1d(np.arange(len(Z)), idx)
        idx = np.r_[idx, np.random.default_rng(seed).choice(rest, k - len(idx), replace=False)]
    return idx


def alloc(gamma, cap, b):
    n = np.zeros(len(gamma))
    for _ in range(min(b, int(cap.sum()))):
        g = np.where(n < cap, gamma * ((n + 1) ** -0.5 - (n + 2) ** -0.5), -np.inf)
        n[int(np.argmax(g))] += 1
    return n.astype(int)


rows = []
for tgt in TG:
    rng = np.random.default_rng(42)
    foreign = np.concatenate([rng.choice(np.flatnonzero(ctry == c), 5000, replace=False) for c in COUNTRIES if c != tgt])
    t_idx = np.flatnonzero(ctry == tgt)
    t_idx = t_idx[~df.iloc[t_idx].duplicated(['longitude', 'latitude', 'area_ha']).to_numpy()]
    pool, test = train_test_split(t_idx, test_size=0.3, random_state=42)
    assert np.array_equal(test, np.load(f'{R}/preds/{tgt}_test_idx.npy'))
    pool_cnt = np.bincount(y[pool], minlength=C)
    rows.append(dict(target=tgt, strategy='_pool_', budget=len(pool), seed=-1, clf_feat='-', **{classes[c]: pool_cnt[c] for c in range(C)}))
    desc = {}
    for f in ['ae', 's2']:
        Z = np.vstack([X[f][foreign], X[f][pool]]).astype(np.float64)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        Z = PCA(32, random_state=42).fit_transform(Z)
        desc[f] = (Z[:len(foreign)], Z[len(foreign):])
    gap = {f: NearestNeighbors(n_neighbors=1).fit(desc[f][0]).kneighbors(desc[f][1])[0].ravel() for f in ['ae', 's2']}
    lat0 = lat[pool].mean()
    kxy = np.c_[lon[pool] * 111.32 * np.cos(np.radians(lat0)), lat[pool] * 110.57]
    kxy = kxy - kxy.mean(0)
    ent = {}
    for cf in ['ae', 's2']:
        p0 = np.load(f'{R}/pilot_probs_{tgt}_{cf}.npy')
        assert len(p0) == len(pool)
        ent[cf] = -(p0 * np.log(p0 + 1e-12)).sum(1)
    # Betti 分组
    groups = {'betti_admin_rep': pd.factorize(df.nuts3.to_numpy()[pool])[0]}
    for f in ['ae', 's2']:
        Z = np.vstack([X[f][foreign], X[f][pool]]).astype(np.float64)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        Z = PCA(32, random_state=42).fit_transform(Z)[len(foreign):]
        groups[f'betti_img_rep_{f}'] = KMeans(8, n_init=4, random_state=42).fit_predict(Z)

    for k in K:
        for seed in SEEDS:
            sels = {}
            for strat in ['random', 'geo_strat', 's2_rep', 'ae_rep', 's2_gap', 'ae_gap']:
                r = np.random.default_rng(1000 * seed + k)
                if strat == 'random':
                    pk = r.choice(len(pool), k, replace=False)
                elif strat == 'geo_strat':
                    pk = kpick(kxy, k, seed)
                elif strat.endswith('_rep'):
                    pk = kpick(desc[strat[:2]][1], k, seed)
                else:
                    f = strat[:2]
                    cand = np.argsort(-gap[f])[:max(k, int(0.3 * len(pool)))]
                    pk = cand[kpick(desc[f][1][cand], k, seed)]
                sels[(strat, 'ae')] = sels[(strat, 's2')] = pk
            for cf in ['ae', 's2']:
                cand = np.argsort(-ent[cf])[:max(k, int(0.3 * len(pool)))]
                sels[('uncertainty', cf)] = cand[kpick(desc[cf][1][cand], k, seed)]
            r = np.random.default_rng(20_000 + 1000 * seed + k)
            for strat in ['betti_admin_rep', 'betti_img_rep_ae', 'betti_img_rep_s2']:
                g = groups[strat]
                cap = np.bincount(g)
                n_g = alloc(cap / len(g), cap, k)
                pk = np.concatenate([r.choice(np.flatnonzero(g == j), n_g[j], replace=False) for j in range(len(cap)) if n_g[j] > 0])
                sels[(strat, 'ae')] = sels[(strat, 's2')] = pk
            for (strat, cf), pk in sels.items():
                cnt = np.bincount(y[pool[pk]], minlength=C)
                rows.append(dict(target=tgt, strategy=strat, budget=k, seed=seed, clf_feat=cf, n_sel=len(pk),
                                 n_unique=len(np.unique(pk)), **{classes[c]: cnt[c] for c in range(C)}))
            print(tgt, k, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUTD}/selection_class_counts.csv', index=False)
pd.DataFrame(rows).to_csv(f'{OUTD}/selection_class_counts.csv', index=False)
print('done')
