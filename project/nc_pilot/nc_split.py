"""
NC 试点第 3 步：样本表、目标域抽取、划分、抽样顺序与 BA 类别（方案第 3、4、5 节）。

只用坐标、编号与类别计数；不读任何特征，不训练，不计算精度。
输出 ~/AlphaCrops_local/nc_pilot/splits/：
  t1_samples.parquet  row（npz 行号）、city、lon、lat、y、role（source / test / cand / buffer）、x、y_m、epsg
  t2_samples.parquet  id（LUCAS id，整数）、nuts0、cprn_lc、y（0..6）、lon、lat、x、y_m、role（G 水体已剔除）
  lucas_meta_raw.parquet  从 GEE 读取的原始元数据（id、nuts0、lc1、cprn_lc、centroid(1) 经纬度、面积）
  orders.npz          键 '{域}_r{r}'：候选样本编号按抽样顺序排列（剔除缺失之前）
  ba_classes.json     {域: [类别下标]}
  targets.json        {任务: [域]}
  manifest.json       以上文件的 sha256

用法：python nc_split.py [--lucas-meta <已存的 parquet>] [--out <目录>]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nc_common as K  # noqa: E402


def draw_targets(groups, seed_str):
    rng = K.rng_of(seed_str)
    return [str(rng.choice(sorted(members))) for _, members in groups]


def split_domain(x, y, grid, off, buf, split_seed):
    """网格划分：返回 role 数组（test / cand / buffer）。坐标先四舍五入到 1 m；候选 = 到最近测试样本中心 ≥ buf。"""
    x, y = np.round(x), np.round(y)
    gx, gy = np.floor((x - off) / grid).astype(np.int64), np.floor((y - off) / grid).astype(np.int64)
    cells = sorted(set(zip(gx.tolist(), gy.tolist())))
    rng = K.rng_of(split_seed)
    pick = rng.choice(len(cells), round(K.TEST_FRAC * len(cells)), replace=False)
    test_cells = {cells[i] for i in pick}
    te = np.array([(a, b) in test_cells for a, b in zip(gx.tolist(), gy.tolist())])
    d, _ = cKDTree(np.c_[x[te], y[te]]).query(np.c_[x[~te], y[~te]])
    role = np.where(te, 'test', 'buffer').astype(object)
    idx = np.flatnonzero(~te)
    role[idx[d >= buf]] = 'cand'
    return role


def n_clusters(x, y, dist):
    if len(x) == 0:
        return 0
    pairs = cKDTree(np.c_[x, y]).query_pairs(dist, output_type='ndarray')
    from scipy.sparse import coo_matrix
    n = len(x)
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    return connected_components(A, directed=False)[0]


def build_t1():
    d = np.load(K.NPZ, allow_pickle=True)
    names = [str(v) for v in d['region_names']]
    reg, yy, lon, lat = d['region'], d['y'].astype(int), d['lon'], d['lat']
    assert set(np.unique(yy)) == set(range(K.T1_C))
    elig = [n for i, n in enumerate(names) if (reg == i).sum() >= K.T1_MIN_N and len(np.unique(yy[reg == i])) >= K.T1_MIN_C]
    assert sorted(elig) == sorted(c for _, g in K.T1_GROUPS for c in g) and len(elig) == 32, elig
    targets = draw_targets(K.T1_GROUPS, 'nc_pilot_v1_lcz')
    city = np.array([names[i] for i in reg], dtype=object)
    df = pd.DataFrame(dict(row=np.arange(len(yy)), city=city, lon=lon, lat=lat, y=yy))
    df = df[df.city.isin(elig)].copy()
    df['role'] = np.where(df.city.isin(targets), '', 'source')
    df['x'], df['y_m'], df['epsg'] = np.nan, np.nan, 0
    for c in elig:                                 # 每城自己的 UTM（提取足迹也用它）
        m = df.city == c
        ep = K.utm_epsg(df.loc[m, 'lon'].mean(), df.loc[m, 'lat'].mean())
        x, y = Transformer.from_crs(4326, ep, always_xy=True).transform(df.loc[m, 'lon'].values, df.loc[m, 'lat'].values)
        df.loc[m, 'x'], df.loc[m, 'y_m'], df.loc[m, 'epsg'] = x, y, ep
    ba = {}
    for c in targets:
        m = (df.city == c).values
        sub = df[m]
        df.loc[m, 'role'] = split_domain(sub.x.values, sub.y_m.values, K.T1_GRID, K.T1_GRID_OFF, K.T1_BUF, f'nc_pilot_v1_split_{c}')
        te = df[m & (df.role == 'test').values]
        ba[c] = [int(k) for k in range(K.T1_C)
                 if (te.y == k).sum() >= K.BA_MIN
                 and n_clusters(te.x[te.y == k].values, te.y_m[te.y == k].values, K.T1_CLUSTER) >= K.BA_MIN_CLUSTERS]
    return df.reset_index(drop=True), targets, ba


def fetch_lucas_meta():
    import ee
    ee.Initialize(project=K.GEE_PROJECT)
    fc = ee.FeatureCollection(K.LUCAS_ASSET)

    def f(ft):
        c = ft.geometry().centroid(1).coordinates()
        return ee.Feature(None, {'id': ft.get('id'), 'nuts0': ft.get('nuts0'), 'lc1': ft.get('lc1'),
                                 'cprn_lc': ft.get('cprn_lc'),
                                 'lon': c.get(0), 'lat': c.get(1), 'area': ft.geometry().area(1)})
    df = ee.data.computeFeatures({'expression': fc.map(f), 'fileFormat': 'PANDAS_DATAFRAME'})
    return df.drop(columns=[c for c in ('geo',) if c in df.columns])


def build_t2(meta):
    df = meta[['id', 'nuts0', 'cprn_lc', 'lon', 'lat', 'area']].copy()
    df['id'] = df.id.astype(np.int64)
    assert df.id.is_unique and df.cprn_lc.notna().all() and len(df) == 63287
    first = df.cprn_lc.str[0]
    assert set(first) <= set('ABCDEFGH'), set(first)
    df = df[first != K.T2_DROP].copy()
    df['y'] = df.cprn_lc.str[0].map({c: i for i, c in enumerate(K.T2_CLASSES)}).astype(int)
    cnt = df.nuts0.value_counts()
    big = sorted(cnt[cnt >= K.T2_MIN_N].index)
    assert big == sorted(c for _, g in K.T2_GROUPS for c in g), big
    targets = draw_targets(K.T2_GROUPS, 'nc_pilot_v1_lucas')
    df['x'], df['y_m'] = Transformer.from_crs(4326, K.T2_EPSG, always_xy=True).transform(df.lon.values, df.lat.values)
    df['role'] = np.where(df.nuts0.isin(targets), '', 'source')
    df = df.sort_values('id').reset_index(drop=True)
    ba = {}
    for c in targets:
        m = (df.nuts0 == c).values
        sub = df[m]
        df.loc[m, 'role'] = split_domain(sub.x.values, sub.y_m.values, K.T2_GRID, K.T2_GRID_OFF, K.T2_BUF, f'nc_pilot_v1_split_{c}')
        te = df[m & (df.role == 'test').values]
        ba[c] = [int(k) for k in range(K.T2_C) if (te.y == k).sum() >= K.BA_MIN]
    return df, targets, ba


def orders_of(df, dom_col, id_col, targets):
    out = {}
    for c in targets:
        ids = np.sort(df.loc[(df[dom_col] == c) & (df.role == 'cand'), id_col].values)
        for r in range(K.R):
            out[f'{c}_r{r}'] = ids[K.rng_of(f'nc_pilot_v1_srs_{c}_r{r}').permutation(len(ids))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lucas-meta', default=None)
    ap.add_argument('--out', default=K.SPLITS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = pd.read_parquet(a.lucas_meta) if a.lucas_meta else fetch_lucas_meta()
    meta.to_parquet(os.path.join(a.out, 'lucas_meta_raw.parquet'), index=False)

    t1, tg1, ba1 = build_t1()
    t2, tg2, ba2 = build_t2(meta)
    t1.to_parquet(os.path.join(a.out, 't1_samples.parquet'), index=False)
    t2.to_parquet(os.path.join(a.out, 't2_samples.parquet'), index=False)
    orders = {**orders_of(t1, 'city', 'row', tg1), **orders_of(t2, 'nuts0', 'id', tg2)}
    np.savez(os.path.join(a.out, 'orders.npz'), **orders)
    json.dump({**ba1, **ba2}, open(os.path.join(a.out, 'ba_classes.json'), 'w'), indent=1)
    json.dump({'T1': tg1, 'T2': tg2}, open(os.path.join(a.out, 'targets.json'), 'w'), indent=1)

    files = ['lucas_meta_raw.parquet', 't1_samples.parquet', 't2_samples.parquet', 'orders.npz', 'ba_classes.json',
             'targets.json']
    man = {f: K.sha256(os.path.join(a.out, f)) for f in files}
    json.dump(man, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)

    print('T1 targets', tg1, '| T2 targets', tg2)
    for task, df, col, tg in (('T1', t1, 'city', tg1), ('T2', t2, 'nuts0', tg2)):
        print(task, 'source', int((df.role == 'source').sum()), 'domains', df.loc[df.role == 'source', col].nunique())
        for c in tg:
            s = df[df[col] == c]
            print(f'  {c}: test {int((s.role == "test").sum())} cand {int((s.role == "cand").sum())} '
                  f'buffer {int((s.role == "buffer").sum())} | BA classes {({**ba1, **ba2})[c]}')
    for f, h in man.items():
        print(f'{f} sha256[:16]={h[:16]}')


if __name__ == '__main__':
    main()
