"""
NC 试点第 4 步：GEE 提取 AE 与 S2 特征（方案第 2 节）。

AE 与 S2 用同一几何、坐标系、尺度、年份；在逐像元特征图上直接 reduceRegions（均值 + 标准差，sharedInputs），
scale 10 m，tileScale 16，不做邻域平滑。
  T1  So2Sat 样本块：中心 ±160 m 的方块，在该城自己的 UTM 中构造与归约；2017
  T2  LUCAS 2018 Copernicus 多边形：EPSG:3035；2018
S2：COPERNICUS/S2_HARMONIZED（L1C），景云量 < 20%；按 system:index 连接 S2_CLOUD_PROBABILITY，
    掩去概率 ≥ 40 与 QA60 第 10、11 位；反射率 ÷ 10000 后计算指数；84 个逐像元统计量与 RSE
    src/extraction/gee_s2_extract.build_s2_composite 相同（年度波段 30、指数年度 24、季度 NDVI/EVI 16、季度关键波段 12、
    绿化 2）。
导出：Export.table.toAsset 到 projects/myproject-alpha-earth/assets/nc_pilot/，再用 computeFeatures 读回本地 parquet。

用法：
  python nc_extract.py submit [--only T1|T2] [--feat ae|s2] [--dry]
  python nc_extract.py status
  python nc_extract.py fetch            # 读回全部已完成资产 → features/{T1,T2}_{ae,s2}.parquet + versions.json
  python nc_extract.py interactive      # 受限模式下批处理排队过慢：对尚无资产的块用交互请求分小批计算（同一计算），
                                        # 存为 features/parts/*.parquet；fetch 时资产缺失则读取对应 part
  python nc_extract.py probe --task T1 --dom zurich --n 20   # 小样本交互测试（只看列与空值，不看标签）
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nc_common as K  # noqa: E402

S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
VI = ['NDVI', 'EVI', 'SAVI', 'NDWI', 'LSWI', 'NDBI']
AE_BANDS = [f'A{i:02d}' for i in range(64)]
CHUNK_T2 = 2000
CELL = 100000.0
SUBDIVIDE = {'T2': {'c36_25', 'c45_36', 'c49_34'}}   # 首轮导出超时的 S2 方格：再按 50 km 子块导出
CLOUD_PROB_MAX = 40


def ee_init():
    import ee
    ee.Initialize(project=K.GEE_PROJECT)
    return ee


# ---------------------------------------------------------------- 影像
def s2_masked(ee, year, region):
    s2 = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterDate(f'{year}-01-01', f'{year + 1}-01-01')
          .filterBounds(region).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    cp = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterDate(f'{year}-01-01', f'{year + 1}-01-01')
          .filterBounds(region))
    joined = ee.ImageCollection(ee.Join.saveFirst('cp').apply(
        primary=s2, secondary=cp, condition=ee.Filter.equals(leftField='system:index', rightField='system:index')))

    def mask(img):
        img = ee.Image(img)
        prob = ee.Image(img.get('cp')).select('probability')
        qa = img.select('QA60')
        ok = prob.lt(CLOUD_PROB_MAX).And(qa.bitwiseAnd(1 << 10).eq(0)).And(qa.bitwiseAnd(1 << 11).eq(0))
        refl = img.select(S2_BANDS).divide(10000).updateMask(ok)
        b2, b3, b4, b8, b11 = (refl.select(b) for b in ('B2', 'B3', 'B4', 'B8', 'B11'))
        ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI')
        evi = (b8.subtract(b4).multiply(2.5).divide(b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1))
               .rename('EVI'))
        savi = b8.subtract(b4).multiply(1.5).divide(b8.add(b4).add(0.5)).rename('SAVI')
        ndwi = b3.subtract(b8).divide(b3.add(b8)).rename('NDWI')
        lswi = b8.subtract(b11).divide(b8.add(b11)).rename('LSWI')
        ndbi = b11.subtract(b8).divide(b11.add(b8)).rename('NDBI')
        return (refl.addBands([ndvi, evi, savi, ndwi, lswi, ndbi])
                .set('system:time_start', img.get('system:time_start')))
    return joined.map(mask)


def guard(ee, col, img, names):
    """集合非空时原样返回 img（计算与首轮导出完全相同）；为空时返回同名全掩膜影像，
    导出为空值（方案第 2 节：某季度没有无云影像时保留为空值）。"""
    empty = ee.Image.constant([0] * len(names)).rename(names).toFloat().updateMask(ee.Image(0))
    return ee.Image(ee.Algorithms.If(col.size().gt(0), img, empty))


def s2_composite(ee, year, region):
    """84 个逐像元统计量，顺序与命名同 RSE build_s2_composite（不做邻域平滑）。"""
    col = s2_masked(ee, year, region)
    bands, names = [], []
    sel = col.select(S2_BANDS)
    med, mean, std = sel.median(), sel.mean(), sel.reduce(ee.Reducer.stdDev())
    for b in S2_BANDS:
        bands.append(med.select(b).rename(f'{b}_median')); names.append(f'{b}_median')
    for b in S2_BANDS:
        bands.append(mean.select(b).rename(f'{b}_mean')); names.append(f'{b}_mean')
    for b in S2_BANDS:
        bands.append(std.select(f'{b}_stdDev').rename(f'{b}_std')); names.append(f'{b}_std')
    v = col.select(VI)
    vmed, vmean, vstd, vmax = v.median(), v.mean(), v.reduce(ee.Reducer.stdDev()), v.max()
    for img, suf, src in ((vmed, 'median', None), (vmean, 'mean', None), (vstd, 'std', 'stdDev'), (vmax, 'max', None)):
        for x in VI:
            bands.append(img.select(f'{x}_{src}' if src else x).rename(f'{x}_{suf}')); names.append(f'{x}_{suf}')
    quarters = {'Q1': (f'{year}-01-01', f'{year}-04-01'), 'Q2': (f'{year}-04-01', f'{year}-07-01'),
                'Q3': (f'{year}-07-01', f'{year}-10-01'), 'Q4': (f'{year}-10-01', f'{year + 1}-01-01')}
    qmed = {}
    for q, (a, b) in quarters.items():
        qc = col.filterDate(a, b).select(['NDVI', 'EVI'])
        m = guard(ee, qc, qc.median(), ['NDVI', 'EVI'])
        s = guard(ee, qc, qc.reduce(ee.Reducer.stdDev()), ['NDVI_stdDev', 'EVI_stdDev'])
        for x in ('NDVI', 'EVI'):
            bands.append(m.select(x).rename(f'{x}_{q}_median')); names.append(f'{x}_{q}_median')
            bands.append(s.select(f'{x}_stdDev').rename(f'{x}_{q}_std')); names.append(f'{x}_{q}_std')
        qmed[q] = m
    for q, (a, b) in quarters.items():
        kc = col.filterDate(a, b).select(['B4', 'B8', 'B11'])
        m = guard(ee, kc, kc.median(), ['B4', 'B8', 'B11'])
        for x in ('B4', 'B8', 'B11'):
            bands.append(m.select(x).rename(f'{x}_{q}_median')); names.append(f'{x}_{q}_median')
    for x in ('NDVI', 'EVI'):
        bands.append(qmed['Q3'].select(x).subtract(qmed['Q1'].select(x)).rename(f'{x}_greenup'))
        names.append(f'{x}_greenup')
    assert names == s2_names() and len(set(names)) == 84
    return ee.Image.cat(bands).toFloat(), names


def ae_image(ee, year, region):
    col = (ee.ImageCollection(K.AE_COLL).filterDate(f'{year}-01-01', f'{year + 1}-01-01').filterBounds(region))
    return col.mosaic().select(AE_BANDS).toFloat(), AE_BANDS, col


def reducer(ee):
    return ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)


def out_columns(names):
    return [f'{n}_mean' for n in names] + [f'{n}_stdDev' for n in names]


# ---------------------------------------------------------------- 几何
def t1_fc(ee, sub):
    ep = int(sub.epsg.iloc[0])
    proj = ee.Projection(f'EPSG:{ep}')
    feats = [ee.Feature(ee.Geometry.Rectangle([x - K.T1_HALF, y - K.T1_HALF, x + K.T1_HALF, y + K.T1_HALF], proj, False),
                        {'sid': int(r)}) for r, x, y in zip(sub.row, sub.x, sub.y_m)]
    return ee.FeatureCollection(feats), f'EPSG:{ep}'


def t2_fc(ee, ids):
    fc = ee.FeatureCollection(K.LUCAS_ASSET).filter(ee.Filter.inList('id', [int(i) for i in ids]))
    return fc.map(lambda f: ee.Feature(f.geometry(), {'sid': f.get('id')})), f'EPSG:{K.T2_EPSG}'


def bbox(ee, lon, lat, pad=0.02):
    return ee.Geometry.Rectangle([float(lon.min()) - pad, float(lat.min()) - pad, float(lon.max()) + pad,
                                  float(lat.max()) + pad], None, False)


def jobs(feat):
    """全部导出单元：(任务, 名称, 样本子表)。分块只影响计算，不影响结果（方案第 2 节）。
    AE：T1 按城市；T2 按国家再按 id 分块（每块 ≤ 2000）。
    S2：T1 按城市内 100 km 方格（该城 UTM）；T2 按 EPSG:3035 的 100 km 方格（不分国家）。"""
    t1 = pd.read_parquet(os.path.join(K.SPLITS, 't1_samples.parquet')).sort_values('row')
    t2 = pd.read_parquet(os.path.join(K.SPLITS, 't2_samples.parquet')).sort_values('id')
    out = []
    if feat == 'ae':
        for c in sorted(t1.city.unique()):
            out.append(('T1', c, t1[t1.city == c]))
        for c in sorted(t2.nuts0.unique()):
            s = t2[t2.nuts0 == c]
            for j, a in enumerate(range(0, len(s), CHUNK_T2)):
                out.append(('T2', f'{c}_{j:02d}', s.iloc[a:a + CHUNK_T2]))
    else:
        g1 = t1.assign(cx=np.floor(t1.x / CELL).astype(int), cy=np.floor(t1.y_m / CELL).astype(int))
        for (c, cx, cy), s in g1.groupby(['city', 'cx', 'cy'], sort=True):
            out.append(('T1', f'{c}_{cx}_{cy}', s))
        g2 = t2.assign(cx=np.floor(t2.x / CELL).astype(int), cy=np.floor(t2.y_m / CELL).astype(int))
        for (cx, cy), s in g2.groupby(['cx', 'cy'], sort=True):
            name = f'c{cx}_{cy}'
            if name in SUBDIVIDE['T2']:
                h = s.assign(hx=np.floor(s.x / (CELL / 2)).astype(int) % 2, hy=np.floor(s.y_m / (CELL / 2)).astype(int) % 2)
                for (hx, hy), q in h.groupby(['hx', 'hy'], sort=True):
                    out.append(('T2', f'{name}_q{hx}{hy}', q.drop(columns=['hx', 'hy'])))
            else:
                out.append(('T2', name, s))
    return out


def asset_id(task, feat, name):
    return f'{K.ASSET_ROOT}/{task.lower()}_{feat}_{name}'


def reduced(ee, task, feat, sub):
    if task == 'T1':
        fc, crs = t1_fc(ee, sub)
    else:
        fc, crs = t2_fc(ee, sub.id.values)
    region = bbox(ee, sub.lon, sub.lat)
    img, names, _ = ae_image(ee, K.YEAR[task], region) if feat == 'ae' else (*s2_composite(ee, K.YEAR[task], region), None)
    red = img.reduceRegions(collection=fc, reducer=reducer(ee), scale=10, crs=crs, tileScale=16)
    cols = out_columns(names)
    return red.map(lambda f: f.setGeometry(ee.Geometry.Point([0, 0]))), cols   # 资产表要求有几何：挂占位点


def ensure_folder(ee):
    try:
        ee.data.getAsset(K.ASSET_ROOT)
    except Exception:
        ee.data.createAsset({'type': 'FOLDER'}, K.ASSET_ROOT)


def cmd_submit(a):
    ee = ee_init()
    ensure_folder(ee)
    n = 0
    existing = {x['name'].split('/')[-1] for x in ee.data.listAssets({'parent': K.ASSET_ROOT}).get('assets', [])}
    for feat in K.FEATS:
        if a.feat and feat != a.feat:
            continue
        for task, name, sub in jobs(feat):
            if a.only and task != a.only:
                continue
            aid = asset_id(task, feat, name)
            if aid.split('/')[-1] in existing:
                continue
            if a.dry:
                print('would submit', aid, len(sub)); n += 1
                continue
            fc, _ = reduced(ee, task, feat, sub)
            t = ee.batch.Export.table.toAsset(collection=fc, description=f'nc_{task}_{feat}_{name}'[:100], assetId=aid)
            t.start()
            n += 1
            print('submitted', aid, len(sub), t.id)
    print('total', n)


def cmd_status(a):
    ee = ee_init()
    ts = [t for t in ee.data.getTaskList() if t.get('description', '').startswith('nc_T')]
    latest = {}
    for t in sorted(ts, key=lambda t: t.get('creation_timestamp_ms', 0)):
        latest[t['description']] = t
    from collections import Counter
    print(Counter(t['state'] for t in latest.values()))
    for t in latest.values():
        if t['state'] == 'FAILED':
            print('FAILED', t['description'], t.get('error_message', '')[:200])


def cmd_fetch(a):
    ee = ee_init()
    os.makedirs(K.FEATS_DIR, exist_ok=True)
    parts, n_part = {}, 0
    have = {x['name'].split('/')[-1] for x in ee.data.listAssets({'parent': K.ASSET_ROOT}).get('assets', [])}
    for feat in K.FEATS:
        for task, name, sub in jobs(feat):
            part = os.path.join(K.FEATS_DIR, 'parts', f'{task}_{feat}_{name}.parquet')
            if asset_id(task, feat, name).split('/')[-1] in have:
                df = ee.data.computeFeatures({'expression': ee.FeatureCollection(asset_id(task, feat, name)),
                                              'fileFormat': 'PANDAS_DATAFRAME'})
            else:
                df = pd.read_parquet(part)
                n_part += 1
            df = df.drop(columns=[c for c in ('geo',) if c in df.columns])
            assert len(df) == len(sub) and set(df.sid.astype(np.int64)) == set(sub['row' if task == 'T1' else 'id']), \
                (task, feat, name)
            parts.setdefault((task, feat), []).append(df)
    ver = {}
    for (task, feat), dfs in parts.items():
        df = pd.concat(dfs, ignore_index=True)
        cols = out_columns(AE_BANDS if feat == 'ae' else s2_names())   # 固定列序
        extra = set(df.columns) - set(cols) - {'sid'}
        assert not extra, extra
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan                              # 全部像元被掩膜时 GEE 不写该键
        df = df[['sid'] + cols].copy()
        df['sid'] = df.sid.astype(np.int64)
        df[cols] = df[cols].astype(np.float32)
        df.sort_values('sid').to_parquet(os.path.join(K.FEATS_DIR, f'{task}_{feat}.parquet'), index=False)
        print(task, feat, df.shape, 'all-NaN rows', int(df[cols].isna().all(axis=1).sum()))
    for task in K.TASKS:                                     # AE 版本属性
        col = ee.ImageCollection(K.AE_COLL).filterDate(f'{K.YEAR[task]}-01-01', f'{K.YEAR[task] + 1}-01-01')
        ver[task] = {p: col.aggregate_histogram(p).getInfo() for p in ('MODEL_VERSION', 'DATASET_VERSION')}
    ver['n_chunks_from_interactive_parts'] = n_part
    json.dump(ver, open(os.path.join(K.FEATS_DIR, 'versions.json'), 'w'), indent=1)
    print(ver)


def cmd_interactive(a):
    """对尚无资产的块：按样本编号顺序每 a.batch 个一组，交互计算（与批处理同一 reduced()），合并后存 part。"""
    import time
    ee = ee_init()
    have = {x['name'].split('/')[-1] for x in ee.data.listAssets({'parent': K.ASSET_ROOT}).get('assets', [])}
    os.makedirs(os.path.join(K.FEATS_DIR, 'parts'), exist_ok=True)
    for feat in K.FEATS:
        for task, name, sub in jobs(feat):
            part = os.path.join(K.FEATS_DIR, 'parts', f'{task}_{feat}_{name}.parquet')
            if asset_id(task, feat, name).split('/')[-1] in have or os.path.exists(part):
                continue
            t0, dfs = time.time(), []
            for i in range(0, len(sub), a.batch):
                fc, cols = reduced(ee, task, feat, sub.iloc[i:i + a.batch])
                dfs.append(ee.data.computeFeatures({'expression': fc, 'fileFormat': 'PANDAS_DATAFRAME'}))
            df = pd.concat(dfs, ignore_index=True)
            df = df.drop(columns=[c for c in ('geo',) if c in df.columns])
            idc = 'row' if task == 'T1' else 'id'
            assert len(df) == len(sub) and set(df.sid.astype(np.int64)) == set(sub[idc])
            df.to_parquet(part, index=False)
            print('interactive', task, feat, name, len(sub), f'{time.time() - t0:.0f}s', flush=True)


def s2_names():
    names = [f'{b}_median' for b in S2_BANDS] + [f'{b}_mean' for b in S2_BANDS] + [f'{b}_std' for b in S2_BANDS]
    names += [f'{x}_{s}' for s in ('median', 'mean', 'std', 'max') for x in VI]
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        names += [f'{x}_{q}_{s}' for x in ('NDVI', 'EVI') for s in ('median', 'std')]
    names += [f'{x}_{q}_median' for q in ('Q1', 'Q2', 'Q3', 'Q4') for x in ('B4', 'B8', 'B11')]
    names += ['NDVI_greenup', 'EVI_greenup']
    return names


def cmd_probe(a):
    """小样本交互测试：只检查列、空值率与耗时，不读标签。"""
    import time
    ee = ee_init()
    for feat in K.FEATS:
        for task, name, sub in jobs(feat):
            if not (task == a.task and name.startswith(a.dom)):
                continue
            sub = sub.head(a.n)
            if True:
                t = time.time()
                fc, cols = reduced(ee, task, feat, sub)
                df = ee.data.computeFeatures({'expression': fc, 'fileFormat': 'PANDAS_DATAFRAME'})
                miss = [c for c in cols if c not in df.columns]
                print(task, name, feat, df.shape, f'{time.time() - t:.1f}s', 'missing cols', len(miss),
                      'NaN frac', float(df.reindex(columns=cols).isna().mean().mean()))
            break


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('submit'); s.add_argument('--only'); s.add_argument('--feat'); s.add_argument('--dry', action='store_true')
    sp.add_parser('status')
    sp.add_parser('fetch')
    q = sp.add_parser('interactive'); q.add_argument('--batch', type=int, default=20)
    p = sp.add_parser('probe'); p.add_argument('--task', default='T1'); p.add_argument('--dom', default='zurich')
    p.add_argument('--n', type=int, default=20)
    a = ap.parse_args()
    {'submit': cmd_submit, 'status': cmd_status, 'fetch': cmd_fetch, 'probe': cmd_probe,
     'interactive': cmd_interactive}[a.cmd](a)


if __name__ == '__main__':
    main()
