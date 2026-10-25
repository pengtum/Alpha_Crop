"""GData 缓存与原始 parquet 的一致性（只读）：标签、国家、坐标、面积全量比对；特征随机抽 20 列全量比对。"""
import json
import os

import numpy as np
import pyarrow.parquet as pq

CACHE = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
meta = json.load(open(os.path.join(CACHE, 'meta.json')))
path = meta['source'].split('|')[0]
st = os.stat(path)
print('source stamp ok:', meta['source'] == f'{path}|{st.st_size}|{int(st.st_mtime)}')
pf = pq.ParquetFile(path)
names = pf.schema_arrow.names
t = pq.read_table(path, columns=['label', 'country', 'longitude', 'latitude', 'area_ha', 'has_ae', 'has_s2'])
keep = np.asarray(t['has_ae']) & np.asarray(t['has_s2'])
print('parquet rows', t.num_rows, 'kept', keep.sum())
lab = np.asarray(t['label'].to_pylist(), dtype=object)[keep]
classes = sorted(set(lab))
print('classes eq', classes == meta['classes'])
ymap = {c: i for i, c in enumerate(classes)}
y = np.array([ymap[c] for c in lab])
print('y eq', np.array_equal(y, np.load(os.path.join(CACHE, 'y.npy'))))
print('ctry eq', np.array_equal(np.asarray(t['country'].to_pylist())[keep].astype('U2'), np.load(os.path.join(CACHE, 'ctry.npy'))))
for a, b in (('longitude', 'lon'), ('latitude', 'lat'), ('area_ha', 'area')):
    print(b, 'eq', np.array_equal(np.asarray(t[a])[keep], np.load(os.path.join(CACHE, f'{b}.npy'))))
ae_cols = [c for c in names if c.startswith('ae_')]
s2_cols = [c for c in names if c.startswith('s2_')]
print('cols eq meta', ae_cols == meta['cols']['ae'], s2_cols == meta['cols']['s2'])
rng = np.random.default_rng(123)
for f, cols in (('ae', ae_cols), ('s2', s2_cols)):
    X = np.load(os.path.join(CACHE, f'X_{f}.npy'), mmap_mode='r')
    js = rng.choice(len(cols), 10, replace=False)
    tt = pq.read_table(path, columns=[cols[j] for j in js])
    ok = []
    for j in js:
        v = np.asarray(tt[cols[j]]).astype(np.float32)[keep]
        ok.append(np.array_equal(v, X[:, j], equal_nan=True))
    print(f, 'feature cols checked', [cols[j] for j in js][:3], '... all eq:', all(ok), ok)
    print(f, 'NaN in cache', int(np.isnan(np.asarray(X[:50000])).sum()))
