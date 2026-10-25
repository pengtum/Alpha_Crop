import os, pandas as pd, numpy as np
D='/Users/perng_mini/AlphaCrops_local/data'
a=pd.read_parquet(os.path.join(D,'ae','embeddings_BE_WAL.parquet'))
print(a.shape, a.parcel_id.isna().sum(), a.parcel_idx.notna().sum(), a.parcel_idx.dtype)
print('both set', (a.parcel_id.notna()&a.parcel_idx.notna()).sum(), 'neither', (a.parcel_id.isna()&a.parcel_idx.isna()).sum())
print(a.parcel_id.dropna().head(3).tolist(), a.parcel_idx.dropna().head(3).tolist())
pid=a.parcel_id.fillna(a.parcel_idx)
print('dup after fill', pid.duplicated().sum(), 'dup among nonnull', pid.dropna().duplicated().sum())
p=pd.read_parquet(os.path.join(D,'parcels_eu.parquet'), columns=['parcel_id','country'])
pset=set(p.parcel_id)
idx_rows=a[a.parcel_id.isna()]
print('parcel_idx rows', len(idx_rows), 'in parcels', idx_rows.parcel_idx.isin(pset).sum())
print('parcel_idx overlapping with existing parcel_id', idx_rows.parcel_idx.isin(set(a.parcel_id.dropna())).sum())
print('all pid in parcels', pid.isin(pset).mean(), 'countries', p[p.parcel_id.isin(set(pid))].country.value_counts().to_dict())
# prefix check
print(pd.Series(pid.dropna().str.split('_').str[:2].str.join('_')).value_counts().to_dict())
# row position of idx rows (last batch?)
pos=np.flatnonzero(a.parcel_id.isna().to_numpy()); print('idx rows positions', pos.min(), pos.max(), len(a))
ae_cols=[f'A{i:02d}_{s}' for i in range(64) for s in ('mean','stdDev')]
print('incomplete among idx rows', idx_rows[ae_cols].isna().any(axis=1).sum(), 'incomplete overall', a[ae_cols].isna().any(axis=1).sum())
# compare against raw batch 0009 CSV
r=pd.read_csv(os.path.join(D,'raw','BE_WAL_ae','ae_BE_WAL_batch_0009.csv'), nrows=5)
print('raw 0009 cols', r.columns[:5].tolist(), r.columns[-5:].tolist(), len(r.columns))
r8=pd.read_csv(os.path.join(D,'raw','BE_WAL_ae','ae_BE_WAL_batch_0008.csv'), nrows=2)
print('raw 0008 cols', r8.columns[:5].tolist(), r8.columns[-5:].tolist(), len(r8.columns))
# dataset_v2: BE_WAL parcels via idx - has_ae?
ds=pd.read_parquet(os.path.join(D,'dataset_v2.parquet'), columns=['parcel_id','country','has_ae','has_s2'])
m=ds[ds.parcel_id.isin(set(idx_rows.parcel_idx))]
print('idx parcels in dataset_v2', len(m), m.has_ae.mean(), m.has_s2.mean())
bew=ds[ds.parcel_id.str.startswith('BE_WAL')]
print('BE_WAL in ds', len(bew), bew.has_ae.sum(), bew.has_s2.sum())
