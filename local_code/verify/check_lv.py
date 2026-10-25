import os, pandas as pd, numpy as np, glob
D='/Users/perng_mini/AlphaCrops_local/data'
s=pd.read_parquet(os.path.join(D,'s2','s2_LV.parquet'))
at=[c for c in pd.read_parquet(os.path.join(D,'s2','s2_AT.parquet')).columns if c!='parcel_id']
std_ok=s[at].notna().all(axis=1); std_any=s[at].notna().any(axis=1)
extra=[c for c in s.columns if c not in at and c!='parcel_id']
print('LV s2 rows', len(s), 'std complete', std_ok.sum(), 'std any nonnull but incomplete', (std_any&~std_ok).sum(), 'extra nonnull rows', s[extra].notna().any(axis=1).sum())
print('extra non-B cols', [c for c in extra if not c.startswith('B') and not c[0].isupper()][:10], [c for c in extra if '_' not in c])
print('dup pid', s.parcel_id.duplicated().sum(), 'pid null', s.parcel_id.isna().sum())
r=pd.read_csv(os.path.join(D,'raw','LV_s2','s2_LV_batch_0008.csv'), usecols=lambda c: c in ('parcel_id','parcel_idx','system:index','B11_Q1_median_mean','B11_Q1_median_mean_mean'))
print('raw 0008 rows', len(r), r.columns.tolist(), r.notna().sum().to_dict())
ids8=set(r.parcel_idx); print('s2 parquet parcel_idx==raw', set(s.parcel_idx.dropna())==ids8)
print('0008 ids: rows in s2 parquet', s.parcel_id.isin(ids8).sum(), 'of which std complete', (s.parcel_id.isin(ids8)&std_ok).sum())
print('non-0008 rows with std incomplete', (~s.parcel_id.isin(ids8)&~std_ok).sum())
ds=pd.read_parquet(os.path.join(D,'dataset_v2.parquet'), columns=['parcel_id','country','has_ae','has_s2','label'], filters=[('country','==','LV')])
print('LV in ds', len(ds), 'has_s2', ds.has_s2.sum())
print('0008 ids in ds', ds.parcel_id.isin(ids8).sum(), 'has_s2 among them', ds[ds.parcel_id.isin(ids8)].has_s2.sum())
nos2=ds[~ds.has_s2]
print('LV has_s2=False', len(nos2), 'of which in 0008', nos2.parcel_id.isin(ids8).sum(), 'not in s2 parquet at all', (~nos2.parcel_id.isin(set(s.parcel_id))).sum())
print('label dist LV all vs 0008 vs rest-has_s2:')
print(pd.concat({'all':ds.label.value_counts(normalize=True),'b0008':ds[ds.parcel_id.isin(ids8)].label.value_counts(normalize=True),'has_s2':ds[ds.has_s2].label.value_counts(normalize=True)},axis=1).round(3).to_string())
# other raw batches: count
for f in sorted(glob.glob(os.path.join(D,'raw','LV_s2','*.csv'))):
    h=pd.read_csv(f,nrows=1)
    print(os.path.basename(f), len(h.columns), 'B11_Q1_median_mean' in h.columns)
