import os, pandas as pd, numpy as np
import pyarrow.parquet as pq
D='/Users/perng_mini/AlphaCrops_local/data'
names=pq.ParquetFile(os.path.join(D,'dataset_v2.parquet')).schema_arrow.names
at=[c for c in pq.ParquetFile(os.path.join(D,'s2','s2_AT.parquet')).schema_arrow.names if c!='parcel_id']
print('s2 cols in ds == AT order', [c[3:] for c in names if c.startswith('s2_')]==at, len(at))
ae_std=[f'A{i:02d}_{s}' for i in range(64) for s in ('mean','stdDev')]
print('ae cols in ds == standard', [c[3:] for c in names if c.startswith('ae_')]==ae_std)
ds=pd.read_parquet(os.path.join(D,'dataset_v2.parquet'), columns=['parcel_id','country','has_ae','has_s2']+[c for c in names if c.startswith('ae_') or c.startswith('s2_')])
both=ds[ds.has_ae&ds.has_s2]
rng=np.random.default_rng(7)
REGMAP={'AT':['AT'],'BE':['BE_VLG','BE_WAL'],'DE':['DE_LS','DE_NRW'],'EE':['EE'],'LT':['LT'],'LV':['LV'],'PT':['PT'],'RO':['RO'],'SE':['SE'],'SI':['SI'],'SK':['SK']}
pick=[]
for c in ['AT','BE','LV','RO','SK']:
    g=both[both.country==c]; pick.append(g.iloc[rng.integers(len(g))])
# plus 2 BE_WAL batch9 parcels
a9=pd.read_parquet(os.path.join(D,'ae','embeddings_BE_WAL.parquet'), columns=['parcel_idx']).parcel_idx.dropna()
for pid in a9.sample(2,random_state=3): 
    r=both[both.parcel_id==pid]
    if len(r): pick.append(r.iloc[0])
for row in pick:
    pid=row.parcel_id; reg=[r for r in REGMAP[row.country] if pid.startswith(r+'_')][0]
    a=pd.read_parquet(os.path.join(D,'ae',f'embeddings_{reg}.parquet'))
    if 'parcel_idx' in a: a['parcel_id']=a.parcel_id.fillna(a.parcel_idx)
    ar=a[a.parcel_id==pid]
    s=pd.read_parquet(os.path.join(D,'s2',f's2_{reg}.parquet'), columns=['parcel_id']+at)
    sr=s[s.parcel_id==pid]
    da=np.abs(ar[ae_std].to_numpy(float)[0]-row[['ae_'+c for c in ae_std]].to_numpy(float)).max()
    dsv=np.abs(sr[at].to_numpy(float)[0]-row[['s2_'+c for c in at]].to_numpy(float)).max()
    print(pid, reg, 'n_ae_rows', len(ar), 'n_s2_rows', len(sr), 'max|dAE|', da, 'max|dS2|', dsv)
# duplicates in raw files with conflicting values?
for reg in ['AT','BE_VLG','BE_WAL','DE_LS','DE_NRW','EE','LT','LV','PT','RO','SE','SI','SK']:
    a=pd.read_parquet(os.path.join(D,'ae',f'embeddings_{reg}.parquet'), columns=['parcel_id']+(['parcel_idx'] if reg=='BE_WAL' else []))
    if reg=='BE_WAL': a['parcel_id']=a.parcel_id.fillna(a.parcel_idx)
    s=pd.read_parquet(os.path.join(D,'s2',f's2_{reg}.parquet'), columns=['parcel_id'])
    print(reg,'AE dup', a.parcel_id.dropna().duplicated().sum(), 'AE null', a.parcel_id.isna().sum(), 'S2 dup', s.parcel_id.dropna().duplicated().sum(), 'S2 null', s.parcel_id.isna().sum())
