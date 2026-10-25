import os, pandas as pd, numpy as np
D='/Users/perng_mini/AlphaCrops_local/data'
a=pd.read_parquet(os.path.join(D,'ae','embeddings_BE_WAL.parquet'), columns=['parcel_id','parcel_idx'])
a['pid']=a.parcel_id.fillna(a.parcel_idx); a['batch']=np.arange(len(a))//5000
p=pd.read_parquet(os.path.join(D,'parcels_eu.parquet'), columns=['parcel_id','EC_hcat_n','label_tier_b','area_ha','longitude','latitude','split'])
m=a.merge(p, left_on='pid', right_on='parcel_id', how='left', suffixes=('_ae',''))
print(pd.crosstab(m.batch, m.label_tier_b).to_string())
print(m.groupby('batch')[['area_ha','longitude','latitude']].median())
print(pd.crosstab(m.batch, m.split))
