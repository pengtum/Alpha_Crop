import os, pandas as pd
D='/Users/perng_mini/AlphaCrops_local/data'
df=pd.read_parquet(os.path.join(D,'dataset_v2.parquet'), columns=['parcel_id','country','EC_hcat_n','label_v1','label','has_ae','has_s2'])
na=df[df.EC_hcat_n.isna()]
print('NaN hcat by country', na.country.value_counts().to_dict(), 'labels', na.label.value_counts(dropna=False).to_dict(), 'label_v1', na.label_v1.value_counts(dropna=False).to_dict(), 'both', (na.has_ae&na.has_s2).sum())
b=df[df.has_ae&df.has_s2]
print('RO hcat top'); print(b[b.country=='RO'].groupby(['EC_hcat_n','label']).size().sort_values(ascending=False).head(25).to_string())
p=pd.read_parquet(os.path.join(D,'parcels_eu.parquet'), columns=['parcel_id','country','EC_hcat_n','EC_trans_n','label_tier_b'])
ro=p[p.country=='RO']
print('RO parcels_eu', len(ro)); print(ro.EC_hcat_n.value_counts(dropna=False).head(15).to_string())
print(p[p.EC_hcat_n.isna()].groupby('country').size().to_dict())
print(p[p.EC_hcat_n.isna()&(p.country=='RO')].EC_trans_n.value_counts().head(10))
