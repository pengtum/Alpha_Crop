import sys, os
import pandas as pd, numpy as np
CODE="/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
sys.dont_write_bytecode = True
import common
from config.class_definitions import TIER_B_MAPPING, TIER_B_CLASSES
D='/Users/perng_mini/AlphaCrops_local/data'
meta=['parcel_id','country','longitude','latitude','EC_hcat_n','label_v1','label','has_ae','has_s2']
df=pd.read_parquet(os.path.join(D,'dataset_v2.parquet'), columns=meta)
print('rows', len(df), 'dup pid', df.parcel_id.duplicated().sum())
print('TIER_B_CLASSES', TIER_B_CLASSES)
print('labels', sorted(df.label.unique()), df.label.nunique())
print('values of TIER_B_MAPPING not in classes', set(TIER_B_MAPPING.values())-set(TIER_B_CLASSES))
print('ADDITIONS overriding existing keys:', {k:(TIER_B_MAPPING[k],v) for k,v in common.TIER_B_ADDITIONS.items() if k in TIER_B_MAPPING and TIER_B_MAPPING[k]!=v})
print('ADDITIONS keys already in orig (same)', [k for k in common.TIER_B_ADDITIONS if k in TIER_B_MAPPING])
# recompute mapping
u=df[['EC_hcat_n','label']].drop_duplicates()
u['re']=u.EC_hcat_n.map(common.tier_b_v2)
print('mismatch mapping rows', (u.re!=u.label).sum()); print(u[u.re!=u.label].head(20))
print('EC_hcat_n NaN', df.EC_hcat_n.isna().sum())
print('excluded present', df.EC_hcat_n.isin(common.EXCLUDE_V2).sum())
# which hcat names in dataset map to other_crops via fallback (not in mapping)
fb=df[~df.EC_hcat_n.isin(common.TIER_B_MAPPING_V2.keys())]
print('fallback other_crops parcels', len(fb)); print(fb.EC_hcat_n.value_counts().head(30).to_string())
# label_v1 check vs orig mapping
u1=df[['EC_hcat_n','label_v1']].drop_duplicates(); u1['re']=u1.EC_hcat_n.map(lambda h: TIER_B_MAPPING.get(h,'other_crops'))
print('label_v1 mismatch vs orig mapping', (u1.re!=u1.label_v1).sum()); print(u1[u1.re!=u1.label_v1].head(20).to_string())
print(pd.crosstab(df.country,[df.has_ae,df.has_s2]).to_string())
both=df[df.has_ae&df.has_s2]; print('both', len(both))
print(pd.crosstab(both.country, both.label).to_string())
# hcat names mapping to each class in both
g=both.groupby(['label','EC_hcat_n']).size().sort_values(ascending=False)
for lab in sorted(both.label.unique()):
    print(lab, g.loc[lab].head(12).to_dict())
