import pandas as pd, numpy as np, geopandas as gpd
L='/Users/perng_mini/AlphaCrops_local/'
nr=pd.read_csv(L+'verify/taskA/nearest_all.csv')
f=nr[nr.in_cntr.notna()]
print('nearest & inside foreign-country polygon: n=',len(f),' max d_assigned (m)=',f.d_assigned_m.max().round(0))
print(f.groupby('country').d_assigned_m.max().round(0).to_dict())
o=nr[nr.in_cntr.isna()]
print('nearest & outside all land: n=',len(o),' by country',o.groupby('country').size().to_dict(),' max d (m)',o.d_assigned_m.max().round(0))
print(nr[(nr.in_cntr==nr.country)][['parcel_id','nuts3','in_poly','d_assigned_m']])
nuts=gpd.read_file(L+'data/boundaries/NUTS_RG_01M_2021_4326_LEVL_3.geojson')
pn=pd.read_parquet(L+'data/parcel_nuts.parquet')
de=pn[pn.country=='DE']
for n1 in ['DE9','DEA','DE5']:
    print(n1,'geojson NUTS3',(nuts.NUTS_ID.str[:3]==n1).sum(),'used',de[de.nuts1==n1].nuts3.nunique())
