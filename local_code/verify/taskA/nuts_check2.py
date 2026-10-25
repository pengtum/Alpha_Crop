import pandas as pd, numpy as np, geopandas as gpd
L='/Users/perng_mini/AlphaCrops_local/'
pn=pd.read_parquet(L+'data/parcel_nuts.parquet')
ds=pd.read_parquet(L+'data/dataset_v2.parquet',columns=['parcel_id','country','longitude','latitude','has_ae','has_s2'])
nuts=gpd.read_file(L+'data/boundaries/NUTS_RG_01M_2021_4326_LEVL_3.geojson')
off=nuts.groupby('CNTR_CODE').size()
t=pn.groupby('country').agg(n=('nuts3','size'),nuts3=('nuts3','nunique'),nuts2=('nuts2','nunique'),nearest=('join_method',lambda s:(s=='nearest').sum()))
t['geojson_nuts3']=off.reindex(t.index)
print(t.to_string())
for c in ['DE','BE','RO','LV','PT','SE','EE','SK','SI','LT','AT']:
    s=pn[pn.country==c]
    print(c, s.groupby('nuts1').size().to_dict())
# DE by nuts2
s=pn[pn.country=='DE']
print(s.groupby('nuts2').size().sort_values().to_string())
# NUTS3 with tiny counts
vc=pn.groupby(['country','nuts3']).size()
print('NUTS3 with <30 parcels:'); print(vc[vc<30].to_string())
