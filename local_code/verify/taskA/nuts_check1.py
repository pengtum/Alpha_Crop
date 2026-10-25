import pandas as pd, numpy as np, geopandas as gpd
L='/Users/perng_mini/AlphaCrops_local/'
pn=pd.read_parquet(L+'data/parcel_nuts.parquet')
ds=pd.read_parquet(L+'data/dataset_v2.parquet',columns=['parcel_id','country','longitude','latitude','has_ae','has_s2'])
print('rows', len(pn), len(ds), 'pid unique pn', pn.parcel_id.is_unique, 'ds', ds.parcel_id.is_unique)
print('same order', (pn.parcel_id.values==ds.parcel_id.values).all(), 'country same', (pn.country.values==ds.country.values).all())
print('nuts3 na', pn.nuts3.isna().sum())
print('prefix!=country', (pn.nuts3.str[:2]!=pn.country).sum())
print(pn.join_method.value_counts())
nuts=gpd.read_file(L+'data/boundaries/NUTS_RG_01M_2021_4326_LEVL_3.geojson')
print(nuts.columns.tolist(), len(nuts))
off=nuts.groupby('CNTR_CODE').size()
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
ds2=ds[ds.has_ae&ds.has_s2]
pn2=pn[pn.parcel_id.isin(ds2.parcel_id)]
t=pn.groupby('country').agg(n=('nuts3','size'),nuts3=('nuts3','nunique'),nuts2=('nuts2','nunique'),nuts1=('nuts1',lambda s: ','.join(sorted(s.unique()))),nearest=('join_method',lambda s:(s=='nearest').sum()))
t['geojson_nuts3']=off.reindex(t.index)
t['nuts3_aeS2']=pn2.groupby('country').nuts3.nunique()
t['n_aeS2']=pn2.groupby('country').size()
t['nearest_pct']=(t.nearest/t.n*100).round(3)
pd.set_option('display.width',250)
print(t)
# NUTS3 in geojson for countries but not used (within nuts1 present)
for c in C:
    used=set(pn.nuts3[pn.country==c]); n1=set(pn.nuts1[pn.country==c])
    allc=nuts[(nuts.CNTR_CODE==c)]
    inn1=allc[allc.NUTS_ID.str[:3].isin(n1)]
    miss=sorted(set(inn1.NUTS_ID)-used)
    print(c,'unused NUTS3 within covered NUTS1:',[(m, inn1.set_index('NUTS_ID').loc[m,'NUTS_NAME']) for m in miss])
