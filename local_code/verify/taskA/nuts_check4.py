import pandas as pd, numpy as np, geopandas as gpd
L='/Users/perng_mini/AlphaCrops_local/'
pn=pd.read_parquet(L+'data/parcel_nuts.parquet')
ds=pd.read_parquet(L+'data/dataset_v2.parquet',columns=['parcel_id','country','longitude','latitude'])
df=ds.merge(pn[['parcel_id','nuts3','nuts1','join_method']],on='parcel_id')
nuts=gpd.read_file(L+'data/boundaries/NUTS_RG_01M_2021_4326_LEVL_3.geojson').to_crs(3035)
for c in ['PT','SE','SK','LV']:
    print(c, df[df.country==c].groupby('nuts3').size().to_dict())
# small / odd groups: distance to the main region
def dist_to(sub, ids):
    u=nuts[nuts.NUTS_ID.isin(ids)].union_all()
    p=gpd.GeoSeries(gpd.points_from_xy(sub.longitude,sub.latitude),crs=4326).to_crs(3035)
    return np.round([u.distance(x) for x in p],0)
pt=df[(df.country=='PT')]
main_pt=pt.nuts3.value_counts(); main_ids=main_pt[main_pt>100].index
sub=pt[pt.nuts3.isin(['PT150','PT16G'])]
print(sub[['parcel_id','longitude','latitude','nuts3','join_method']].assign(d_to_big_groups_m=dist_to(sub,main_ids)).to_string())
de=df[(df.country=='DE')&~df.nuts1.isin(['DE9','DEA','DE5'])]
d=dist_to(de,[i for i in nuts.NUTS_ID if i[:3] in ('DE9','DEA')])
print('DE parcels outside NI/NRW/Bremen:',len(de),'dist to NI+NRW (m): max',d.max(),'median',np.median(d))
be=df[df.nuts3=='BE100']; print('BE100 parcels regions', be.parcel_id.str[:6].value_counts().to_dict())
