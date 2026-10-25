import pandas as pd, numpy as np, geopandas as gpd
from shapely import contains_xy
L='/Users/perng_mini/AlphaCrops_local/'
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
pn=pd.read_parquet(L+'data/parcel_nuts.parquet')
ds=pd.read_parquet(L+'data/dataset_v2.parquet',columns=['parcel_id','country','longitude','latitude'])
df=ds.merge(pn[['parcel_id','nuts3','join_method']],on='parcel_id',validate='one_to_one')
nuts=gpd.read_file(L+'data/boundaries/NUTS_RG_01M_2021_4326_LEVL_3.geojson')
allnuts=nuts.copy()
geom=nuts.set_index('NUTS_ID').geometry
# 1) independent check: within rows -> point inside assigned polygon
w=df[df.join_method=='within']
ok=np.zeros(len(w),bool)
for nid,idx in w.groupby('nuts3').indices.items():
    sub=w.iloc[idx]
    ok[idx]=contains_xy(geom[nid], sub.longitude.values, sub.latitude.values)
print('within rows inside assigned polygon:', ok.sum(), '/', len(w), ' (boundary-touch may be False)')
if (~ok).any(): print(w[~ok].head())
# 2) nearest rows: distances in LAEA
nr=df[df.join_method=='nearest'].copy()
pts=gpd.GeoDataFrame(nr,geometry=gpd.points_from_xy(nr.longitude,nr.latitude),crs=4326).to_crs(3035)
g3035=allnuts.to_crs(3035)
gi=g3035.set_index('NUTS_ID').geometry
nr['d_assigned_m']=[p.distance(gi[n]) for p,n in zip(pts.geometry,nr.nuts3)]
# which polygon (any country in full geojson) contains the point
j=gpd.sjoin(pts,g3035[['NUTS_ID','CNTR_CODE','geometry']],how='left',predicate='within')
j=j[~j.index.duplicated()]
nr['in_poly']=j['NUTS_ID'].values; nr['in_cntr']=j['CNTR_CODE'].values
# distance to own country outline (union) boundary
own={c:g3035[g3035.CNTR_CODE==c].union_all() for c in C}
nr['d_own_boundary_m']=[own[c].boundary.distance(p) for p,c in zip(pts.geometry,nr.country)]
# coastal flag: distance to any non-own land polygon
land=g3035.union_all()
nr['in_any_land']=[land.contains(p) for p in pts.geometry]
nr['d_land_m']=[0 if inl else land.distance(p) for p,inl in zip(pts.geometry,nr.in_any_land)]
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
print(nr['d_assigned_m'].describe(percentiles=[.5,.9,.99]).round(0))
print('by country: n, max d_assigned, n in foreign polygon, n outside all land')
print(nr.groupby('country').agg(n=('parcel_id','size'),dmax=('d_assigned_m','max'),dmed=('d_assigned_m','median'),
      in_foreign=('in_cntr',lambda s:s.notna().sum()),outside_land=('in_any_land',lambda s:(~s).sum())).round(0).to_string())
print('in-foreign breakdown:'); print(nr.groupby(['country','in_cntr']).size().to_string())
nr.to_csv('nearest_all.csv',index=False)
s=nr.sample(10,random_state=0)[['parcel_id','country','longitude','latitude','nuts3','in_poly','in_cntr','in_any_land','d_assigned_m','d_own_boundary_m','d_land_m']]
print(s.round(4).to_string())
print('largest 10 d_assigned:')
print(nr.sort_values('d_assigned_m').tail(10)[['parcel_id','country','longitude','latitude','nuts3','in_poly','in_cntr','d_assigned_m','d_own_boundary_m']].round(4).to_string())
