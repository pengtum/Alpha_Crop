import os, json, pandas as pd, numpy as np
D='/Users/perng_mini/AlphaCrops_local/data'
p=pd.read_parquet(os.path.join(D,'parcels_eu.parquet'), columns=['parcel_id','longitude','latitude']).set_index('parcel_id')
def centroid(g):
    g=json.loads(g)
    if g['type']=='Point': return g['coordinates']
    c=g['coordinates']
    arr=np.array(c[0] if g['type']=='Polygon' else c[0][0])
    return arr[:,0].mean(), arr[:,1].mean()
for b,col in [('0008','parcel_id'),('0009','parcel_idx')]:
    r=pd.read_csv(os.path.join(D,'raw','BE_WAL_ae',f'ae_BE_WAL_batch_{b}.csv'), usecols=[col,'.geo','A00_mean'])
    print(b, len(r), json.loads(r['.geo'].iloc[0])['type'])
    xy=np.array([centroid(g) for g in r['.geo']])
    q=p.loc[r[col]]
    d=np.hypot(xy[:,0]-q.longitude.to_numpy(), (xy[:,1]-q.latitude.to_numpy()))*111
    print(f'  batch {b} ({col}): median dist km {np.median(d):.3f}, p95 {np.percentile(d,95):.3f}, max {d.max():.3f}, frac<0.5km {(d<0.5).mean():.4f}')
    # compare with embeddings parquet values
    a=pd.read_parquet(os.path.join(D,'ae','embeddings_BE_WAL.parquet'), columns=['parcel_id','parcel_idx','A00_mean'])
    a['pid']=a.parcel_id.fillna(a.parcel_idx)
    mm=r.merge(a[['pid','A00_mean']], left_on=col, right_on='pid', suffixes=('_raw','_pq'))
    print('  matched', len(mm), 'max abs diff A00_mean', np.abs(mm.A00_mean_raw-mm.A00_mean_pq).max())
