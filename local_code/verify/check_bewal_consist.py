import os, pandas as pd, numpy as np, lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score, accuracy_score
D='/Users/perng_mini/AlphaCrops_local/data'
a=pd.read_parquet(os.path.join(D,'ae','embeddings_BE_WAL.parquet'), columns=['parcel_id','parcel_idx'])
idxset=set(a.parcel_idx.dropna())
import pyarrow.parquet as pq
cols=[c for c in pq.ParquetFile(os.path.join(D,'dataset_v2.parquet')).schema_arrow.names if c.startswith('ae_') or c.startswith('s2_')]
ds=pd.read_parquet(os.path.join(D,'dataset_v2.parquet'), columns=['parcel_id','label','has_ae','has_s2']+cols, filters=[('country','==','BE')])
ds=ds[ds.parcel_id.str.startswith('BE_WAL')&ds.has_ae&ds.has_s2].reset_index(drop=True)
is9=ds.parcel_id.isin(idxset).to_numpy()
print('BE_WAL both', len(ds), 'batch9', is9.sum())
rng=np.random.default_rng(0)
other=np.flatnonzero(~is9); rng.shuffle(other)
ctrl=other[:5000]; tr=other[5000:]
ae=[c for c in cols if c.startswith('ae_')]; s2=[c for c in cols if c.startswith('s2_')]
y=ds.label.to_numpy()
for name,fc in [('AE',ae),('S2',s2)]:
    m=lgb.LGBMClassifier(n_estimators=150,num_leaves=31,n_jobs=2,verbose=-1,random_state=0).fit(ds.loc[tr,fc].to_numpy(np.float32),y[tr])
    for nm,ix in [('control',ctrl),('batch9',np.flatnonzero(is9))]:
        pr=m.predict(ds.loc[ix,fc].to_numpy(np.float32))
        print(name,nm,'acc %.3f bal %.3f'%(accuracy_score(y[ix],pr),balanced_accuracy_score(y[ix],pr)))
# AE->S2 cross-modal: predict an S2 feature from AE (ridge)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
tcol=[c for c in s2 if 'B8' in c or 'NDVI' in c.upper()][:3]
print('targets', tcol)
for t in tcol:
    rr=Ridge(1.0).fit(ds.loc[tr,ae],ds.loc[tr,t])
    print(t,'R2 control %.3f batch9 %.3f'%(r2_score(ds.loc[ctrl,t],rr.predict(ds.loc[ctrl,ae])), r2_score(ds.loc[is9,t],rr.predict(ds.loc[is9,ae]))))
