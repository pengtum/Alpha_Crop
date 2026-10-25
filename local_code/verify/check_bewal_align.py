import os, pandas as pd, numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
D='/Users/perng_mini/AlphaCrops_local/data'
a=pd.read_parquet(os.path.join(D,'ae','embeddings_BE_WAL.parquet'), columns=['parcel_id','parcel_idx'])
idxset=set(a.parcel_idx.dropna())
cols=[c for c in pq.ParquetFile(os.path.join(D,'dataset_v2.parquet')).schema_arrow.names if c.startswith('ae_') or c.startswith('s2_')]
ds=pd.read_parquet(os.path.join(D,'dataset_v2.parquet'), columns=['parcel_id','label','has_ae','has_s2']+cols, filters=[('country','==','BE')])
ds=ds[ds.has_ae&ds.has_s2].reset_index(drop=True)
ae=[c for c in cols if c.startswith('ae_')]; s2=[c for c in cols if c.startswith('s2_')]
S=ds[s2].to_numpy(); S=(S-S.mean(0))/S.std(0)
A=ds[ae].to_numpy()
is9=ds.parcel_id.isin(idxset).to_numpy()
wc=(ds.label=='winter_cereals').to_numpy()
wal=ds.parcel_id.str.startswith('BE_WAL').to_numpy()
rng=np.random.default_rng(1)
vlg_wc=np.flatnonzero(wc&~wal); rng.shuffle(vlg_wc)
wal_wc=np.flatnonzero(wc&wal&~is9); rng.shuffle(wal_wc)
hold_vlg=vlg_wc[:1500]; hold_wal=wal_wc[:400]
test9=np.flatnonzero(is9)
mask=np.ones(len(ds),bool); mask[np.r_[hold_vlg,hold_wal,test9]]=False
# train on everything else (includes remaining winter cereals)
r=Ridge(alpha=10).fit(A[mask],S[mask])
def r2(ix, perm=False):
    P=r.predict(A[ix]); Y=S[ix]
    if perm: Y=Y[rng.permutation(len(ix))]
    return np.mean(1-((Y-P)**2).sum(0)/((Y-Y.mean(0))**2).sum(0))
for nm,ix in [('VLG winter_cereals holdout',hold_vlg),('WAL b0-8 winter_cereals holdout',hold_wal),('WAL batch9 (parcel_idx)',test9)]:
    print(f'{nm:35s} n={len(ix):5d} mean R2 true={r2(ix):.3f}  permuted={r2(ix,True):.3f}')
# per-parcel rank test on batch9: distance of predicted S2 to own S2 vs to random other batch9 parcel
P=r.predict(A[test9]); Y=S[test9]
own=((P-Y)**2).sum(1)
oth=((P-Y[rng.permutation(len(test9))])**2).sum(1)
print('batch9 frac own-dist < random-other-dist', (own<oth).mean())
P2=r.predict(A[hold_vlg]); Y2=S[hold_vlg]
print('VLG ctrl frac own<other', (((P2-Y2)**2).sum(1) < ((P2-Y2[rng.permutation(len(hold_vlg))])**2).sum(1)).mean())
# test for off-by-k shift within batch9 order (as stored in AE file)
order=a.parcel_idx.dropna().tolist()
pos={p:i for i,p in enumerate(order)}
t9=ds.loc[test9,'parcel_id'].map(pos).to_numpy(); o=np.argsort(t9)
Ps, Ys = P[o], Y[o]
for k in [-2,-1,0,1,2]:
    if k>=0: e=((Ps[k:]-Ys[:len(Ys)-k])**2).sum(1).mean() if k>0 else ((Ps-Ys)**2).sum(1).mean()
    else: e=((Ps[:k]-Ys[-k:])**2).sum(1).mean()
    print('shift',k,'mean sq err',round(e,2))
