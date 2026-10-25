import os, pandas as pd, numpy as np
D='/Users/perng_mini/AlphaCrops_local/data'
p=pd.read_parquet(os.path.join(D,'parcels_eu.parquet'), columns=['parcel_id','label_tier_b','EC_hcat_n'])
REG=['AT','BE_VLG','BE_WAL','DE_LS','DE_NRW','EE','LT','LV','PT','RO','SE','SI','SK']
for r in REG:
    cols=['parcel_id']+(['parcel_idx'] if r=='BE_WAL' else [])
    a=pd.read_parquet(os.path.join(D,'ae',f'embeddings_{r}.parquet'), columns=cols)
    if r=='BE_WAL': a['parcel_id']=a.parcel_id.fillna(a.parcel_idx)
    a['batch']=np.arange(len(a))//5000
    m=a.merge(p,on='parcel_id',how='left')
    top=m.groupby('batch').label_tier_b.agg(lambda s: f"{s.value_counts(normalize=True).index[0]}:{s.value_counts(normalize=True).iloc[0]:.2f}")
    wc=m.groupby('batch').label_tier_b.apply(lambda s:(s=='winter_cereals').mean()).round(3)
    print(r, 'unmatched', m.label_tier_b.isna().sum(), 'wc frac by batch', wc.tolist())
    # also s2 file order
    s=pd.read_parquet(os.path.join(D,'s2',f's2_{r}.parquet'), columns=['parcel_id'])
    s['batch']=np.arange(len(s))//5000
    ms=s.merge(p,on='parcel_id',how='left')
    print('   s2 wc frac by batch', ms.groupby('batch').label_tier_b.apply(lambda s:(s=='winter_cereals').mean()).round(3).tolist())
# parcels_eu class composition for BE_WAL overall
bw=p[p.parcel_id.str.startswith('BE_WAL')]
print('parcels_eu BE_WAL n', len(bw)); print(bw.label_tier_b.value_counts().to_dict())
