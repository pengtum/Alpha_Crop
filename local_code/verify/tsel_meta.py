# 任务B 审查：pilot_target_selection 的元数据检查（不训练模型）
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
P='/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet'
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
cols=['parcel_id','country','longitude','latitude','label','has_ae','has_s2']
df=pq.read_table(P,columns=cols).to_pandas()
df=df[df.has_ae]; df=df[df.has_s2]; df=df.reset_index(drop=True)
print('rows',len(df),'parcel_id unique',df.parcel_id.is_unique)
classes=sorted(df.label.unique()); print('classes',len(classes),classes)
y=df.label.map({c:i for i,c in enumerate(classes)}).to_numpy(); ctry=df.country.to_numpy()
csv=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/pilot_selection.csv')
for tgt in ['PT','SE','SK','LV']:
    rng=np.random.default_rng(42)
    foreign=np.concatenate([rng.choice(np.flatnonzero(ctry==c),5000,replace=False) for c in COUNTRIES if c!=tgt])
    t=np.flatnonzero(ctry==tgt)
    pool,test=train_test_split(t,test_size=0.3,random_state=42)
    print(f'\n== {tgt}: n={len(t)} pool={len(pool)} test={len(test)} 0.3*pool={int(0.3*len(pool))} foreign={len(foreign)} uniqF={len(np.unique(foreign))}')
    print('  overlap pool/test', len(np.intersect1d(pool,test)), ' foreign∩target', len(np.intersect1d(foreign,t)))
    fc=np.bincount(y[foreign],minlength=10); pc=np.bincount(y[pool],minlength=10); tc=np.bincount(y[test],minlength=10)
    print('  foreign cls', fc.tolist()); print('  pool cls   ', pc.tolist()); print('  test cls   ', tc.tolist())
    # 同坐标地块跨 pool/test
    ll=df.loc[t,['longitude','latitude']].round(7)
    key=ll.longitude.astype(str)+'_'+ll.latitude.astype(str)
    kp=set(key[np.isin(t,pool)]); kt=set(key[np.isin(t,test)])
    print('  same-centroid pool&test keys', len(kp&kt), ' dup centroid within target', key.duplicated().sum())
    lon=df.longitude.to_numpy()[pool]; lat=df.latitude.to_numpy()[pool]
    print('  lon range %.2f..%.2f lat %.2f..%.2f; std lon %.3f lat %.3f'%(lon.min(),lon.max(),lat.min(),lat.max(),lon.std(),lat.std()))
    if tgt=='PT':
        print('  PT parcels lon<-12 (Azores/Madeira):', (lon<-12).sum())
    # 随机策略复现 + 有效目标权重份额（class_weight=balanced 与 sample_weight 相乘）
    for k in [50,200,1000]:
        for s in [0,1,2]:
            pk=np.random.default_rng(1000*s+k).choice(len(pool),k,replace=False); sel=pool[pk]
            tr=np.r_[foreign,sel]; w=np.r_[np.ones(len(foreign)),np.full(k,0.25*len(foreign)/k)]
            cw=compute_sample_weight('balanced',y[tr]); ew=w*cw
            share=ew[len(foreign):].sum()/ew[:len(foreign)].sum()
            ncls=len(np.unique(y[sel]))
            c=csv[(csv.target==tgt)&(csv.strategy=='random')&(csv.budget==k)&(csv.seed==s)]
            print(f'  random k={k} s={s}: ncls={ncls} csv={c.n_classes_sel.tolist()} eff_share={share:.3f} (nominal 0.25)')
    both=np.r_[foreign,pool]; cw=compute_sample_weight('balanced',y[both])
    print('  foreign+target_full: raw target share %.3f, eff share after class_weight %.3f'%(len(pool)/len(foreign), cw[len(foreign):].sum()/cw[:len(foreign)].sum()))
