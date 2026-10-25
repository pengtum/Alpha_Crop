# 复现 PT random k=50 (AE) 两行的 BA，并给出各类召回（2 线程，nice）
import time, numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
from sklearn.model_selection import train_test_split
P='/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet'
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
names=pq.ParquetFile(P).schema_arrow.names
meta=pq.read_table(P,columns=['country','label','has_ae','has_s2']).to_pandas()
keep=np.flatnonzero(meta.has_ae.to_numpy()&meta.has_s2.to_numpy()); df=meta.iloc[keep].reset_index(drop=True)
classes=sorted(df.label.unique()); y=df.label.map({c:i for i,c in enumerate(classes)}).to_numpy(); ctry=df.country.to_numpy()
rng=np.random.default_rng(42)
foreign=np.concatenate([rng.choice(np.flatnonzero(ctry==c),5000,replace=False) for c in COUNTRIES if c!='PT'])
t=np.flatnonzero(ctry=='PT'); pool,test=train_test_split(t,test_size=0.3,random_state=42)
aecols=[c for c in names if c.startswith('ae_')]
need=np.r_[foreign,pool,test]
tab=pq.read_table(P,columns=aecols).take(keep[need]); A=np.column_stack([tab.column(c).to_numpy().astype(np.float32) for c in aecols]); del tab
pos={g:i for i,g in enumerate(need)}
def rows(ix): return A[[pos[i] for i in ix]]
PARAMS=dict(n_estimators=300, learning_rate=0.05, num_leaves=63, max_depth=-1, min_child_samples=20, class_weight='balanced',
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1)
csv=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/pilot_selection.csv')
Xte=rows(test); yte=y[test]
for seed in [0]:
    t0=time.time(); k=50
    pk=np.random.default_rng(1000*seed+k).choice(len(pool),k,replace=False); sel=pool[pk]
    tr=np.r_[foreign,sel]; w=np.r_[np.ones(len(foreign)),np.full(k,0.25*len(foreign)/k)]
    m=lgb.LGBMClassifier(**PARAMS,random_state=seed).fit(rows(tr),y[tr],sample_weight=w)
    pr=m.predict(Xte)
    rec={classes[c]:(round(float((pr[yte==c]==c).mean()),3),int((yte==c).sum())) for c in np.unique(yte)}
    ba=np.mean([(pr[yte==c]==c).mean() for c in np.unique(yte)])
    ref=csv[(csv.target=='PT')&(csv.strategy=='random')&(csv.budget==50)&(csv.seed==seed)&(csv.clf_feat=='ae')].ba.iloc[0]
    print(f'seed {seed}: BA={ba:.6f} csv={ref:.6f} diff={ba-ref:+.2e} ({time.time()-t0:.0f}s)')
    print('  recall(n_test):',rec)
