"""Upper-bound reference arm: remove the top-K CROP-predictive dimensions and measure
how much in-distribution crop accuracy is lost. Together with the random arm (lower
bound) this calibrates the country-ranked arm: close to random => separable,
close to crop-ranked => the same dimensions carry both signals."""
import numpy as np, json, time, gc, os
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
SEED=42; NTRAIN=90000; NEST=250
OUT='results/experiments/exp8_debias_test/crop_ranked_arm.json'
AE=np.load('/tmp/AE.npy',mmap_mode='r'); S2=np.load('/tmp/S2.npy',mmap_mode='r')
m=np.load('/tmp/meta.npz',allow_pickle=True); y=m['y']
S=json.load(open('/tmp/shap_entangle.json'))
cls=np.unique(y); yi=np.searchsorted(cls,y); n=len(yi)
rest,te=train_test_split(np.arange(n),test_size=0.10,random_state=SEED,stratify=yi)
trpool,va=train_test_split(rest,test_size=0.10,random_state=SEED,stratify=yi[rest])
rng=np.random.RandomState(SEED); trpool=rng.permutation(trpool)[:NTRAIN]
tr=np.sort(trpool); va=np.sort(va); te=np.sort(te)
KG=[0,2,4,8,12,16,20,24,32,40,48,56,64]
print(f'train={len(tr):,} val={len(va):,} test={len(te):,}  K={KG}',flush=True)
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
t0=time.time()
for K in KG:
    for name,X in [('AlphaEarth',AE),('Sentinel-2',S2)]:
        for rank in ['crop','country','random']:
            key=f'{name}|{rank}|K{K}'
            if key in res: continue
            d=S['AE' if name=='AlphaEarth' else 'S2']; nf=len(d['names'])
            if K==0: drop=set()
            elif rank=='crop':    drop=set(np.argsort(-np.array(d['crop_imp']))[:K].tolist())
            elif rank=='country': drop=set(np.argsort(-np.array(d['country_imp']))[:K].tolist())
            else:                 drop=set(np.random.RandomState(7+K).choice(nf,K,replace=False).tolist())
            cols=[i for i in range(nf) if i not in drop]
            mdl=LGBMClassifier(n_estimators=NEST,learning_rate=0.05,num_leaves=63,min_child_samples=20,
                class_weight='balanced',subsample=0.8,colsample_bytree=0.8,subsample_freq=1,
                n_jobs=-1,random_state=SEED,verbose=-1)
            mdl.fit(np.asarray(X[tr])[:,cols],yi[tr],
                    eval_set=[(np.asarray(X[va])[:,cols],yi[va])],
                    callbacks=[early_stopping(50,verbose=False)])
            ba=balanced_accuracy_score(yi[te],mdl.predict(np.asarray(X[te])[:,cols]))*100
            res[key]={'feat':name,'rank':rank,'K':K,'crop_id_ba':float(ba)}
            json.dump(res,open(OUT,'w'),indent=1)
            print(f'{name:12s} {rank:8s} K={K:>3}  crop ID BA={ba:5.2f}%  {time.time()-t0:.0f}s',flush=True)
            del mdl; gc.collect()
            if K==0: break     # K=0 identical for all three rankings
print('CROP-RANKED ARM DONE %.0fs'%(time.time()-t0),flush=True)
