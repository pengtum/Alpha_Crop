"""How much geographic information actually survives after dropping the top-K
country-predictive dimensions? Without this, a null effect on transfer is
uninterpretable: it could mean 'removing geography does not help' OR
'geography was never removed'. Same fixed test/val split as the learning curve."""
import numpy as np, json, time, gc, os
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
SEED=42; NTRAIN=90000; NEST=250
OUT='results/experiments/exp8_debias_test/country_leakage_vs_K.json'
AE=np.load('/tmp/AE.npy',mmap_mode='r'); S2=np.load('/tmp/S2.npy',mmap_mode='r')
m=np.load('/tmp/meta.npz',allow_pickle=True); ctry=m['country']
S=json.load(open('/tmp/shap_entangle.json'))
cls=np.unique(ctry); ci=np.searchsorted(cls,ctry); n=len(ci)
rest,te=train_test_split(np.arange(n),test_size=0.10,random_state=SEED,stratify=ci)
trpool,va=train_test_split(rest,test_size=0.10,random_state=SEED,stratify=ci[rest])
rng=np.random.RandomState(SEED); trpool=rng.permutation(trpool)[:NTRAIN]
tr=np.sort(trpool); va=np.sort(va); te=np.sort(te)
KG=[0,2,4,8,12,16,20,24,32,40,48,56,64]
print(f'train={len(tr):,} val={len(va):,} test={len(te):,}  K grid={KG}',flush=True)
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
t0=time.time()
for K in KG:
    for name,X in [('AlphaEarth',AE),('Sentinel-2',S2)]:
        key=f'{name}|K{K}'
        if key in res: continue
        d=S['AE' if name=='AlphaEarth' else 'S2']; nf=len(d['names'])
        drop=set(np.argsort(-np.array(d['country_imp']))[:K].tolist()) if K else set()
        cols=[i for i in range(nf) if i not in drop]
        mdl=LGBMClassifier(n_estimators=NEST,learning_rate=0.05,num_leaves=63,min_child_samples=20,
            class_weight='balanced',subsample=0.8,colsample_bytree=0.8,subsample_freq=1,
            n_jobs=-1,random_state=SEED,verbose=-1)
        mdl.fit(np.asarray(X[tr])[:,cols],ci[tr],eval_set=[(np.asarray(X[va])[:,cols],ci[va])],
                callbacks=[early_stopping(50,verbose=False)])
        ba=balanced_accuracy_score(ci[te],mdl.predict(np.asarray(X[te])[:,cols]))*100
        res[key]={'feat':name,'K':K,'country_ba':float(ba),'n_features_left':len(cols)}
        json.dump(res,open(OUT,'w'),indent=1)
        print(f'{name:12s} K={K:>3}  country BA={ba:5.2f}%  ({len(cols)} feats left)  {time.time()-t0:.0f}s',flush=True)
        del mdl; gc.collect()
print('LEAKAGE CURVE DONE %.0fs'%(time.time()-t0),flush=True)
