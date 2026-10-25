"""Country-probe learning curve: how does country recoverability depend on training size?
Fixed held-out test set (10%) and fixed validation set, so ONLY the training size varies
and every point is evaluated on identical data. Training subsamples are nested."""
import numpy as np, json, time, gc, os
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
SEED=42
OUT='results/experiments/exp8_debias_test/probe_learning_curve.json'
AE=np.load('/tmp/AE.npy',mmap_mode='r'); S2=np.load('/tmp/S2.npy',mmap_mode='r')
m=np.load('/tmp/meta.npz',allow_pickle=True); ctry=m['country']
cls=np.unique(ctry); ci=np.searchsorted(cls,ctry); n=len(ci)
# fixed test (10%) and fixed val (10% of the rest) -- identical for every point
rest,te = train_test_split(np.arange(n),test_size=0.10,random_state=SEED,stratify=ci)
trpool,va = train_test_split(rest,test_size=0.10,random_state=SEED,stratify=ci[rest])
rng=np.random.RandomState(SEED); trpool=rng.permutation(trpool)     # nested subsamples
SIZES=[8000,16000,32000,64000,128000,256000,len(trpool)]
print(f'test={len(te):,}  val={len(va):,}  train pool={len(trpool):,}',flush=True)
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
t0=time.time()
for N in SIZES:
    for name,X in [('AlphaEarth',AE),('Sentinel-2',S2)]:
        key=f'{name}|{N}'
        if key in res: continue
        tr=trpool[:N]
        Xtr=np.asarray(X[np.sort(tr)]); ytr=ci[np.sort(tr)]
        Xv=np.asarray(X[np.sort(va)]);  yv=ci[np.sort(va)]
        Xte=np.asarray(X[np.sort(te)]); yte=ci[np.sort(te)]
        mdl=LGBMClassifier(n_estimators=500,learning_rate=0.05,num_leaves=63,max_depth=-1,
            min_child_samples=20,class_weight='balanced',subsample=0.8,colsample_bytree=0.8,
            n_jobs=-1,random_state=SEED,verbose=-1)
        mdl.fit(Xtr,ytr,eval_set=[(Xv,yv)],callbacks=[early_stopping(50,verbose=False)])
        p=mdl.predict(Xte)
        ba=balanced_accuracy_score(yte,p)*100; oa=accuracy_score(yte,p)*100
        res[key]={'feat':name,'train_n':int(N),'balanced_acc':float(ba),'overall_acc':float(oa),
                  'test_n':int(len(te))}
        json.dump(res,open(OUT,'w'),indent=1)
        print(f'{name:12s} train={N:>7,}  BA={ba:5.2f}%  OA={oa:5.2f}%   ({time.time()-t0:.0f}s)',flush=True)
        del Xtr,Xv,Xte,mdl; gc.collect()
print('CURVE DONE %.0fs'%(time.time()-t0),flush=True)
