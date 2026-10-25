"""Fine-grained separability sweep: remove the top-K country-predictive dimensions
for K = 0,2,...,40 and re-train the crop classifier. Random-drop controls on a
coarser grid. Same protocol as the coarse run (100k train, 250 trees) so the two
are directly comparable. Writes incrementally so partial results stay usable."""
import numpy as np, json, time, gc, os
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
SEED=42; NTRAIN=100000; NEST=250
OUT='results/experiments/exp8_debias_test/sweep.json'
AE=np.load('/tmp/AE.npy',mmap_mode='r'); S2=np.load('/tmp/S2.npy',mmap_mode='r')
m=np.load('/tmp/meta.npz',allow_pickle=True); ctry=m['country']; y=m['y']
S=json.load(open('/tmp/shap_entangle.json'))
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
cls=np.unique(y); yi=np.searchsorted(cls,y)
def fit_eval(X,cols,tr,te):
    r=np.random.RandomState(SEED); tr=r.choice(tr,min(NTRAIN,len(tr)),replace=False)
    Xtr=np.asarray(X[tr])[:,cols]; Xte=np.asarray(X[te])[:,cols]
    a,v,ya,yv=train_test_split(Xtr,yi[tr],test_size=0.1,random_state=SEED,stratify=yi[tr])
    mdl=LGBMClassifier(n_estimators=NEST,learning_rate=0.05,num_leaves=63,min_child_samples=20,
        class_weight='balanced',subsample=0.8,colsample_bytree=0.8,subsample_freq=1,
        n_jobs=-1,random_state=SEED,verbose=-1).fit(a,ya)
    ba=balanced_accuracy_score(yi[te],mdl.predict(Xte))
    del Xtr,Xte,a,v,mdl; gc.collect(); return float(ba)
def run(X,cols):
    r=np.random.RandomState(SEED); mask=r.rand(len(yi))<0.8
    idb=fit_eval(X,cols,np.where(mask)[0],np.where(~mask)[0])
    oods=[fit_eval(X,cols,np.where(ctry!=c)[0],np.where(ctry==c)[0]) for c in C]
    return idb,float(np.mean(oods)),dict(zip(C,oods))
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
t0=time.time()
# Dense where the action is (0-24), coarser out to 40, then a tail to show the decay.
KGRID = [16, 20, 24, 32, 40, 48, 56, 64]
RAND_K=[22, 38, 56]
jobs=[]
for K in KGRID:                              # K-major: AE and S2 side by side at every K
    for f in ['AE','S2']: jobs.append((f,'top',K))
    if K in RAND_K:
        for f in ['AE','S2']: jobs.append((f,'random',K))
for f,mode,K in jobs:
    key=f'{f}|{mode}|K{K}'
    if key in res: continue
    X = AE if f=='AE' else S2
    d=S[f]; nf=len(d['names'])
    if K==0: drop=set()
    elif mode=='top': drop=set(np.argsort(-np.array(d['country_imp']))[:K].tolist())
    else: drop=set(np.random.RandomState(7+K).choice(nf,K,replace=False).tolist())
    cols=[i for i in range(nf) if i not in drop]
    idb,ood,per=run(X,cols)
    res[key]={'id':idb,'ood':ood,'per_country':per,'K':K,'mode':mode,'feat':f,
              'n_features':nf,'frac_removed':K/nf}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'{key:16s} ID={idb*100:5.2f} OOD={ood*100:5.2f}  [{len(res)}/{len(jobs)}] {time.time()-t0:.0f}s',flush=True)
print('SWEEP DONE %.0fs'%(time.time()-t0),flush=True)
