"""Does a model's feature-importance explanation survive crossing a border?
Train on one source country, then compute SHAP importance on (a) held-out source data
(in-distribution) and (b) every other country (transfer). If the ranking shifts, the
interpretation is domain-specific and cannot be carried across regions."""
import numpy as np, json, time, gc, os
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr
import shap
SEED=42; NEST=200; NSHAP=2000; NTRAIN=40000
OUT='results/experiments/exp9_shap_stability/stability_full.json'
os.makedirs(os.path.dirname(OUT),exist_ok=True)
AE=np.load('/tmp/AE.npy',mmap_mode='r'); S2=np.load('/tmp/S2.npy',mmap_mode='r')
m=np.load('/tmp/meta.npz',allow_pickle=True); ctry=m['country']; y=m['y']
aef=list(np.load('/tmp/aef.npy',allow_pickle=True)); s2f=list(np.load('/tmp/s2f.npy',allow_pickle=True))
cls=np.unique(y); yi=np.searchsorted(cls,y)
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
SOURCES=C   # all 11 countries as source -> full 11x11 stability matrix
rng=np.random.RandomState(SEED)
def imp(mdl,X,idx,nf):
    sv=np.array(shap.TreeExplainer(mdl).shap_values(np.asarray(X[np.sort(idx)])))
    if sv.ndim==3:
        ax=[i for i,s in enumerate(sv.shape) if s==nf][0]
        return np.abs(sv).mean(axis=tuple(i for i in range(3) if i!=ax))
    return np.abs(sv).mean(0)
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
t0=time.time()
for src in SOURCES:
    for fname,X,names in [('AE',AE,aef),('S2',S2,s2f)]:
        key=f'{fname}|{src}'
        if key in res: continue
        nf=len(names)
        si=np.where(ctry==src)[0]
        try:                       # stratify unless a class is a singleton (Slovakia has one)
            tr,te=train_test_split(si,test_size=0.2,random_state=SEED,stratify=yi[si])
        except ValueError:
            tr,te=train_test_split(si,test_size=0.2,random_state=SEED)
            print(f'  note: {src} has a singleton class; using an unstratified split',flush=True)
        if len(tr)>NTRAIN: tr=rng.choice(tr,NTRAIN,replace=False)
        mdl=LGBMClassifier(n_estimators=NEST,learning_rate=0.05,num_leaves=63,min_child_samples=20,
            class_weight='balanced',subsample=0.8,colsample_bytree=0.8,subsample_freq=1,
            n_jobs=-1,random_state=SEED,verbose=-1).fit(np.asarray(X[np.sort(tr)]),yi[np.sort(tr)])
        # in-distribution importance (held-out source parcels)
        # two disjoint in-distribution draws: one defines the reference importance,
        # the other gives the sampling-noise ceiling that cross-border rho must be read against
        te_sh=rng.permutation(te)
        idsub=te_sh[:min(NSHAP,len(te)//2)]
        idsub_b=te_sh[len(te)//2:len(te)//2+len(idsub)]
        v_id=imp(mdl,X,idsub,nf)
        v_id_b=imp(mdl,X,idsub_b,nf)
        rho_base=float(spearmanr(v_id,v_id_b).statistic)
        o=set(np.argsort(-v_id)[:10]); k=set(np.argsort(-v_id_b)[:10])
        rec={'source':src,'feat':fname,'id_importance':v_id.tolist(),
             'baseline_within_country':{'spearman':rho_base,'top10_overlap':len(o&k)/10,
                                        'n_per_draw':int(len(idsub)),'importance_b':[round(float(z),8) for z in v_id_b]},
             'targets':{}}
        for tgt in C:
            if tgt==src: continue
            ti=np.where(ctry==tgt)[0]
            tsub=rng.choice(ti,min(NSHAP,len(ti)),replace=False)
            v_ood=imp(mdl,X,tsub,nf)
            rho=spearmanr(v_id,v_ood).statistic
            o=set(np.argsort(-v_id)[:10]); k=set(np.argsort(-v_ood)[:10])
            ba=balanced_accuracy_score(yi[np.sort(tsub)],mdl.predict(np.asarray(X[np.sort(tsub)])))
            rec['targets'][tgt]={'spearman':float(rho),'top10_overlap':len(o&k)/10,'ba':float(ba*100),
                                 'importance':[round(float(z),8) for z in v_ood]}
        res[key]=rec; json.dump(res,open(OUT,'w'))
        rr=[v['spearman'] for v in rec['targets'].values()]
        print(f'{fname} src={src}: 基线(同国两批) rho={rho_base:.3f} | 跨境 rho={np.mean(rr):.3f} '
              f'(min {min(rr):.3f}, max {max(rr):.3f}) | 落差 {rho_base-np.mean(rr):+.3f}  {time.time()-t0:.0f}s',flush=True)
        del mdl; gc.collect()
print('DONE %.0fs'%(time.time()-t0),flush=True)
