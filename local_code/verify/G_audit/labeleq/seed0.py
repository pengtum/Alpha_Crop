import sys, os, time, numpy as np, pandas as pd
os.chdir('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.path.insert(0,'.')
import guidance_G as G
from stageA import class_weights_from
D=G.GData()
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/S.pkl')
res=[]
countries=sys.argv[1].split(',')
seeds=[int(s) for s in sys.argv[2].split(',')]
for tgt in countries:
    S=dict(np.load(f'{OUT}/stage0/{tgt}_s0.npz'))
    fr,test=S['foreign'],S['test']; yT=D.y[test]; big=np.flatnonzero(np.bincount(yT,minlength=10)>=G.MIN_N)
    cw=class_weights_from(D.y[fr])
    for clf in ('s2','ae'):
        X=D.X[clf]
        logged=R[(R.target==tgt)&(R.split==0)&(R.rep==0)&(R.method=='foreign_only')&(R.clf==clf)].ba_ge30.iloc[0]
        for sd in seeds:
            t0=time.time()
            P=G.train_predict(X[fr],D.y[fr],cw[D.y[fr]],X[test],8,sd)[300]
            _,ba,_,_=G._scores(yT,P.argmax(1),big); _,ba16,_,_=G._scores(yT,P.astype(np.float16).astype(np.float64).argmax(1),big)
            res.append(dict(target=tgt,clf=clf,seed=sd,ba=ba,ba16=ba16,logged_seed0=logged,sec=round(time.time()-t0)))
            print(res[-1],flush=True)
pd.DataFrame(res).to_csv(f'/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/seed_k0_{"_".join(countries)}.csv',index=False)
