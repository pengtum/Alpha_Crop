import sys, os, time, numpy as np, pandas as pd
os.chdir('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.path.insert(0,'.')
import guidance_G as G
from stageA import class_weights_from
D=G.GData()
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
tgt=sys.argv[1]; split=int(sys.argv[2]); seeds=[int(s) for s in sys.argv[3].split(',')]
S=dict(np.load(f'{OUT}/stage0/{tgt}_s{split}.npz'))
fr,test=S['foreign'],S['test']; yT=D.y[test]; big=np.flatnonzero(np.bincount(yT,minlength=10)>=G.MIN_N)
cw=class_weights_from(D.y[fr])
res=[]
for clf in ('s2','ae'):
    X=D.X[clf]
    P0=S[f'P0test_{clf}_i300'].astype(np.float64)
    _,ba_stored,_,_=G._scores(yT,P0.argmax(1),big)
    for sd in seeds:
        t0=time.time()
        P=G.train_predict(X[fr],D.y[fr],cw[D.y[fr]],X[test],3,sd)[300]
        _,ba,_,_=G._scores(yT,P.argmax(1),big)
        res.append(dict(target=tgt,split=split,clf=clf,seed=sd,ba=ba,ba_stored_seed0=ba_stored,sec=round(time.time()-t0)))
        print(res[-1],flush=True)
pd.DataFrame(res).to_csv(f'/Users/perng_mini/AlphaCrops_local/verify/G_audit/labelscale/seed_k0_{tgt}_s{split}.csv',index=False)
