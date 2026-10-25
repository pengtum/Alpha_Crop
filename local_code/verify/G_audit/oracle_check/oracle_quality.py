import glob, os, numpy as np, pandas as pd
import guidance_G as G
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D=G.GData()
def knn_ba(X,y,k=10):
    idx=NearestNeighbors(n_neighbors=k+1).fit(X).kneighbors(X,return_distance=False)[:,1:]
    pred=np.array([np.bincount(y[i],minlength=10).argmax() for i in idx])
    return 100*np.mean([(pred[y==c]==c).mean() for c in np.unique(y) if (y==c).sum()>=10])
def part(A,d):
    A=np.asarray(A,float)[:,:d]; return A/np.sqrt(A.var(0).sum()/d)
def bal_acc_argmax(P,y):
    p=P.argmax(1); return 100*np.mean([(p[y==c]==c).mean() for c in np.unique(y) if (y==c).sum()>=10])
def tail(X):
    r=np.linalg.norm(X-X.mean(0),axis=1); return float(r.max()/np.median(r))
rows=[]
for p in sorted(glob.glob(OUT+'/oracle/*.npz')):
    tgt,split=os.path.basename(p)[:-4].split('_s'); split=int(split)
    S=dict(np.load(f'{OUT}/stage0/{tgt}_s{split}.npz')); O=np.load(p)
    pp=G.pool_positions(S,split,0); y=D.y[S['allT'][pp]]
    ref=np.sort(np.random.default_rng(0).choice(len(pp),3000,replace=False))
    r=dict(target=tgt,split=split)
    for name,X,P in (('ae',part(S['G_ae'][pp],32),S['P_T_ae'][pp]),('s2',part(S['G_s2'][pp],32),S['P_T_s2'][pp]),('oracle',part(O['G'],10),O['P_T'])):
        P=np.asarray(P,float)
        r[f'knn_{name}']=knn_ba(X[ref],y[ref]); r[f'nll_{name}']=float(-np.log(np.clip(P[np.arange(len(y)),y],1e-300,None)).mean())
        r[f'balacc_{name}']=bal_acc_argmax(P,y); r[f'tail_{name}']=tail(X)
    rows.append(r); print(tgt,split,flush=True)
Q=pd.DataFrame(rows); Q.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_check/oracle_quality.csv',index=False)
print(Q.drop(columns=['target','split']).mean().round(3))
